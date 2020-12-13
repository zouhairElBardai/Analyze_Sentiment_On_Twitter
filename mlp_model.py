from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

data_df = pd.read_csv('data/final.csv')
# remove the "Neutral" class
data_df = data_df[data_df['sentiment'] != "neutral"]

# change values to numeric
data_df['sentiment'] = data_df['sentiment'].map({'positive': 1, 'negative': 0})

# idneitfy the data and the labels
data = data_df['text']
target = data_df['sentiment']

data_df = data_df.dropna()

# Use TfidfVectorizer for feature extraction (TFIDF to convert textual data to numeric form):
tf_vec = TfidfVectorizer()
X = tf_vec.fit_transform(data)

# Training Phase
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.50, random_state=0)



mlp =MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(mlp.predict(X_test))
print(classification_report(y_test,predictions))

with open("models/arabic_sentiment_NN_tokenizer.pickle", "wb") as f:
    pickle.dump(tf_vec, f)
with open("models/arabic_sentiment_NN.pickle", "wb") as f:
    pickle.dump(mlp, f)