"""# Step 8: ML Classification"""

import pickle

import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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

"""# SVM Classifier"""
# create the classifer and fit the training data and lables
classifier_svm = svm.SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)

print("SVM accuracy: %.2f"%classifier_svm.score(X_test, y_test))
#
#
# #do a 10 fold cross-validation
# results_svm = cross_val_score(classifier_svm, X,target, cv=10)
# print("\n10-fold cross-validation:")
# print(results_svm)
#
# print("The average accuracy of the SVM classifier is : %.2f" % np.mean(results_svm))
#
# print("\nConfusion matrix of the SVM classifier:")
# predicted_svm = classifier_svm.predict(X_test)
# print(confusion_matrix(y_test,predicted_svm))
#
# print("\nClassification_report of SVM classifier:")
# print(classification_report(y_test,predicted_svm))
# print("----------------------------------------------------------------------------")



