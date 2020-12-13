import pickle

from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model



class Inference(object):

    def __init__(self):
 
        """
        The models are preloaded so that it wont take time during inference
        """
        self.lstm_model = load_model('models/arabic_sentiment_lstm.hdf5')
        with  open("models/arabic_sentiment_lstm.pickle", "rb") as f:
            self.tokenizer = pickle.load(f)
        with  open("models/arabic_sentiment_svm.pickle", "rb") as f:
            self.svm_model = pickle.load(f)
        with  open("models/arabic_sentiment_svm_tokenizer.pickle", "rb") as f:
            self.svm_tfidf = pickle.load(f)
        with  open("models/arabic_sentiment_NN.pickle", "rb") as f:
            self.NN_model = pickle.load(f)
        with  open("models/arabic_sentiment_NN_tokenizer.pickle", "rb") as f:
            self.NN_tfidf = pickle.load(f)         
        self.rnn_model = load_model('models/arabic_sentiment_rnn.hdf5')
        with  open("models/arabic_sentiment_rnn.pickle", "rb") as f:
            self.tokenizer = pickle.load(f)
               
        self.lstm_bid_model = load_model('models/arabic_sentiment_lstm_bid.hdf5')
        with  open("models/arabic_sentiment_lstm_bid.pickle", "rb") as f:
            self.tokenizer = pickle.load(f)     
            
            
            
            
    def get_sentiment(self, df, model):
        
        
        """
        Takes a text input that you want to run sentiment analysis on.
        Returns with sentiment score and sentiment class (positive or negative)

        :param text_input: Text to run sentiment analysis on
        :return: (sentiment_score, sentiment_class)
        """

        if model == 'LSTM':
            sequences = self.tokenizer.texts_to_sequences(df['tweet'])
            data = pad_sequences(sequences, maxlen=100) 
            num_class = self.lstm_model.predict(data)
            df['sentiment_score'] = num_class
     
        elif model == 'SVM':
            data = df['tweet']
            X = self.svm_tfidf.transform(data)
            num_class = self.svm_model.predict_proba(X)
            df['sentiment_score'] = [num[1] for num in num_class]
        elif model == 'MLP':
            data = df['tweet']
            X = self.NN_tfidf.transform(data)
            num_class = self.NN_model.predict_proba(X)
            df['sentiment_score'] = [num[1] for num in num_class]
        elif model == 'RNN':
            sequences = self.tokenizer.texts_to_sequences(df['tweet'])
            data = pad_sequences(sequences, maxlen=100)
            num_class = self.rnn_model.predict(data)
            df['sentiment_score'] = num_class
        elif model == 'LSTM bidirectionnelle':
            sequences = self.tokenizer.texts_to_sequences(df['tweet'])
            data = pad_sequences(sequences, maxlen=100)
            num_class = self.lstm_bid_model.predict(data)
            df['sentiment_score'] = num_class   
            
       
        def score_segregate(value):
            if value <= 0.35:
                return 'Negative'
            elif value > 0.35 and value < 0.65:
                return 'Neutral'
            elif value >= 0.65:
                return 'Positive'

        df['sentiment_class'] = df['sentiment_score'].apply(score_segregate)

        return df


def main():
    """
    To test the classifier
    """
    import pandas as pd
    df = Inference().get_sentiment(pd.read_csv('corona.csv'), 'svm')
    df


if __name__ == '__main__':
    main()
