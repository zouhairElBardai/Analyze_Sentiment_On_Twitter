# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:29:31 2020

@author: youness hnida
"""

import codecs
import pickle
import re

import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense, LSTM ,Bidirectional 
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.2


def get_stop_words():
    path = "data/stop_words.txt"
    stop_words = []
    with codecs.open(path, "r", encoding="utf-8", errors="ignore") as myfile:
        stop_words = myfile.readlines()
    stop_words = [word.strip() for word in stop_words]
    return stop_words


def get_text_sequences(texts):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=100)
    return data, word_index


# Clean/Normalize Arabic Text

def clean_str(text):
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t',
              '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ',
               ' ! ']

    # remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, "", text)

    # remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)

    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    # trim
    text = text.strip()

    return text


df = pd.read_csv("data/final.csv")
## Clean and drop stop words
df['text'] = df.text.apply(lambda x: clean_str(x))
stop_words = r'\b(?:{})\b'.format('|'.join(get_stop_words()))
df['text'] = df['text'].str.replace(stop_words, '')
df['binary_sentiment'] = df.sentiment.map(dict(positive=1, negative=0))
df = shuffle(df)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['binary_sentiment'], test_size=0.20, random_state=42)

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(X_train)

sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)
data = pad_sequences(sequences, maxlen=100)
test_data = pad_sequences(test_sequences, maxlen=100)

# Model defnition
model_lstm_bid = Sequential()

model_lstm_bid.add(Embedding(20000, 100, input_length=100))

model_lstm_bid.add (Bidirectional (LSTM (100, dropout=0.2, recurrent_dropout=0.2)))

model_lstm_bid.add (BatchNormalization())

model_lstm_bid.add(Dense(1, activation='sigmoid'))

checkpoint = ModelCheckpoint('models/arabic_sentiment_lstm_bid.hdf5', monitor='val_acc',verbose=1,save_best_only=True, mode='max')

model_lstm_bid.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model_lstm_bid.fit(data, y_train, validation_split=0.4, epochs=3, batch_size=32,)

model_lstm_bid.summary()








def plot_history(model):
    acc = model.history['accuracy']
    val_acc = model.history['val_accuracy']
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, "b", label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

with open("models/arabic_sentiment_lstm_bid.pickle", "wb") as f:
    pickle.dump(tokenizer, f)
model_lstm_bid.save('models/arabic_sentiment_lstm_bid.hdf5')
