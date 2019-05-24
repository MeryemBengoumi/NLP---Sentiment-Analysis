import numpy as np
import pandas as pd
import string
import re
import tokenize
import nltk 
import time
import unidecode

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, Flatten, Dropout, Dense, LSTM, Conv2D

from keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from sklearn.metrics import accuracy_score, f1_score

import spacy
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA


def clean(w): 
  #Cleaning the dataset by removing spaces, accents, etc. 
  w = re.sub(r'[\']+', '', w.lower())
  w = re.sub(r'[^a-zA-Z0-9]+', ' ', w)
  return unidecode.unidecode(w)

def read_data(file): 
  df = pd.read_csv(file, sep='\t', header= None)
  df.columns = ['polarity', 'aspect_category', 'target_term', 'character_offsets_term', 'sentence']
  df.sentence = df.sentence.apply(clean)
  return df

def polarity_terms_2(dataset, nlp):
  polarity_terms = []
  i = 0 
  for sentence in nlp.pipe(dataset['sentence']):
    if sentence.is_parsed:
        polarity_terms.append(' '.join(
            [token.lemma_ for token in sentence 
                if (not token.is_stop and not token.is_punct 
                    and (token.pos_ == "ADJ" or token.pos_ == "VERB")
                    and dist(token, dataset.category_terms_found[i], sentence) < 3)]))
    else:
        polarity_terms.append('')
    i += 1
  dataset['polarity_terms'] = polarity_terms
  
def dist(polarity, aspect_terms, sentence):
  aspect_terms = aspect_terms.split()
  sentence = sentence.text.split()
  aspect_terms = [word for word in aspect_terms if len(word) > 0]
  a = [abs(sentence.index(polarity.text) - sentence.index(word)) for word in aspect_terms if polarity.text in sentence and word in sentence]
  if len(a) == 0:
    return 0
  return min(a)

def category_terms(dataset, nlp):
    category_terms = []
    for sentence in nlp.pipe(dataset.sentence):
      chunks = [(chunk.root.text) for chunk in sentence.noun_chunks if (chunk.root.pos_ == 'NOUN' and not chunk.root.is_stop)]
      category_terms.append(' '.join(chunks))
    dataset['category_terms_found'] = category_terms


#Create the Y dataset to use an input for the model

def create_Y_dataset(df, encoder):
      #our labels are composed of the dummy features representing the 3 polarities: "positive", "negative", or "neutral"
      value_polarity = encoder.fit_transform(df.polarity)
      dum_polarity =  to_categorical(value_polarity)
      return dum_polarity


#Create the X dataset to use an input for the model
def create_X_dataset(df, nlp, tokenizer):
    #Our features are composed of the tokenized polarity words
    polarity_terms_2(df, nlp)
    polarity_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(df.polarity_terms))
    return polarity_tokenized

class Classifier:
    """The Classifier"""
    def __init__(self):
        self.trained = False
    
    
    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        
        
        nlp = spacy.load('en_core_web_sm')
        self.nlp = nlp
        
        print("#### Training ###")
    
        
        #Reading dataset
        print("Reading dataset...")
        df_train = read_data(trainfile)
        self.df_train = df_train
        
        #Loading tokenizer
        print("Loading Tokenizer...")
        tokenizer = Tokenizer(num_words=6000)
        tokenizer.fit_on_texts(df_train.sentence)
        
        self.tokenizer = tokenizer
        
        print("Computing X_train...")
        nlp = spacy.load('en_core_web_sm')
        category_terms(df_train, nlp)
        X_train = create_X_dataset(df_train, nlp, tokenizer)
        self.X_train = X_train
        
        print("Computing Y_train...")
        encoder = LabelEncoder()
        Y_train = create_Y_dataset(df_train, encoder)
        
        print("Defining a Deep Learning Network...")
        
        #For the Model, we decided to use a succession of Dense Layer
        #As it is the model that gives us the best score on dev set 
        model = Sequential()

        model.add(Dense(1024, input_shape=(6000,), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        print("Fitting the model ...")
        model.fit(X_train, Y_train, epochs=5, verbose=1)
        self.Y_train = Y_train
        self.model = model
        self.trained = True
        
#Below is a model with LSTM Neural Network that we tried, but that gives lower scores than the model above
        
#    def train_lstm(X_train, Y_train):
#        embed_dim = 128
#        lstm_out = 200
#    
#        model = Sequential()
#        model.add(Embedding(512, embed_dim,input_length = X_train.shape[1]))
#        model.add(LSTM(lstm_out))
#        model.add(Dense(3,activation='softmax'))
#        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
#        print(model.summary())
#        model.fit(X_train, Y_train, epochs=2, verbose=1)
#        model.save('model.lstm')
#        
#        print(model.summary())
#        return model


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        print("#### Predicting ###")
        
        nlp = spacy.load('en_core_web_sm')
        self.nlp = nlp
        #Reading dataset
        print("Reading dataset...")
        df_test = read_data(datafile)
        category_terms(df_test, nlp)
        X_test = create_X_dataset(df_test, nlp, self.tokenizer)
        self.X_test = X_test
        
        Y_true = df_test.polarity
        encoder = LabelEncoder()
        
        
        Y_prediction = self.model.predict_classes(X_test)
        transform = encoder.fit_transform(self.df_train.polarity)
        Y_prediction = encoder.inverse_transform (Y_prediction)
        
        self.Y_true = Y_true
        self.Y_prediction = Y_prediction
  
        accuracy = accuracy_score(Y_true, Y_prediction)
        print("accuracy score: {}".format(accuracy))
        
        return Y_prediction
    

