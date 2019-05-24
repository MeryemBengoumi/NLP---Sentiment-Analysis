NLP Exercise 2: Aspect Based Sentiment Analysis

Student who took part in this assignment:

-     Meryem BEN-GOUMI <meryem.ben-goumi@student-ecp.fr>
-     Sarah BOURIAL <sarah.bourial@student-cs.fr>
-     Gabrielle RAPPAPORT <gabrielle.rappaport@student-ecp.fr>
-     Josiak KAYO-KOUAKAM <josias.kayo-kouakam@student-ecp.fr>

Description of our delivrable

	a) Feature representation
  To prepare our dataset for training, we limited the preprocessing so as not to modify the structure of the sentence and alter the spaCy   detection. We first got rid of all punctuations and turned all words into lower case. We then extracted the category terms 
  (words in the sentences that relate to the review category). We encoded our category terms as vectors using the BOW method of the spaCY   dependency parser (as as we cannot include them as strings into the model. 
  
  We then extracted polarity terms (essentially the words translating the main sentiment of the review) and computed the distance 
  between remanings words in the sentences and our main sentiment term. We used spaCy dependency for a Part Of Speech (POS) Tagging 
  to filter and keep only adjectives and verbs which we assumed to best translate the connotation of the review sentiment. The same 
  BOW word embedding as before was used here to feed our model with this data.
  
  Finally, we turned our polarity levels into categorical values, again to make the training smoother (positive: 1; neutral: 0; negative:   -1).
  
        b) Classifier
  We tried three models:
  - Neural network with dense layers, made with fully connected layers characterized by a linear operation on the layer’s input vector.
  - 1D Convolutional neural network, essentially a layer that consists of a set of “filters”. The filters take a subset of the input data     at a time, but are applied across the full input (by sweeping over the input). The operations performed by this layer are still           linear/matrix multiplications, but they go through an activation function at the output, which is usually a non-linear operation.
  - Long Short Term Memory networks (LSTMs), a special kind of RNN, capable of learning long-term dependencies. 
  
  Our final ABSA implementation was done with a series of Dense Layers.

  
        c) Results:
  Our best accuracy was achieved with a series of Dense Layers which led to an accuracy of 0.75.
  
  We also trained an LSTM on our classifier, however the results were less satisfying generating an accuracy of 0.70 only. 
  
  An interesting extension of our work would be attempting to train our model with an svm-SVC model which we believe could lead to 
  an increased accuracy. Another extension would be computing a Vader polarizer score (it did not seem to be allowed for this assignment
  on the reviews before feeding the data into our model, of which probabilities should lead to better predictions on the 'neutral' reviews
  which was the polarity level that was the least well predicted by our model. 
