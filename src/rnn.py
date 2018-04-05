import sys
reload(sys) 

# sys.setdefaultencoding('utf8')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections #Used to count word frequency
import nltk #Used to Posing tagging
import numpy as np

## Step 1
'''
Before starting, make a preliminary exploration of the data used. 
In particular, we need to know 
how many different words in the data and How many words make up every word.
'''
maxlen = 0                              #The maximum sentence length
word_freqs = collections.Counter()      #word frequency
num_recs = 0                            #Number of samples
with open('yelp_dataset_all.txt','r+') as f:
    for line in f:
        label = line[0]
        sentence = line[2:]
        # label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
print('The maximum sentence length is: ',maxlen)
print('The number of words is: ', len(word_freqs))
print('The number of samples is: ', num_recs)


'''
Depending on the number of different words (nb_words), 
we can set the size of the vocabulary to a fixed value, 
and for words that are not in the vocabulary, replace them with the pseudo-word UNK. 
According to the maximum length of the sentence (max_lens), 
we can unify the length of the sentence, the short sentence filled with 0.
'''
## Prepare the data
'''
As mentioned earlier, we set VOCABULARY_SIZE to 30002. 
Contains the first 30000 words in the training data sorted by descending order of words, 
plus a dummy word UNK and padding word 0. 
The maximum sentence length MAX_SENTENCE_LENGTH set to 500.
'''
MAX_FEATURES = 30000
MAX_SENTENCE_LENGTH = 500
'''
Next, create two lookup tables, word2index and index2word, for word and number conversion
'''
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}
'''
The following is based on the lookup table to convert the sentence into a sequence of numbers, 
and the length of the uniform MAX_SENTENCE_LENGTH, not enough fill 0, more cut off.
'''
X = np.empty(num_recs,dtype=list)
y = np.zeros(num_recs)
i=0
with open('yelp_dataset_all.txt','r+') as f:
    for line in f:
        label = line[0]
        sentence = line[2:]
        # label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i] = seqs
        y[i] = int(label)
        i += 1
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

'''
Finally, divide the data, 80% as training data, 20% as the test data.
'''
## Data division
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

## Network construction
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 6

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

## Network training
model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))

## prediction
score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)

print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('Prediction','True','Sentence'))
for i in range(50):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,500)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
    print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))
    