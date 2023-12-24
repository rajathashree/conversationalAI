
import nltk
#nltk.download('punkt')

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import tflearn
import tensorflow as tf
import random
import numpy
from tensorflow.python.framework import ops

import json
with open('HarzI.json') as jsonData:
    intents = json.load(jsonData)

words = []
doc = []
classes = []
ignored = ['?']

for intent in intents['intents']:
    for patterns in intent['patterns']:

        word = nltk.word_tokenize(patterns)

        words.append(word)

        doc.append((word, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(word.lower()) for word in words if word in ignored]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))


print(classes, doc, words)

training = []
output = []
output_empty = [0]*len(classes)

for doc in doc:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    output_row =list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = numpy.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])



# reset underlying graph data

ops.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

       