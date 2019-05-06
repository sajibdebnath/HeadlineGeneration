#!/usr/bin/env python
# coding: utf-8

import numpy
google_news_numpy = numpy.load('google-news-numpy.npy')


google_news_numpy = numpy.float32(google_news_numpy)
print(google_news_numpy.shape)
length_vocab, embedding_size = google_news_numpy.shape

import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Activation
from keras.utils import np_utils
from keras.preprocessing import sequence


max_len_head = 25
max_len_desc = 50
max_length = max_len_head + max_len_desc
rnn_layers = 4
rnn_size = 600
# first 40 numebers from hidden layer output used for
# simple context calculation
activation_rnn_size = 50

empty_tag_location = 0
eos_tag_location = 1
unknown_tag_location = 2
learning_rate = 1e-4

#minimum headline should be genrated
min_head_line_gen = 10
dont_repeat_word_in_last = 5

rnn_model = Sequential()

# TODO: look at mask zero flag
rnn_model.add(
        Embedding(
                length_vocab, embedding_size,
                input_length=max_length,
                weights=[google_news_numpy], mask_zero=True,
                name='embedding_layer'
        )
)

for i in range(rnn_layers):
    lstm = LSTM(rnn_size, return_sequences=True,
        name='lstm_layer_%d' % (i + 1)
    )
    rnn_model.add(lstm)
    # No drop out added !
'''
model.add(Lambda(self.simple_context,
                mask=lambda inputs, mask: mask[:, max_len_desc:],
                output_shape=self.output_shape_simple_context_layer,
                name='simple_context_layer'))

'''

rnn_model.add(TimeDistributed(Dense(length_vocab,
                        name='time_distributed_layer')))
        
rnn_model.add(Activation('softmax', name='activation_layer'))
        
rnn_model.compile(loss='categorical_crossentropy', optimizer='adam')
K.set_value(rnn_model.optimizer.lr, np.float32(learning_rate))
print (rnn_model.summary())


