#!/usr/bin/env python
# coding: utf-8


import gensim
# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


print([method for method in dir(model)])

print(len(model.vectors), model.vector_size)


def make_dataset(model):
    """Make dataset from pre-trained Word2Vec model.
    Paramters
    ---------
    model: gensim.models.word2vec.Word2Vec
        pre-traind Word2Vec model as gensim object.
    Returns
    -------
    numpy.ndarray((vocabrary size, vector size))
        Sikitlearn's X format.
    """
    V = model.index2word
    X = np.zeros((len(V), model.vector_size))

    for index, word in enumerate(V):
        X[index, :] += model[word]
    return X


google_list = make_dataset(model)



print(google_list)


import numpy
numpy.save('F:/VIGHNESH/Cognitive Computing/Final Project/google-news-numpy.npy',google_list)


for i,k in enumerate(model.vocab):
    if(i > 10):
        break
    print(k)





