# encoding=utf-8

from __future__ import print_function, division
import dataInput
from hyperparams import Hyperparams as params
import keras
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Input
from keras.layers import Convolution1D, MaxPooling1D, LSTM
from keras.layers import AveragePooling1D, GlobalAveragePooling1D, Bidirectional
from keras.layers.embeddings import Embedding
from keras import optimizers, losses
import RNA2Vec
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras import regularizers
import numpy as np
import os
import pylab as plt
from PyramidPooling import PyramidPooling
import scipy.io as io
from sklearn import metrics
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

# hyperparam
cv_fold_num = params.cv_fold_num
batch_size = params.batch_size
epoch_num = params.epoch_num
seqlen = params.seqLength
pathlen = params.path_Length
k_mer = params.k_mer
lrate = params.learning_rate
specify_threshold = params.specify_threshold
save_dir = os.path.join(os.getcwd(), 'saved_models')

def paddingTruncate_sequence(seqSet, max_len=seqlen, repkey='N'):    # padding or truncate sequence prefix
    seqList = []
    for seqTemp in seqSet:
        seqTemp = seqTemp.replace('T', 'U')
        if len(seqTemp) <= max_len:
            seq_len = len(seqTemp)
            new_seq = seqTemp
            if seq_len < max_len:
                gap_len = max_len - seq_len
                new_seq = repkey * gap_len + seqTemp
        elif len(seqTemp) > max_len:
            new_seq = seqTemp[0:max_len]
        seqList.append(new_seq)
    return seqList

def paddingDiseasePath(pathSet, max_len=pathlen, repkey=int(0)):
    pathList =[]
    for path in pathSet:
        if len(path) <= pathlen:
            path_len = len(path)
            new_path = path
            if path_len < max_len:
                gap_len = max_len - path_len
                [path.insert(0, repkey) for i in range(gap_len)]
                new_path = path
        elif len(path) > max_len:
            new_path = path[0:max_len]
        pathList.append(new_path)
    return pathList

def getCircList(train_set):
    seqList = []
    [seqList.append(circFaDict[str(i[0] + 1)]) for i in train_set]
    return seqList

def getDisList(train_set):
    disList = []
    [disList.append(i[1] + 1) for i in train_set]
    return disList

def getCircVec(seqSet, nn_dict):
    trids = RNA2Vec.get_trids()
    k = len(trids[0])
    vecSet=[]
    for seq in seqSet:
        vec_index = []
        for x in range(len(seq) + 1 - k):
            kmer = seq[x:x + k]
            if kmer in trids:
                ind = trids.index(kmer)
                vec_index.append(nn_dict[str(ind)]+1)
            else:
                vec_index.append(0)
        vecSet.append(vec_index)
    return vecSet

def getKmerEncode (seqList,seqLength, nn_dict):
    circSeqSetFixedLength = paddingTruncate_sequence(seqList, seqLength)
    selected_circSeqKmerEncode = getCircVec(circSeqSetFixedLength, nn_dict)
    return selected_circSeqKmerEncode

def getDisPathVec(pathSet, doid_dict):
    vecSet = []
    for path in pathSet:
        vec_index = []
        for i in path:
            if i in doid_dict.keys():
                vec_index.append(doid_dict[i] + 1)
            else:
                vec_index.append(0)
        vecSet.append(vec_index)
    return vecSet

def getDisEncode(disList, doid_dict, pathlen):
    tmpPath = []
    [tmpPath.append(dis_path[str(i)]) for i in disList]
    tpath = getDisPathVec(tmpPath, doid_dict)
    selected_disEncode = paddingDiseasePath(tpath,pathlen)
    return selected_disEncode

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

dataInput = dataInput_multiFeaGloVe.DataLoader() # initialize positive and negative samples
circFaDict, kMerWord2vec, dis_path, disVec = dataInput.getPretrained()

nn_dict = dataInput.read_dict('data/gloveSeqkm5dim128Win3_vocab.txt')
doid_dict = dataInput.read_dict('data/gloveW4dim100_vocab.txt')
num_vocab, embeddingAdd = dataInput.get_embed(kMerWord2vec)
num_doid, disWordEmbedding = dataInput.get_embed(disVec)

print(('begin training:').center(50, '='))

for cv_fold_id in range(cv_fold_num):
    dataInput.shuffleMakTraining(cv_fold_id)
    print(('The %d fold:').center(50, '='), cv_fold_id)
    # model architecture
    seq1_input = Input(shape=(seqlen - k_mer + 1,), dtype='float32', name='seq1_input')
    x1 = Embedding(input_length=(seqlen - k_mer + 1), input_dim=num_vocab, output_dim=128,
                   weights=[embeddingAdd],
                   trainable=True)(seq1_input)
    x1 = Convolution1D(filters=64, kernel_size=4, strides=3, padding='valid', activation='relu')(x1)
    x1 = Bidirectional(LSTM(64))(x1)  # input-3d
    x1 = Dense(32)(x1)
    x1 = Activation('relu')(x1)
    x1 = Dense(16)(x1)
    x1 = Dropout(0.5)(x1)
    seq1_output = Activation('relu')(x1)


    seq2_input = Input(shape=(88,), dtype='float32', name='seq2_input')
    x2 = Dense(64)(seq2_input)
    seq2_output = Activation('relu')(x2)
    dis1_input = Input(shape=(None,), dtype='float32', name='dis_input')
    x3 = Embedding(input_dim=num_doid, output_dim=100, weights=[disWordEmbedding], trainable=True)(dis1_input)
    x3 = Bidirectional(LSTM(64))(x3)  # input-3d
    x3 = Dense(32)(x3)
    x3 = Activation('relu')(x3)
    x3 = Dropout(0.5)(x3)
    x3 = Dense(16)(x3)
    dis1_output = Activation('relu')(x3)

    dis2_input = Input(shape=(630,), dtype='float32', name='dis2_input')
    x4 = Dense(128)(dis2_input)
    x4 = Activation('relu')(x4)
    x4 = Dense(64)(x4)
    x4 = Dropout(0.5)(x4)
    dis2_output = Activation('relu')(x4)
    x12 = keras.layers.concatenate([seq1_output, seq2_output])
    y12 = keras.layers.concatenate([dis1_output, dis2_output])
    x = keras.layers.concatenate([x12, y12])
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    output = Activation('sigmoid')(x)

    model = Model(inputs=[seq1_input, seq2_input, dis1_input, dis2_input], output=output)
    # model.summary()

    adadelta = optimizers.Adadelta(lr=lrate, epsilon=1e-6, decay=1e-6, rho=0.9)
    model.compile(loss=losses.binary_crossentropy, optimizer=adadelta, metrics=['accuracy'])

    y_label = []
    y_pred = []
    train_set = dataInput.train_set
    seqList_train = getCircList(train_set)
    disList_train = getDisList(train_set)
    selected_circSeqKmerEncode = getKmerEncode(seqList_train, seqlen, nn_dict)
    selected_disEncode = getDisEncode(disList_train, doid_dict, pathlen)
    Xc_fea, Xd_fea = dataInput.interaction_fea(train_set)

    x_tr = [np.array(selected_circSeqKmerEncode), np.array(Xc_fea), np.array(selected_disEncode), np.array(Xd_fea)]
    y_tr = np.asarray([np.asarray(row, dtype=float) for row in dataInput.train_label], dtype=float)

    checkpointer = ModelCheckpoint(filepath="./saved_models/seql%d_pathl%d_lr%.2f_{accuracy:.2f}.hdf5"
                                   % (seqlen, pathlen, lrate), verbose=1, save_best_only=True, mode='auto',monitor='accuracy')

    earlystopper = EarlyStopping(monitor='accuracy', patience=25, verbose=2)
    callbacks_list = [checkpointer, earlystopper]

    model.fit(x_tr, y_tr, batch_size=batch_size, epochs=epoch_num, callbacks=callbacks_list, verbose=1)
    test_set = dataInput.test_set
    seqList_test = getCircList(test_set)
    disList_test = getDisList(test_set)
    selected_circSeqKmerEncodeTest = getKmerEncode(seqList_test, seqlen, nn_dict)
    selected_disEncodeTest = getDisEncode(disList_test, doid_dict, pathlen)
    XcTest_fea, XdTest_fea = dataInput.interaction_fea(test_set)
    x_test = [np.array(selected_circSeqKmerEncodeTest), np.array(XcTest_fea), np.array(selected_disEncodeTest),
              np.array(XdTest_fea)]
    y_test = np.asarray([np.asarray(row, dtype=float) for row in dataInput.test_label], dtype=float)
    loss, acc = model.evaluate(x_test, y_test)
    pred = model.predict(x_test)   
