from hyperparamsRL import Hyperparams as params
import numpy as np
import pickle
import random

random.seed(params.static_random_seed)
neg_pos_ratio = params.neg_pos_ratio
cv_fold_num = params.cv_fold_num

class DataLoader():
    def __init__(self):
        # load data
        with open('../data/posNegSamples1.pickle', 'rb') as file:
            pos_set, neg_set = pickle.load(file)
        with open('../data/interMatrix1.npy', 'rb') as file:
            self.matrix = np.load(file)

        random.shuffle(pos_set)
        self.pos_set = pos_set
        random.shuffle(neg_set)
        self.neg_set = neg_set
        self.train_num_ratio = float(cv_fold_num - 1) / cv_fold_num
        self.train_set = []
        self.train_size = 0
        self.test_set = []
        self.test_size = 0

    # load pre-training data
    def getPretrained(self):
        # circ fasta dict
        with open('../data/circFasta1.pkl', 'rb') as f1:
            circFaDict = pickle.load(f1)

        # circ k-mer glove embedding
        with open('data/gloveSeq_vector.pkl', 'rb') as f2:
            kMerWord2vec = np.array(pickle.load(f2))

        # get ontology path for diseases
        with open('data/dis_path1.pkl','rb') as f3:
            dis_path = pickle.load(f3)

        with open('data/gloveDis_vector.pkl', 'rb') as f4:
            disVec = pickle.load(f4)

        return circFaDict, kMerWord2vec, dis_path, np.array(disVec)

    def get_embedDoid_dim(self, embed_file):
        embedded_dim = embed_file.shape
        n_aa_symbols, embedded_dim = embedded_dim
        num_vocab = n_aa_symbols
        embedding_weights = np.zeros((num_vocab, embedded_dim))
        embedding_weights[1:, :] = embed_file[0:-1,:]
        embedding_weights[0] = embed_file[n_aa_symbols-1]
        return num_vocab, embedding_weights

    # read k-mer dict
    def read_dict(self, file='string'):
        odr_dict = {}
        with open(file, 'r') as fp:
            ind = 0
            for line in fp:
                value = line.strip()
                odr_dict[value] = ind
                ind = ind + 1
        return odr_dict

    def get_embedSeq_dim(self, embed_file):
        embedded_dim = embed_file.shape    
        n_aa_symbols, embedded_dim = embedded_dim # glove embedding
        num_vocab = n_aa_symbols + 1 
        embedding_weights = np.zeros((num_vocab, embedded_dim))
        embedding_weights[1:, :] = embed_file[0]
        return num_vocab, embedding_weights

    def interaction_fea(self, trainSet):
        XL_batch = []
        XR_batch = []
        for i, j, l in trainSet:
            temp = self.matrix[i][j]
            self.matrix[i][j] = 0
            XL_batch.append(self.matrix[i])
            XR_batch.append(self.matrix[:, j])
            self.matrix[i][j] = temp
        XL_batch = np.array(XL_batch)
        XR_batch = np.array(XR_batch)
        return XL_batch, XR_batch

    def shuffleMakTraining(self, cv_fold_id):
        assert 0 <= cv_fold_id < cv_fold_num
        test_num_ratio = 1 - self.train_num_ratio
        pos_lens = len(self.pos_set)
        test_set = self.pos_set[
                  cv_fold_id * int(pos_lens * test_num_ratio + 1e-9):(cv_fold_id + 1) * int(
                      pos_lens * test_num_ratio + 1e-9)]
        self.test_set = test_set + self.neg_set[:neg_pos_ratio * len(test_set)]
        self.test_label = len(test_set) * [1] + len(test_set) * [0]
        self.test_size = len(self.test_set)
       
        trainPos_set = self.pos_set[:cv_fold_id * int(pos_lens * test_num_ratio + 1e-9)] \
                       + self.pos_set[(cv_fold_id + 1) * int(pos_lens * test_num_ratio + 1e-9):]

        self.train_set = trainPos_set + self.neg_set[:neg_pos_ratio * len(trainPos_set)]
        self.train_label = len(trainPos_set)*[1] + len(trainPos_set)* neg_pos_ratio *[0]
        self.train_size = len(self.train_set)