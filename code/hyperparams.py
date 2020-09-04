#coding=utf-8

class Hyperparams:

    # word2vec seq
    k_mer = 5 # k_mer
    dim = 128   # output dim

    # cross validation num
    cv_fold_num = 5

    # generating negative samples
    neg_pos_ratio = 1 # balance data

    # training ratio
    train_val_ratio = 0.9 # each turn train val ratio

    # random seed
    static_random_seed = 1

    # sequence limit
    seqLength = 500
    path_Length = 5
    specify_threshold = 0.5
    # model
    learning_rate = 0.06 # learning rate
    batch_size = 64
    epoch_num = 100
