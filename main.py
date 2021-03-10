#coding=utf-8
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import json

import keras
from keras.layers import *
from keras.models import Model



def read_stopwords(path):
    return pd.read_csv(path, index_col=False, quoting=3, sep="\t", names=["stopwords"], encoding="utf-8")["stopwords"].values

def preprocess_data(infile, outfile, stopwordsfile):
    X = []
    Y = []
    if not os.path.exists(outfile):
        stopwords = read_stopwords(stopwordsfile)
        fr = open(infile, "r", encoding="utf-8")
        fw = open(outfile, "w", encoding="utf-8")
        for eachline in fr.readlines():
            eachline.rstrip("\n")
            linelist = eachline.split("_!_")
            label = int(linelist[1])-100
            if label==15:
                label=5
            if label==16:
                label=11
            Y.append(label)
            segs = jieba.lcut(linelist[3])
            segs = filter(lambda x:len(x)>1, segs)
            segs = filter(lambda x:x not in stopwords, segs)
            segs = " ".join(segs)
            X.append(segs)
            fw.write("__label__%02d , %s\n" % (label, segs))
        fr.close()
        fw.close()
    else:
        fr = open(outfile, "r", encoding="utf-8")
        for eachline in fr.readlines():
            label_segs = eachline.rstrip("\n")
            label_segs = label_segs.split(" , ")
            if len(label_segs)==2:
                X.append(label_segs[1])
                Y.append(int(label_segs[0][-2:]))
        fr.close()
    return X,Y

def data_split():
    X, Y = preprocess_data()
    do = lambda Y:[int(y)-100 for y in Y]
    Y = do(y)
    train_data, test_data, train_target, test_target = train_test_split(X, Y, random_state=1234)
    return train_data, test_data, train_target, test_target

if __name__ == "__main__":
    base_path = "../data/"
    infile = base_path + "toutiao_cat_data.txt"
    outfile = base_path + "toutiao_split_data.txt"
    stopwordsfile = base_path + "stopwords.txt"
    wordindexfile = "word_index.json"
    X, Y = preprocess_data(infile, outfile, stopwordsfile)
    
    num_words = 20000
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X) # 输出的是文本对应的单词索引串
    word_index = tokenizer.word_index
    if not os.path.exists(wordindexfile):
        with open(wordindexfile, "w", encoding="utf-8") as f:
            f.write(json.dumps(word_index))
    #with open(wordindexfile, "r", encoding="utf-8") as f:
    #    word_index = json.load(f) 
    MAX_LENGTH = 200
    data = pad_sequences(sequences, maxlen=MAX_LENGTH)
    labels = to_categorical(np.asarray(Y), num_classes=15)
    indices = list(np.arange(data.shape[0]))
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    train_num = int(data.shape[0]*0.8)
    x_train = data[:train_num]
    x_val = data[train_num:]
    y_train = labels[:train_num]
    y_val = labels[train_num:]

    EMBEDDING_DIM = 16
    # model
    x_in = Input(shape=(None,))
    # 输入(None, input_length),最大小于len(word_index)+1,最长input_length, 输出(None,input_length,EMBEDDING_DIM)
    # input_length->input_length,Embedding_dim
    x = Embedding(num_words,
              EMBEDDING_DIM,
              input_length=MAX_LENGTH)(x_in)
    # 输入(batch_size, steps, input_dim)
    # 输出(batch_size, new_steps, filters)
    x1 = Conv1D(128, 3, padding="valid", activation="relu")(x)
    x2 = Conv1D(128, 4, padding="valid", activation="relu")(x)
    x3 = Conv1D(128, 5, padding="valid", activation="relu")(x)
    x = Concatenate(axis=1)([x1, x2, x3])

    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=0.5)(x)
    p = Dense(15, activation='softmax')(x)
    model = Model(x_in, p)

    learning_rate = 5e-5
    model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.adam(lr=learning_rate), metrics=["acc"])
    # 
    model.load_weights("./cnn_model.weights")
    model.save_weights("./tc_cnn_weights.h5")
    model.save("./tc_cnn_model.h5")
    with open("tc_cnn_model.json", "w", encoding="utf-8") as f:
        f.write(model.to_json())
    
    ##train
    #history = model.fit(data, labels, validation_split=0.2,  epochs=15,  batch_size=128)
    #model.save_weights("./cnn_model.weights")
