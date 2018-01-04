#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import os
import re
from tensorflow.python.platform import gfile
import pandas as pd
import pickle
import time
import sys

def create_graph():

    # an automatic download could be put here,
    # but it is likely one have this file somewhere
    
    if not os.path.isfile('classify_image_graph_def.pb'):
        print("")
        print('No classify_image_graph_def.pb in current directory')
        print('Download it and extract from http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz')
        sys.exit(1)
    
    with tf.gfile.FastGFile('classify_image_graph_def.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _=tf.import_graph_def(graph_def, name='')
            graph_def
            return None

    

    
def load_cifar_data(dset,rate=1):

    cifar_dir='cifar-10-batches-py'
    cifar_content=os.listdir(cifar_dir)

    print("cifar_content")
    print(cifar_content)
    fl=cifar_content[dset]

    print("cifar batch is"+fl)
    
    with open(cifar_dir+'/'+fl, 'rb') as f:
        dct = pickle.load(f, encoding='bytes')

    # what I need is an array 10000x10 instead of 10000x1 acting
    # as goal function

    classes=dct[b'labels']

    arr=np.zeros((len(classes),10),dtype='float32')

    for ind,i in enumerate(classes):
        arr[ind,i]=1

    return (dct[b'data'],arr,dct[b'labels'])

if __name__=="__main__":

    no_features=2048

    # hardcoded, 0 batch_3 1, test, on my machine
    idata,itype,ilabel =  load_cifar_data(1)

    create_graph()

    sess = tf.Session()
    #writer = tf.summary.FileWriter("logs/", sess.graph)
    #sys.exit(0) # I only need graph dump

    features=np.zeros((len(ilabel),no_features),dtype='float32')
    #pool_3:0 read from graph dumped previously
    next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
    
    for i,l in enumerate(ilabel):
        if i%50==1:
            print("iteration "+str(i))
#            time.sleep(12) #avoid processor overheat
#            break
        image_data=np.reshape(idata[i,],[1,3,32,32])
        image=np.transpose(image_data,[0,2,3,1])
        predictions = sess.run(next_to_last_tensor,{'ExpandDims:0': image})
        features[i,:] = np.squeeze(predictions)


    # ready for export 
    print(np.shape(features))
    print(np.shape(np.reshape(ilabel,[10000,1])))
    features=np.concatenate([features,np.reshape(ilabel,[10000,1])],1)
    print(np.shape(features))

    names = ['feat'+str(i+1) for i in range(no_features)]+["class"]
    df = pd.DataFrame(features)
    df.to_csv("test_features.csv",header=names)
#    df.to_csv("train_features.csv",header=names)
