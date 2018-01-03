#!/usr/bin/env python3

import pandas as pd
import numpy as np
import tensorflow as tf
import sys
from sklearn import svm

no_feats=2048

#train_set=pd.read_csv('train_features.csv', sep=',').values
#test_set=pd.read_csv('test_features.csv', sep=',').values

train_set=pd.read_csv('file_path.csv', sep=',').values
test_set=pd.read_csv('file_path_test.csv', sep=',').values


train_features=train_set[:,1:no_feats]
train_labels=train_set[:,-1]
test_labels=test_set[:,-1]
test_features=test_set[:,1:no_feats]

#print(train_labels)
#sys.exit(0)
#model=svm.SVC(kernel='linear',C=1)
#model.fit(train_features,train_labels)
#model_response=model.predict(test_features)

#print(list(model_response[1:1000]))
#print(list(train_labels[1:1000]))


clf = svm.LinearSVC(C=.1, loss='squared_hinge', penalty='l2',multi_class='ovr',max_iter=2000)
clf.fit(train_features, train_labels)
y_pred = clf.predict(test_features)
print(list(y_pred[1:1000]))
print(np.count_nonzero(y_pred-test_labels))
#print(np.count_nonzero(model_response-test_labels))
