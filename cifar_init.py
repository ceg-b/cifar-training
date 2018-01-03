#!/usr/bin/env python3

import sys
import os
import subprocess as sbp
import os.path
import getopt
import pickle
import numpy as np
import matplotlib.pyplot as plt

cifar_link='https://www.cs.toronto.edu/~kriz/'
cifar_file='cifar-10-python.tar.gz'
cifar_dir='cifar-10-batches-py'

try:
    opts, args = getopt.getopt(sys.argv[1:], "F")
except:
    sys.exit(2)

force_download=False
for o, a in opts:
     if o == "-F":
        force_download=True
                               

##################################################################
#download and extract
##################################################################

if os.path.isfile(cifar_file) and not force_download:
    print("Looks like cifar data are in cwd")
else:
    p = sbp.Popen(["wget "+cifar_link+cifar_file], shell=True,stdout=sbp.PIPE,stderr=sbp.PIPE)
    (out,err) = p.communicate();
    code=p.returncode

    if code!=0:
        print("something went wrong, cifar data not downloaded")
        sys.exit(1)
    else:
        print("Wget finished with success")

if os.path.isdir(cifar_dir):
    print("Looks like cifar data are extracted in cwd")
else:
    p = sbp.Popen(["tar xzf "+cifar_file], shell=True,stdout=sbp.PIPE,stderr=sbp.PIPE)
    (out,err) = p.communicate();
    code=p.returncode

    if code!=0:
        print("something went wrong with extracting data.")
        sys.exit(1)
    else:
        print("Data extracted")

###############################################################################
#
###############################################################################

cifar_content=os.listdir(cifar_dir)

fl=cifar_content[0]

with open(cifar_dir+'/'+fl, 'rb') as f:
    dct = pickle.load(f, encoding='bytes')

# print(dct.keys())
# print(dct[b'labels'])
print(np.shape(dct[b'data']))

tmp=zip(dct[b'labels'],range(len(dct[b'labels'])))
ordered_labels=sorted(list(tmp), key=lambda tup: tup[0])

# Sorting is not necessary before grouping.
# I need explicitly evaluated list, not iterator object
grouped_labels=map(lambda key:filter(lambda k: k[0]==key,ordered_labels),range(10))
grouped_labels=[list(g) for g in grouped_labels]

# take 10 elements from each group

example_rows   = map(lambda g: np.random.randint(low=len(g),size=10).tolist(),grouped_labels)
matrix_entries = map(lambda tuple_: map(lambda i: (tuple_[1])[i][1],tuple_[0]),zip(example_rows,grouped_labels))
matrix_entries = [me for me in matrix_entries]
#print(list(map(list,matrix_entries)))
f, axarr = plt.subplots(10, 10)

print(matrix_entries)
for indx,row in enumerate(matrix_entries):
#    print(row)
    for indy,column in enumerate(row):
#        print(column)
        img_data=np.reshape(dct[b'data'][column,:],(3,32,32))
        axarr[indx,indy].imshow(img_data.transpose(1,2,0))
        axarr[indx,indy].axis('off')






f.subplots_adjust(hspace=0.1)
plt.axis('off')
plt.show()
