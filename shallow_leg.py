#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from scipy import misc
from functools import reduce
import pickle
import os
import time
## prepare Haar wavelets: helper functions

fc_depth=384
depth=2

def haar1d(l,n):

    a=1.0/np.sqrt(n)
    if l==0:
        return a*np.ones((n,1),dtype='float32')
    else:
        n=n/np.power(2,l+1)
        return np.concatenate([a*np.ones((int(n),1),dtype='float32'),-a*np.ones((int(n),1),dtype='float32')],0)

def haar_filter(ln,lm,n,m):

    h1=haar1d(ln,n)
    h2=np.transpose(haar1d(lm,m))
    h3=np.outer(h1,h2)

    nn=np.shape(h3)[0]
    mm=np.shape(h3)[1]

    H1=np.reshape(h3,[nn,mm,1,1])
    H0=0*H1

    ff1=np.concatenate([H1,H0,H0],2)
    ff2=np.concatenate([H0,H1,H0],2)
    ff3=np.concatenate([H0,H0,H1],2)
    
    return np.concatenate([ff1,ff2,ff3],3)


def legendre_filter(n):

    x=np.linspace(-1,1,num=n)

    y0=np.reshape(1+0*x,(n,1))
    y1=np.reshape(x,(n,1))
    y2=np.reshape(x*x,(n,1))

    m=np.concatenate([y0,y1,y2],1)

    q,r = np.linalg.qr(m)

    base= [np.outer(q[:,p1],np.transpose(q[:,p2])) for p1 in range(3) for p2 in range(3)]

    #print(base)

    leg_f=[]
    for f in base:
        F1=np.reshape(f,[n,n,1,1])
        F0=0*F1
        
        ff1=np.concatenate([F1,F0,F0],2)
        ff2=np.concatenate([F0,F1,F0],2)
        ff3=np.concatenate([F0,F0,F1],2)
    
        ff =np.concatenate([ff1,ff2,ff3],3)

        leg_f.append(ff)

    return leg_f

# It assumes that directory 'cifar-10-batches-py'
# exists in cwd and has a number of examples inside

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

########################################### main stuff


if __name__=="__main__":

    # prepare wavelets:

    m=n=8 # 16 # for given figures
    

    #wavelet=[haar_filter(i,j,m,n) for i in range(2) for j in range(2) ]

    #print(len(wavelet))

    lf=legendre_filter(m)

    #print([np.shape(ll) for ll in lf])
    #sys.exit(0)

    xs = tf.placeholder(tf.float32, [None,32*32*3])
    ys = tf.placeholder(tf.float32, [None,10])

    # this is how the data are given (cifar_init l. 94 is the proof)
    image0=tf.reshape(xs,[-1,3,32,32])  
    image=tf.transpose(image0,[0,2,3,1])

    # just 3 conolutions

    c1=tf.nn.conv2d(image,lf[0],[1,n//2,n//2,1],padding='VALID')  
    c2=tf.nn.conv2d(image,lf[1],[1,n//2,n//2,1],padding='VALID')  
    c3=tf.nn.conv2d(image,lf[2],[1,n//2,n//2,1],padding='VALID')  
    c4=tf.nn.conv2d(image,lf[3],[1,n//2,n//2,1],padding='VALID')
    c5=tf.nn.conv2d(image,lf[4],[1,n//2,n//2,1],padding='VALID')
    c6=tf.nn.conv2d(image,lf[5],[1,n//2,n//2,1],padding='VALID')
    c7=tf.nn.conv2d(image,lf[6],[1,n//2,n//2,1],padding='VALID')
    c8=tf.nn.conv2d(image,lf[7],[1,n//2,n//2,1],padding='VALID')
    c9=tf.nn.conv2d(image,lf[8],[1,n//2,n//2,1],padding='VALID')  

    out_cn=tf.concat([tf.reshape(c1,[tf.reduce_prod(tf.shape(c1)[1:4]),-1]),
                      tf.reshape(c2,[tf.reduce_prod(tf.shape(c2)[1:4]),-1]),
                      tf.reshape(c3,[tf.reduce_prod(tf.shape(c3)[1:4]),-1]),
                      tf.reshape(c4,[tf.reduce_prod(tf.shape(c4)[1:4]),-1]),
                      tf.reshape(c5,[tf.reduce_prod(tf.shape(c5)[1:4]),-1]),
                      tf.reshape(c6,[tf.reduce_prod(tf.shape(c6)[1:4]),-1]),
                      tf.reshape(c7,[tf.reduce_prod(tf.shape(c7)[1:4]),-1]),
                      tf.reshape(c8,[tf.reduce_prod(tf.shape(c8)[1:4]),-1]),
                      tf.reshape(c9,[tf.reduce_prod(tf.shape(c9)[1:4]),-1])],0)

        
    # Fc layer1
    # InvalidArgumentError (see above for traceback): Matrix size-incompatible: In[0]: [64,1256], In[1]: [484,30000]
    # it's difficult to determine exact vector size after convolution,
    # and TF does not allow to dynamically define var domensions.
    # Thus the procedure is : run, read error string, correct to the right dimension
    c_mat=tf.Variable(tf.truncated_normal([fc_depth,1323 ]))
    c_bis=tf.Variable(0*tf.random_uniform([fc_depth, 1]))
    
    out_fc1=tf.sigmoid(tf.matmul(c_mat,out_cn)+c_bis)
 
    #Fc layer2
    if depth==2:
        d_mat=tf.Variable(tf.truncated_normal([fc_depth, fc_depth]))
        d_bis=tf.Variable(0*tf.random_uniform([fc_depth, 1]))

        out_fc2=tf.sigmoid(tf.matmul(d_mat,out_fc1)+d_bis)

    # output

    e_mat=tf.Variable(tf.truncated_normal([10, fc_depth]))
    e_bis=tf.Variable(0*tf.random_uniform([10, 1]))

    if depth==2:
        prediction = tf.sigmoid(tf.matmul(e_mat,out_fc2)+e_bis)
    else:
        prediction = tf.sigmoid(tf.matmul(e_mat,out_fc1)+e_bis)
#    prediction = tf.matmul(e_mat,out_fc2)+e_bis
                      


    
    #    out=sess.run(prediction,feed_dict={xs: np.reshape(np.random.rand(32*32*3*40,1),(40,3072))})

    yss=tf.transpose(ys,(1,0))
    err_f=tf.reduce_mean(tf.reduce_sum((yss-prediction)*(yss-prediction),0))
#    err_f=tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=yss, logits=prediction)))
    
#    step =tf.train.GradientDescentOptimizer(0.1).minimize(err_f)
    step =tf.train.AdamOptimizer(0.001).minimize(err_f)
    match_f=tf.argmax(prediction,0)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # hardcoded, in fact....
    idata,itype,ilabel =  load_cifar_data(0)
    tdata,ttype,tlabel =  load_cifar_data(1)

    pred=sess.run(match_f,feed_dict={xs: idata,ys: itype})
    print(np.shape(itype))
    print(np.shape(pred))
    print(pred)

    saver = tf.train.Saver()

    sfile='./legendre'+str(fc_depth)+' '+str(depth)+'.ckpt'
    if os.path.isfile(sfile+'.index'):
        saver.restore(sess, sfile)
        print("look like the model has saved state")
    else:
        print("Starting with new blind state state")

    # Do some training
    #writer = tf.summary.FileWriter("logs/", sess.graph)
    #sys.exit(0)
    for i in range(10000):
        i0=0
        i1=200
        sess.run(step, feed_dict={xs: idata[i0:i1,], ys: itype[i0:i1,]})
        if i%50==1:
            pred1=sess.run(match_f,feed_dict={xs: idata[i0:i1,],ys: itype[i0:i1,]})
            pred2=sess.run(err_f,feed_dict={xs: idata[i0:i1,],ys: itype[i0:i1,]})
            pred3=sess.run(match_f,feed_dict={xs: tdata[i0:i1,],ys: ttype[i0:i1,]})
            pred4=sess.run(err_f,feed_dict={xs: tdata[i0:i1,],ys: ttype[i0:i1,]})

            print(np.count_nonzero(pred1-ilabel[i0:i1]),pred2,np.count_nonzero(pred3-tlabel[i0:i1]),pred4)
            
            save_path = saver.save(sess, sfile)
            print("Model saved in file: %s" % save_path)
            time.sleep(5)
