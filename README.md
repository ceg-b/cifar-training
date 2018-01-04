# cifar-training

How to reproduce described steps:

1. Run cifar_init.py. It downloads and writes CIFAR-10 data

2. shallow*py train NN with fixed convolution filters. The train
   set is batch_3 and test set is test_batch. Can be different
   for different systems, one must check argument in
   load_cifar_data

3. inception_features.py passes the data through inception-v3
   graph. Should be run twice, for train and test set.
   These sets are given as arguments for load_cifar_data
   (look above). Csv files weight almost 400M, thus
   are not a part of the repo.

4. transfer_svm runs svm classifier on above csv files.
   The exact result for batch_3 and test is 1277 misses
   for 10000 test cases. If U do not want to wait ages
   for producing aboce csv, this is the result.
