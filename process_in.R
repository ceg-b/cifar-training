library("ggplot2")

#it takes a long time
#train=read.csv("file_path.csv")
#test=read.csv("file_path_test.csv")


if (ncol(test)==2050) {
    train_labels=train[,2050]
    test_labels=test[,2050]
    train=train[,2:2049]
    test=test[,2:2049]
}

#tr_svd=svd(train)
#tst_svd=svd(test)


feats=data.frame(tr_svd$u[,1:50],train_labels)
feats_tst=data.frame(tst_svd$u[,1:50],test_labels)

# make small representative tables, just for the plot
Sfeats=feats[sample.int(nrow(feats),1000),]
Sfeats_tst=feats_tst[sample.int(nrow(feats),1000),]
names(Sfeats)=names(feats)
names(Sfeats_tst)=names(feats_tst)

## plot
postscript(file="train_set.eps",onefile=FALSE)
plot(Sfeats$X1,Sfeats$X2,col=Sfeats$train_labels,cex=1,pch=20,xaxt="n",yaxt="n")
dev.off()

postscript(file="test_set.eps",onefile=FALSE)
plot(Sfeats_tst$X1,Sfeats_tst$X2,col=Sfeats_tst$test_labels,cex=1,pch=20,xaxt="n",yaxt="n")
dev.off()


## and write the most important features to file for svm
## svm will be done in Python.

## The question: why this small part is R?
## It's about interactive session I used for
## playing around the data
write.csv(feats,"train_features.csv")
write.csv(feats_tst,"test_features.csv")

