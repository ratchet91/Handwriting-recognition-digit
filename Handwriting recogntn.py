from numpy import *
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets
from scipy.misc import imread,bytescale,imresize
digits=datasets.load_digits()
x,y=digits.data,digits.target
print x.shape
print x.dtype
print x.max()
print x.min()
clf=svm.SVC(gamma=0.004)
clf.fit(x,y)
img=imread("R.jpg")
img=imresize(img,(8,8))
img=img.astype(x.dtype)
img=bytescale(img,high=16.0,low=0)
print img
x_testdata=[]
for r in img:
    for c in r:
        x_testdata.append(sum(c)/3.0)

data=clf.predict([x_testdata])
print data