# -*- coding: utf-8 -*-
"""KNN2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bf3TlYUGV5g70WrztXvFAlhWXyNPau31
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# %matplotlib inline
import os
from scipy.spatial import distance_matrix
import math
from sklearn.metrics import accuracy_score


plt.rcParams["font.size"] = 18
plt.rcParams["axes.grid"] = True
plt.rcParams["figure.figsize"] = 8,6
plt.rcParams['font.serif'] = "Cambria"
plt.rcParams['font.family'] = "serif"

# %load_ext autoreload
# %autoreload 2

def plot_ROC(S,truth,classes):
  TPR=[]
  FPR=[]

  mx=max(max(x) for x in S)
  mn=min(min(x) for x in S)
  print(mx,mn)

  x=np.linspace(mx,mn,10000)

  for k in range(1,len(x)):
    TP,TN,FP,FN=0,0,0,0

    for i in range(len(testfv)):
        for j in range(classes):
          if(S[i][j]>=x[k]):
            if(truth[i]==(j)):
                TP=TP+1
            else:
                FP=FP+1
          else:
            if(truth[i]==(j)):
                FN=FN+1
            else:
                TN=TN+1
    TPR.append(float(TP/(TP+FN)))
    FPR.append(float(FP/(FP+TN))) 

  plt.grid()
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.plot(FPR,TPR)
  plt.show()

def extract(filename):
  with open(filename) as f:                                                 #change name to trian/trian1/train2
      array = [[float(x.strip()) for x in line.strip().split(" ")] for line in f]
  d=int(array[0][0])
  f=int(array[0][1])
  
  feat_vecs=[]
  for i in range(1,f+1):
    feat_vecs.append(array[i])
  fv=np.array(feat_vecs)
  return fv

traindir=['a','a','a','a','a']
traindir[0] = 'Isolated_Digits/1/train'
traindir[1] = 'Isolated_Digits/2/train'
traindir[2] = 'Isolated_Digits/3/train'
traindir[3] = 'Isolated_Digits/5/train'
traindir[4] = 'Isolated_Digits/o/train'
ext = ('.mfcc')

trainfv=[]
true=[]

for i in range(5):
  for files in os.listdir(traindir[i]):
      if files.endswith(ext):
        data=extract(traindir[i]+'/'+files)
        temp=[]
        for k in range(len(data)):
          for j in range(len(data[0])):
            temp.append(data[k][j])
        
        trainfv.append(temp)
        true.append(i)

avg_len=0
for i in range(len(trainfv)):
  avg_len=avg_len+len(trainfv[i])

avg_len=int(avg_len/len(trainfv))
print(avg_len)

for i in range(len(trainfv)):
  leng=len(trainfv[i])
  if(leng<avg_len):
   for n in range(leng,avg_len):
    trainfv[i].append(float(0))
  else:
   trainfv[i]=trainfv[i][:avg_len]

testdir=['a','a','a','a','a']
testdir[0] = 'Isolated_Digits/1/dev'
testdir[1] = 'Isolated_Digits/2/dev'
testdir[2] = 'Isolated_Digits/3/dev'
testdir[3] = 'Isolated_Digits/5/dev'
testdir[4] = 'Isolated_Digits/o/dev'
ext = ('.mfcc')

testfv=[]
truth=[]

for i in range(5):
  for files in os.listdir(testdir[i]):
      if files.endswith(ext):
        data=extract(testdir[i]+'/'+files)
        temp=[]
        for k in range(len(data)):
          for j in range(len(data[0])):
            temp.append(data[k][j])
    
        testfv.append(temp)
        truth.append(i)


for i in range(len(testfv)):
  leng=len(testfv[i])
  if(leng<avg_len):
   for n in range(leng,avg_len):
    testfv[i].append(float(0))
  else:
   testfv[i]=testfv[i][:avg_len]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
trainfv=sc.fit_transform(trainfv)
testfv=sc.fit_transform(testfv)

Xtrain=np.array(trainfv)


print(len(Xtrain))

Xtest=np.array(testfv)


print(len(Xtest))

classes = np.unique(true)
classes

"""**bold text**# synthetic dataset"""





## Calculates accuracy of the model
def accuracy(y_pred,y_actual):
    true_count=0
    for i in range(len(y_pred)):
        if y_pred[i]==y_actual[i]:
            true_count+=1;
    return(true_count/len(y_pred))

def euclidean(p1,p2):
    ans=np.linalg.norm(np.array(p1)-np.array(p2))
     
    return ans

def euclidean2(p1,p2):
    ans,i,j=0,0,0
    arr=np.zeros(len(p1[0]))
    while( i < len(p1) and j<len(p2)):
     ans+=np.linalg.norm(np.array(p1[i])-np.array(p2[j]))
     i=i+1
     j=j+1
    while(i<len(p1) ):
      ans+=np.linalg.norm(np.array(p1[i])-np.array(arr))
      i=i+1
    while(j<len(p2) ):
      ans+=np.linalg.norm(np.array(p2[j])-np.array(arr))
      j=j+1
    return ans

from google.colab import drive
drive.mount('/content/drive')

def knn(x,y,test,k):
    distances=[]
    for i in range(len(x)):
        d=euclidean(x[i],test)
        l=(d,x[i],y[i])
        distances.append(l)
    distances.sort(key = lambda x:x[0])
    
    count=Counter()
    sum=[0 for i in range(k)]
    for i in distances[:k]:
        count[i[2]]+=1
        sum[i[2]]+=1
    
    for i in range(len(sum)):
      sum[i]=sum[i]/k
    pred=count.most_common(1)[0][0]
    return(sum,pred)
    
def knn2(x,y,test,k):
    distances=[]
    for i in range(len(x)):
        d=euclidean2(x[i],test)
        l=(d,x[i],y[i])
        distances.append(l)
    distances.sort(key = lambda x:x[0])
    count=Counter()
    sum=[0 for i in range(k)]
    for i in distances[:k]:
        count[i[2]]+=1
        sum[i[2]]+=1
    
    for i in range(len(sum)):
      sum[i]=sum[i]/k
    pred=count.most_common(1)[0][0]
    return(sum,pred)

print(len(Xtrain))
count=1
#for k in range(1,20):
# ytestpred_1 = []
 #for i in Xtest:
  #  ytestpred_1.append(knn(Xtrain, true, i, k)[1])
 #print(accuracy_score(ytestpred_1,truth))

ytestpred_1 = []
S=[]
for i in Xtest:
    ytestpred_1.append(knn(Xtrain, true, i, 11)[1])
    S.append(knn(Xtrain, true, i, 11)[0])
print(accuracy_score(ytestpred_1,truth))
print(S)

print(accuracy_score(ytestpred_1,truth))

cm_knn_test = confusion_matrix(truth,ytestpred_1)
plt.figure(figsize=[6,6])
sns.heatmap(cm_knn_test, annot=True, cbar=False)
plt.title("KNN-Confusion Matrix-Mfcc Dataset")
plt.ylabel("Predicted Class")
plt.xlabel("Actual Class")
plt.show()

plot_ROC(S,truth,5)

def extract2(filename):
  array=[]
  with open(filename) as f:                                                 #change name to trian/trian1/train2
        array = [[float(x.strip()) for x in line.strip().split(" ")] for line in f]
  leng=int(array[0][0])
  fv=[]
  for i in range(1,2*leng,2):
    temp=[]
    temp.append(array[0][i])
    temp.append(array[0][i+1])
    fv.append(temp)
  slopes=[]
  for i in range(len(fv)-1):
    slope=[]
    a=fv[i+1][1]-fv[i][1]
    b=fv[i+1][0]-fv[i][0]
    if(b==0):
      if(a>0):
       slope.append(90)
      else:
       slope.append(-90)
    else:
      s=math.degrees(math.atan(a/b))
      slope.append(s)
    
    slopes.append(slope)
  return slopes

dirname=['a','a','a','a','a']
dirname[0] = 'Handwriting_Data/a/dev'
dirname[1] = 'Handwriting_Data/ai/dev'
dirname[2] = 'Handwriting_Data/chA/dev'
dirname[3] = 'Handwriting_Data/dA/dev'
dirname[4] = 'Handwriting_Data/lA/dev'
traindir=['a','a','a','a','a']
traindir[0] = 'Handwriting_Data/a/train'
traindir[1] = 'Handwriting_Data/ai/train'
traindir[2] = 'Handwriting_Data/chA/train'
traindir[3] = 'Handwriting_Data/dA/train'
traindir[4] = 'Handwriting_Data/lA/train'
ext = ('.txt')

trainfv=[]
true=[]

for i in range(5):
  for files in os.listdir(traindir[i]):
      if files.endswith(ext):
        data=extract2(traindir[i]+'/'+files)
        A=data
         
        trainfv.append(data)
        true.append(i)

testfv=[]
truth=[]

for i in range(5):
  for files in os.listdir(dirname[i]):
      if files.endswith(ext):
        data=extract2(dirname[i]+'/'+files)
        
    
        testfv.append(data)
        truth.append(i)





Xtrain=np.array(trainfv)


print(len(Xtrain))

Xtest=np.array(testfv)


print(len(Xtest))

classes = np.unique(true)
classes

ytestpred_1 = []
S=[]

ytestpred_1 = []
for i in Xtest:
    ytestpred_1.append(knn2(Xtrain, true, i, 9)[1])
    S.append(knn2(Xtrain, true, i, 9)[0])
print(accuracy_score(ytestpred_1,truth))
print(S)



cm_knn_test = confusion_matrix(truth,ytestpred_1)
plt.figure(figsize=[6,6])
sns.heatmap(cm_knn_test, annot=True, cbar=False)
plt.title("KNN-Confusion Matrix-Handwritten Dataset")
plt.ylabel("Predicted Class")
plt.xlabel("Actual Class")
plt.show()

S=sc.fit_transform(S)

plot_ROC(S,truth,5)

def extract3(filename):
  with open(filename) as f:                                                 #change name to trian/trian1/train2
      array = [[(float(x.strip())) for x in line.strip().split(" ")] for line in f]

  
  feat_vecs=[]
  for i in range(0,36):
    feat_vecs.append(array[i])
  fv=np.array(feat_vecs)
  return fv 

dirname=['a','a','a','a','a']

dirname[0] = 'Features/coast/dev'
dirname[1] = 'Features/forest/dev'
dirname[2] = 'Features/highway/dev'
dirname[3] = 'Features/mountain/dev'
dirname[4] = 'Features/opencountry/dev'
traindir=['a','a','a','a','a']
traindir[0] = 'Features/coast/train'
               
traindir[1] = 'Features/forest/train'
traindir[2] = 'Features/highway/train'
traindir[3] = 'Features/mountain/train'
traindir[4] = 'Features/opencountry/train'

trainfv=[]
true=[]

for i in range(5):
  for files in os.listdir(traindir[i]):
    
        data=extract3(traindir[i]+'/'+files)
        
        temp=[]
        for k in range(len(data)):
          for j in range(len(data[0])):
            temp.append(data[k][j])
        
        trainfv.append(temp)
        true.append(i)

print(len(trainfv))




testfv=[]
truth=[]

for i in range(5):
  for files in os.listdir(dirname[i]):
        data=extract3(dirname[i]+'/'+files)
        temp=[]
        for k in range(len(data)):
          for j in range(len(data[0])):
            temp.append(data[k][j])
    
        testfv.append(temp)
        truth.append(i)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
trainfv=sc.fit_transform(trainfv)
testfv=sc.fit_transform(testfv)

Xtrain=np.array(trainfv)


print(len(Xtrain))

Xtest=np.array(testfv)


print(len(Xtest))

classes = np.unique(true)
classes

ytestpred_1 = []
S=[]
for i in Xtest:
    ytestpred_1.append(knn(Xtrain, true, i, 10)[1])
    S.append(knn(Xtrain, true, i, 15)[0])
print(accuracy_score(ytestpred_1,truth))

cm_knn_test = confusion_matrix(truth,ytestpred_1)
plt.figure(figsize=[6,6])
sns.heatmap(cm_knn_test, annot=True, cbar=False)
plt.title("KNN-Confusion Matrix-Image Dataset")
plt.ylabel("Predicted Class")
plt.xlabel("Actual Class")
plt.show()

print(S)

plot_ROC(S,truth,5)

Z,array,train=[],[],[]

with open('train.txt') as f:                                                 #change name to trian/trian1/train2
    array = [[float(x) for x in line.split(",")] for line in f]
    
for x in range(len(array)):
  Z.append(int(array[x][2]))
  train.append([array[x][0],array[x][1]])

Z_tst,array,test=[],[],[]

with open('dev.txt') as f:                                                 #change name to trian/trian1/train2
    array = [[float(x) for x in line.split(",")] for line in f]
    
for x in range(len(array)):
  Z_tst.append(int(array[x][2]))
  test.append([array[x][0],array[x][1]])

trainfv=train
true=Z
testfv=test
truth=Z_tst

Xtrain=np.array(trainfv)


print(len(Xtrain))

Xtest=np.array(testfv)


print(len(Xtest))

classes = np.unique(true)
classes

ytestpred_1 = []
S=[]
for i in Xtest:
    ytestpred_1.append(knn(Xtrain, true, i, 15)[1])
    S.append(knn(Xtrain, true, i, 15)[0])
print(accuracy_score(ytestpred_1,truth))

cm_knn_test = confusion_matrix(truth,ytestpred_1)
plt.figure(figsize=[6,6])
sns.heatmap(cm_knn_test, annot=True, cbar=False)
plt.title("KNN-Confusion Matrix-Synthetic Dataset")
plt.ylabel("Predicted Class")
plt.xlabel("Actual Class")
plt.show()

plot_ROC(S,truth,2)