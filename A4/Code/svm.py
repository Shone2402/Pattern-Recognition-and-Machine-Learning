# -*- coding: utf-8 -*-
"""2)SVM_synthetic.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YhiA0Cog7nIOITT8xe-48neEEZi9QGRH
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import math
from sklearn.decomposition import PCA

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

#extract the files
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
    slope.append(fv[i][0])
    slope.append(fv[i][1])
    slopes.append(slope)
  return slopes



#path of the training directories
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

#path of the training directories
devdir=['a','a','a','a','a']
devdir[0] = 'Isolated_Digits/1/dev'
devdir[1] = 'Isolated_Digits/2/dev'
devdir[2] = 'Isolated_Digits/3/dev'
devdir[3] = 'Isolated_Digits/5/dev'
devdir[4] = 'Isolated_Digits/o/dev'
ext = ('.mfcc')

testfv=[]
truth=[]

for i in range(5):
  for files in os.listdir(devdir[i]):
      if files.endswith(ext):
        data=extract(devdir[i]+'/'+files)
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


# split X and y into training and testing sets

#Run SVM with polynomial kernel

# instantiate classifier with polynomial kernel and C=1.0
poly_svc=SVC(kernel='rbf', C=1.0 ,probability=True) 


# fit classifier to training set
poly_svc.fit(trainfv,true)


# make predictions on test set
y_pred=poly_svc.predict(testfv)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(truth, y_pred)))

# Print the Confusion Matrix and slice it into four pieces



print(y_pred-truth)
cm = confusion_matrix(truth, y_pred)
poly_svc.fit(trainfv,true)
S=poly_svc.predict_proba(testfv)
plot_ROC(S,truth,5)

# visualize confusion matrix with seaborn heatmap
import seaborn as sns
cm_matrix = pd.DataFrame(data=cm, columns=['True Class:1', 'True Class:2','True Class:3','True Class:5','True Class:o'], 
                                 index=['Predict Class:1', 'Predict Class:2','Predict Class:3','Predict Class:5','Predict Class:o'])
ax = plt.axes()
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu',ax=ax)
ax.set_title('Confusion Matrix-SVC-MFCC Dataset')
plt.show()

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
        temp=[]
        for k in range(len(data)):
          for j in range(len(data[0])):
            temp.append(data[k][j])
        
        trainfv.append(temp)
        true.append(i)

print(len(trainfv))

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

   
testfv=[]
truth=[]

for i in range(5):
  for files in os.listdir(dirname[i]):
      if files.endswith(ext):
        data=extract2(dirname[i]+'/'+files)
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

#Run SVM with polynomial kernel

# instantiate classifier with polynomial kernel and C=1.0
poly_svc=SVC(kernel='rbf', C=2.0 ,probability=True) 


# fit classifier to training set
poly_svc.fit(trainfv,true)


# make predictions on test set
y_pred=poly_svc.predict(testfv)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(truth, y_pred)))

print(y_pred-truth)
cm = confusion_matrix(truth, y_pred)
poly_svc.fit(trainfv,true)
S=poly_svc.predict_proba(testfv)
plot_ROC(S,truth,5)

# visualize confusion matrix with seaborn heatmap
import seaborn as sns
cm_matrix = pd.DataFrame(data=cm, columns=['True Class:a', 'True Class:ai','True Class:chA','True Class:dA','True Class:lA'], 
                                 index=['Predict Class:a', 'Predict Class:ai','Predict Class:chA','Predict Class:dA','Predict Class:lA'])
ax = plt.axes()
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu',ax=ax)
ax.set_title('Confusion Matrix-SVC-Handwritten Dataset')
plt.show()

def extract3(filename):
  with open(filename) as f:                                                 #change name to trian/trian1/train2
      array = [[(float(x.strip())) for x in line.strip().split(" ")] for line in f]

  
  feat_vecs=[]
  for i in range(0,36):
    feat_vecs.append(array[i])
  fv=np.array(feat_vecs)
  return fv 

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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

#Run SVM with polynomial kernel

# instantiate classifier with polynomial kernel and C=1.0
poly_svc=SVC(kernel='poly' ,probability=True) 


# fit classifier to training set
poly_svc.fit(trainfv,true)


# make predictions on test set
y_pred=poly_svc.predict(testfv)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(truth, y_pred)))

print(y_pred-truth)
cm = confusion_matrix(truth, y_pred)
poly_svc.fit(trainfv,true)
S=poly_svc.predict_proba(testfv)
plot_ROC(S,truth,5)

# visualize confusion matrix with seaborn heatmap
import seaborn as sns
cm_matrix = pd.DataFrame(data=cm, columns=['True Class:coast', 'True Class:forest','True Class:mount','True Class:highw','True Class:openc'], 
                                 index=['Predict Class:coast', 'Predict Class:forest','Predict Class:mount','Predict Class:highw','Predict Class:openc'])
ax = plt.axes()
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu',ax=ax)
ax.set_title('Confusion Matrix-SVC-Image Dataset')
plt.show()

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

#Run SVM with polynomial kernel

# instantiate classifier with polynomial kernel and C=1.0
poly_svc=SVC(kernel='poly', C=1 ,probability=True) 


# fit classifier to training set
poly_svc.fit(train,Z)


# make predictions on test set
y_pred=poly_svc.predict(test)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(Z_tst, y_pred)))

cm = confusion_matrix(Z_tst, y_pred)
poly_svc.fit(train,Z)
S=poly_svc.predict_proba(test)
for i in range(len(Z_tst)):
  Z_tst[i]=Z_tst[i]-1
plot_ROC(S,Z_tst,2)

import seaborn as sns
cm_matrix = pd.DataFrame(data=cm, columns=['True Class:1', 'True Class:2'], 
                                 index=['Predict Class:1', 'Predict Class:2'])
ax = plt.axes()
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu',ax=ax)
ax.set_title('Confusion Matrix-SVC-Synthetic Dataset')
plt.show()