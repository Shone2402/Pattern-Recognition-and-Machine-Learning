
#Assignment 1-Prml
#Shone Pansambal
#CS19B042

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh,norm,inv
from scipy.linalg import svd
from numpy import array
from matplotlib import image
from PIL import Image

#reading the image into a matrix A

A=image.imread('47.jpg')
A=np.array(A,dtype=np.float64)

#eigen value decomposition

e1,ev1=np.linalg.eig(A)

temp=[0]*len(e1)      #temporary dictionary to store eigen value and vector pair to be sorted
temp1=[0]*len(e1)     #to store sorted eigen values
temp2=[0]*len(e1)     #to store sorted eigen vectors

for i in range(len(e1)):
  temp[i]=(e1[i],ev1[:,i])
temp.sort()                         
temp.reverse()

for k in range(len(e1)):
  temp1[k]=(temp[k][0])
  temp2[k]=(temp[k][1].tolist())

P=array(temp2)        #matrix used to store eigen vectors
P=P.T
Sig=np.diag(array(temp1))      #sigma matrix with eigen values
Pi=inv(P)             #inverse matrix of P

x1=[]
y1=[]
for k in range(1,256):
 e1 = np.matmul(P[:,:k],np.matmul(Sig[:k,:k],Pi[:k,:]))   #the reconstructed image with top k eigen values
 e1=np.array(e1,dtype=np.float64)
 Frobenius_norm1 =np.linalg.norm(A-e1)   #calculating frobenius norm of the error image
 y1.append(Frobenius_norm1)
 x1.append(k)

plt.plot(x1, y1)                      #ploting the norm against k
plt.xlabel('k')
plt.ylabel('Frobenius norm w.r.t orignal image')
plt.title('Plot for evd')
plt.show()

ev,V=eigh(A.T@A)       #calculating eigen vectors of A'*A to generate eigen values
ev=ev.tolist()         
ev.reverse()            #sorting eigen values in reverse order
ev=array(ev)      
rev_ev = [(V[:,len(ev)-1-i]) for i in range(len(ev))]   #eigen vectors in V are reversed
temp=[0]*len(ev)                                  
for i in range(len(ev)):
  ev[i]=ev[i]**0.5                                      #square root of eigen values
  temp[i]=rev_ev[i].tolist()
VT=array(temp)                                          #VT is required matrix

U2 = np.zeros(len(A)*len(A)).reshape(len(A),len(A))    #matrix to store U
for i in range(len(ev)):
 h1 = np.array([VT[i]]).T
 U2[i] = np.true_divide(A@h1,ev[i]).T[0]               #Ui=1/diag(i)*A*vi
U=U2.T                                                 #U is generated

ev=array(ev)
Sig=np.diag(ev)

x=[]
y=[]
for k in range(1,255):
 e1 = np.matmul(U[:,:k],np.matmul(Sig[:k,:k],VT[:k,:]))
 Frobenius_norm1 = np.linalg.norm(A-e1)
 y.append(Frobenius_norm1)
 x.append(k)

# plotting the svd plot for error image
plt.plot(x, y)
plt.xlabel('k')
plt.ylabel('Frobenius norm w.r.t orignal image')
plt.title('Plot for svd')
plt.show()

#showing the reconstructed image by evd for k=200
k=255
e1 = np.matmul(P[:,:k],np.matmul(Sig[:k,:k],Pi[:k,:]))
e1=np.array(e1,dtype=np.float64)
plt.imshow(e1,cmap=plt.get_cmap('gray'))
plt.show()

#reconstructed image for k=30 using svd
k=30
e1 = np.matmul(U[:,:k],np.matmul(Sig[:k,:k],VT[:k,:]))
print(e1)
plt.imshow(e1, cmap=plt.get_cmap('gray'))
plt.show()