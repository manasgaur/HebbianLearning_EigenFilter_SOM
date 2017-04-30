import numpy as np
#import numpy.linalg as la
import random as rand
#import random
import scipy.linalg as lin

from scipy.stats import norm

 # Here we are arbitrarily taking a covariance matrix which has all diagonal elements as 1.0 and the upper 
# triangular matrix is the mirror image of the lower triangular matrix   
covariance=[[1.0,0.9,0.6,0.3,0.4,0.3],
            [0.9,1.0,0.5,0.3,0.2,0.2],
            [0.6,0.5,1.0,0.8,0.4,0.3],
            [0.3,0.3,0.8,1.0,0.6,0.7],
            [0.4,0.4,0.4,0.6,1.0,0.7],
            [0.3,0.3,0.3,0.7,0.7,1.0]]
#print ("The determinant of covariance matrix:\n") 
            #It should be a positive value
#print (la.det(covariance))
            

#We determine the lower triangular matrix of the cholesky matrix. We can also take the upper triangular matrix
#   but then it will make the computation difficult while taking the transpose
Chol=lin.cholesky(covariance, lower=True)
# We calculate the eigen values and the eigen vectors of the six dimensional matrix
evalue,evector=lin.eig(covariance)
print ("The six dimensions eigen vectors matrix is:\n")
print(np.array(evector))
# Now we generate a random six dimensional dataset of 100 datapoints.
data=norm.rvs(size=(6,800))
#Now we do the coordinate transformation by taking the dot product of the dataset and the lower triangular matrix of 
# the cholesky covariance matrix.
data=np.dot(Chol,data).T

C_M=[]	
for i in range(6):
	row=[]
	for j in range(6):
		row.append(0)
	C_M.append(row)
for i in range(6):
	for j in range(6):
		summ=0.0
		for val in range(800):
			summ+=data[val][i]*data[val][j]
		summ=summ/(len(data)-1)
		C_M[i][j]=summ
cm=np.array(C_M)
#print ("The synthesised dataset covaiance matrix is:\n")
#print (np.array(cm))
evaluenew,evectornew=lin.eig(cm)
print ("The six dimensions synthesised dataset eigen values matrix is :\n")
print(evaluenew)
print ("The six dimensions synthesised dataset eigen vector matrix is :\n")
print (np.array(evectornew))

X_n =6
Y_n =4
y=[0 for i in range(Y_n)]
w=[]
for j in range(Y_n):
	wj=[]
	for i in range(X_n):
		wj.append(rand.random()/20)
	w.append(wj)
#print("The weight matrix is:")
#print (w[:])
learning_rate=0.0001;
print("The four most important dimensions eigen vectors are:\n")
for train in range(10000000):
	x=data[rand.randrange(0,len(data))]
	for j in range(Y_n):
		y[j]=0.0
		for i in range(X_n):
			y[j]+= x[i]*w[j][i]
	for j in range(Y_n):
		for i in range(X_n):
			foo=0.0
			for k in range(j+1):
				foo+=w[k][i]*y[k]
			w[j][i]+=learning_rate*y[j]*(x[i]-foo)
for i in range(Y_n):
    for j in range(X_n):
        print(w[i][j]),
    print('\t')




