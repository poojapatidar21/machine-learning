#Pooja Patidar
#8516921968
#B19255

#import libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

#read data
df=pd.read_csv("landslide_data3.csv")

#***************************Question 1***************
#part(a)
print('Question 1','\n(a)')
for i in df.columns[2:]:
    Q1=df[i].quantile(0.25)
    Q3=df[i].quantile(0.75)
    IQR=Q3-Q1
    lower_whisker=Q1-1.5*IQR
    upper_whisker=Q3+1.5*IQR

#replace outliers with median
    for k in df[i]:
        if k<lower_whisker or k>upper_whisker:
            df.replace(k,df[i].median(),inplace=True)

#creating table for min-max values before normalization
df_a=df.copy()
data=pd.concat((df_a[df_a.columns[2:]].min(),df_a[df_a.columns[2:]].max()),axis=1)
index=[['Minimum','Maximum']]
data=data.T
data.set_index(index,inplace=True)
data=data.T
print('\nMin-Max values before normalization')
print(data)

#normalizing the data
for j in df_a.columns[2:]:
    a=df_a[j].min()
    b=df_a[j].max()
    df_a[j]=(((df_a[j]-a)*6)/(b-a)+3)

#min-max values after normalization
data=pd.concat((df_a[df_a.columns[2:]].min(),df_a[df_a.columns[2:]].max()),axis=1)
index=[['Minimum','Maximum']]
data=data.T
data.set_index(index,inplace=True)
data=data.T
print('\nMin-Max values after normalization')
print(data)

#part(b)
#calulating mean and standard deviation before standardization
print('\n(b)')
df_b=df.copy()
data=pd.concat((df_b[df_b.columns[2:]].mean(),df_b[df_b.columns[2:]].std()),axis=1)
index=[['Mean','Standard Deviation']]
data=data.T
data.set_index(index,inplace=True)
data=data.T
print('\nMean-Std values before standardization')
print(data)

#standardization dat
for j in df_b.columns[2:]:
    a=df_b[j].mean()
    b=df_b[j].std()
    df_b[j]=(df_b[j]-a)/b

#mean and standard devition values after standardization
data=pd.concat((round(df_b[df_b.columns[2:]].mean(),6),df_b[df_b.columns[2:]].std()),axis=1)
index=[['Mean','standard deviation']]
data=data.T
data.set_index(index,inplace=True)
data=data.T
print('\nMean-Std values after standardization')
print(data)


#*********************Question 2 *************
#part(a)
print('\n(a)')
mean=np.array([0,0]) #mean array
cov=np.array([[5,10],[10,13]])  #covariance matrix
D=(np.random.multivariate_normal(mean,cov,1000)).T #multivariant data 
plt.scatter(D[1],D[0],marker='+',color='blue') #plot
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('scatter plot')
plt.show()

#part(b)
print('\n(b)') 
eigval,eigvec=np.linalg.eig(np.cov(D))  # eigenvalues and eigenvectors
print('Eigen values:',*eigval,'\nEigen vectors:',*eigvec)
plt.scatter(D[1],D[0],marker='+',color='blue')  #plot
plt.quiver([0],[0],eigvec.T[0][0],eigvec.T[1][0],angles="xy",color='red',scale=3)
plt.quiver([0],[0],eigvec.T[0][1],eigvec.T[1][1],angles="xy",color='red',scale=6)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Plot of 2D synthetic data and Eigen vectors')
plt.show()

#part (c)
prj=np.dot(D.T,eigvec)  #projection of vector
plt.scatter(D[1],D[0],marker='+',color='blue')  #plot
plt.quiver([0],[0],eigvec.T[0][0],eigvec.T[1][0],angles="xy",color='red',scale=3)
plt.quiver([0],[0],eigvec.T[0][1],eigvec.T[1][1],angles="xy",color='red',scale=6)
plt.scatter(prj[:,1]*eigvec[0][0],prj[:,1]*eigvec[0][1],color='magenta',marker='+')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Projected values on 1st eigen vector')
plt.show()

plt.scatter(D[1],D[0],marker='+',color='blue') #plot
plt.quiver([0],[0],eigvec.T[0][0],eigvec.T[1][0],angles="xy",color='red',scale=3)
plt.quiver([0],[0],eigvec.T[0][1],eigvec.T[1][1],angles="xy",color='red',scale=6)
plt.scatter(prj[:,0]*eigvec[1][0],prj[:,0]*eigvec[1][1],color='magenta',marker='+')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Projected values on 2nd eigen vector')
plt.show()

#part(d)
print('\n(d)')
D1=(np.dot(prj,eigvec.T)).T        #reconstructing data
error=mean_squared_error(D,D1)  #mean square error
print('Mean square error :',round(error,6))


#*************************Question 3*******************
#part(a)
print('\nQuestion 3')
print('\n(a)')
df=df_b.copy()
df.drop(['dates','stationid'],axis=1,inplace=True)
eigval,eigvec=np.linalg.eig(np.cov(df.T))  #eigenvalues and eigenvectors of data
eigval=sorted(eigval,reverse=True)
pca=PCA(n_components=2) #reduce dimensionality
Data=pca.fit_transform(df)  #fit the modal with df
#calculating variance and eigen value corresponding eigen vectors
for i in range(2):
    print('Variance along Eigen Vector',i+1,':',np.var(Data.T[i]),  
          '\nEigen Value corresponding to Eigen Vector',i+1,':',eigval[i],'\n')

plt.scatter(Data.T[0],Data.T[1],color='blue') #plot
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatter plot of reduced dimensional data')
plt.show()

#part(b)
plt.bar(range(1,8),eigval,color='orange')   #plot
plt.xlabel('Index')
plt.ylabel('Eigen Value')
plt.title('Eigen Values in Descending Order')
plt.show()

#part(c)
RMSE=[]
for i in range(1,8):
    pca=PCA(n_components=i) #defining value of l
    Data=pca.fit_transform(df)
    D1=pca.inverse_transform(Data)   #transform data back to its original space
    p=(mean_squared_error(df.values,D1))**.5
    RMSE.append(p)
    
plt.bar(range(1,8),RMSE,color='orange') #plot
plt.ylabel('RMSE')
plt.xlabel('l')
plt.title('Reconstruction Error')
plt.show()

