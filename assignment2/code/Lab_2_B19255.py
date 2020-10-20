# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:53:01 2020

@author: Pooja Patidar
"""


#Pooja Patidar
#B19255
#8516921968

#import libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

#read data
original=pd.read_csv("pima_indians_diabetes_original.csv")
missing=pd.read_csv("pima_indians_diabetes_miss.csv")

#****************************************Question1*****************************
print("Question 1 : ")
#defining dataframe
df=pd.DataFrame(missing)
y=df.isnull().sum()
p=[]
for i in range(9):
    p.append(y[i])
#creating plot
x=['pregs','plas','pres','skin','test','BMI','pedi','Age','class']
plt.bar(x,p)
plt.xlabel('attribute names')
plt.ylabel('no. of missing values')
plt.title('bar graph of missing value')
plt.show()


#**********************************************Question2***********************
#********************part(a)********************
print("\nQuestion 2 (a) : ")
#calculating tuples having 1/3rd missing attributes
row=[]
for i in range(len(df.index)):
    if (df.iloc[i].isnull().sum())>=3:
        row.append(i)
print("\nTotal deleted rows : ",len(row))
print("\nRow no. of deleted tuples : ",row)
#deleting tuples 
for j in range(len(row)):
    df.drop(row[j],axis=0,inplace=True)


#********************part(b)********************
print("\nQuestion 2 (b) : ")
#calculating tuples having misssing attributes
row=df[df['class'].isna()]
#deleting tuples
df.drop(row.index,inplace=True)
print("\nTotal deleted rows :",len(row.index))
print("\nRow no. of deleted tuples : ",*row.index)


#**********************************************Question3***********************
print("\nQuestion 3 : \n")
#calculating total missing values
a=df.isnull().sum()
b=df.isnull().sum().sum()
print(a)
print("\nTotal no. of missing values : ",b)


#**********************************************Question4***********************
#********************part(i)********************
#defining function
def filling(df):
    #creating table for statistics
    data=pd.concat((df.mean(),df.median(),df.mode().loc[0],df.mode().loc[1],df.std()),axis=1)
    index=[['Mean','Median','Mode1','Mode2','Standard Dev']]
    data=data.T
    data.set_index(index,inplace=True)
    data=data.T
    print('statistics of missing data')
    print(data)
    
    data_org=pd.concat((original.mean(),original.median(),original.mode().loc[0],original.mode().loc[1],original.std()),axis=1,)
    data_org=data_org.T
    data_org.set_index(index,inplace=True)
    data_org=data_org.T
    print('\nstatistics of Original Data:\n')
    print(data_org)
    
#********************part(ii)********************
#define function
def RMSE(df_):
    #calculate rmse
    RMSE=[0]*len(df.columns)
    n=0
    for i in df.columns:
        null_index=df[i][df[i].isna()].index
        if len(null_index)==0:
            RMSE[n]=0
        else:
            for j in null_index:
                RMSE[n]+=(df_[i][j]-original[i][j])**2
                
            RMSE[n]/=len(null_index)
            RMSE[n]**=0.5
        print(i,':',RMSE[n])
        n+=1
    #creat plot
    plt.bar(df.columns,RMSE)
    plt.xlabel('Attributes')
    plt.ylabel('RMSE')
    plt.show()
#a
print('\nQuestion 4 (a)')
#replace missing value with mean of attribute
df_a=df.fillna(df.mean())
#i
print('i)\n')
filling(df_a)
#ii
print('\nii)\n')
RMSE(df_a)

#b
print('\nQuestion 4 (b)))')
#replace missing value using linear interpolation
df_b=df.interpolate()
#i
print('i)\n')
filling(df_b)
#ii
print('\nii)\n')
RMSE(df_b)


#**********************************************Question5*********************** 
#calculating Q1 Q2 Q3
Q1=df_b['Age'].quantile(0.25)
Q2=df_b['Age'].quantile(0.50)
Q3=df_b['Age'].quantile(0.75)
IQR=Q3-Q1
lower_whisker=Q1-1.5*IQR
upper_whisker=Q3+1.5*IQR

#total outliers
outlier=[]
for i in df_b['Age']:
    if i<lower_whisker or i>upper_whisker:
        outlier.append(i)
#create plots
plt.boxplot(df_b['Age'],patch_artist=True)
plt.ylabel('Age')
plt.title('Age')
plt.ylim(60,80)
plt.show()
print("\nQuestion 5\n")
print("i)\n")
print("outliers in Age :",outlier)
print('total outliers :',len(outlier))

#replace outliers with median
for j in df_b['Age']:
    if j<lower_whisker or j>upper_whisker:
        df_b.replace(j,df_b['Age'].median(),inplace=True)

#create plot
plt.boxplot(df_b['Age'],patch_artist=True)
plt.ylabel('Age')
plt.ylim(60,80)
plt.title('Age after replacing')
plt.show()   

#same as above
Q1=df_b['BMI'].quantile(0.25)
Q2=df_b['Age'].quantile(0.25)
Q3=df_b['BMI'].quantile(0.75)
IQR=Q3-Q1
lower_whisker=Q1-1.5*IQR
upper_whisker=Q3+1.5*IQR
outlier_BMI=[]
for i in df_b['BMI']:
    if i<lower_whisker or i>upper_whisker:
        outlier_BMI.append(i)
plt.boxplot(df_b['BMI'],patch_artist=True)
plt.ylabel('BMI')
plt.title('BMI')
plt.show()
print("\n(ii): \n")
print("outliers in BMI :",outlier_BMI)
print('total outliers :',len(outlier_BMI))


for j in df_b['BMI']:
    if j<lower_whisker or j>upper_whisker:
        df_b.replace(j,df_b['BMI'].median(),inplace=True)
        
plt.boxplot(df_b['BMI'],patch_artist=True)
plt.ylabel('BMI')
plt.title('BMI after replacing')
plt.show()
