# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:45:01 2020

@author: Pooja Patidar
"""

#Pooja Patidar
#B19255
#8278785526

#import libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

#read data
data=pd.read_csv("landslide_data3.csv").fillna(value=0)

#*****************************************Question1*******************************
print("Quetion1")
a=['temperature','humidity','pressure','rain','lightavgw/o0','lightmax','moisture']

#simply using inbuilt functions of data and printing that values

mean=[]
median=[]
mode=[]
minimum=[]
maximum=[]
st_dev=[]
for i in range(len(a)): 
    print("\n",i+1,a[i])
    mean.append(data[a[i]].mean()) 
    print("mean :",mean[i])                
    median.append(data[a[i]].median()) 
    print("median :",median[i])        
    mode.append(data[a[i]].mode()[0])   
    print("mode :",mode[i])       
    minimum.append(data[a[i]].min())
    print("minimum :",minimum[i])                    
    maximum.append(data[a[i]].max())
    print("maximum :",maximum[i])                    
    st_dev.append(data[a[i]].std())  
    print("standard deviation :",st_dev[i])                
print("\n")


#***********************************************Question2******************************

#creat data
a=['temperature','humidity','pressure','lightavgw/o0','lightmax','moisture']
for i in range(len(a)):
    x=data.rain
    y=data[a[i]]
 
    #creat plot
    plt.scatter(x,y)
    
    #set x axis label
    plt.xlabel('rain')
    
    #set y axis label
    plt.ylabel(str(a[i]))
    
    #set title
    plt.title('rain and '+str(a[i]))
    plt.show()

#same process as above   
a=['humidity','pressure','rain','lightavgw/o0','lightmax','moisture']
for i in range(len(a)):
    x=data.temperature
    y=data[a[i]]
    plt.scatter(x,y)
    plt.xlabel('temperature')
    plt.ylabel(str(a[i]))
    plt.title('temperature and '+str(a[i]))
    plt.show()


#*******************************************Question3*******************************    

print("Question 3 \n")
#creat data
a=['temperature','humidity','pressure','lightavgw/o0','lightmax','moisture']
for i in range(len(a)):
    x=data.rain
    y=data[a[i]]
 
#finding correlation
    corr=np.corrcoef(x,y)
    print("corr coeff of rain & "+str(a[i]),"     :  %.3f"%corr[0,1])
print("\n")

#same process as above
a=['humidity','pressure','rain','lightavgw/o0','lightmax','moisture']
for i in range(len(a)):
    x=data.temperature
    y=data[a[i]]
    corr=np.corrcoef(x,y)
    print("corr coeff of temperature & "+str(a[i]),"       : %.3f"%corr[0,1])


#********************************************Question4******************************

#creat data
x=data["rain"]

#plot
x.hist()
plt.xlabel('rain')
plt.ylabel("frequency")
plt.title("histogram for rain")
plt.show()

#same process as above
y=data["moisture"]
y.hist()
plt.xlabel('moisture')
plt.ylabel("frequency")
plt.title("histogram for moisture")
plt.show()


#************************************************Question5***********************


#creat data and ploting

#using direct function in pandas
data.rain.hist(by=data.stationid)


#***********************************************Question6************************


#creat data
data.boxplot(column='rain',grid=False,whis=[15,85])

#set label for y-axis
plt.ylabel('rain(in mm)')

#set title 
plt.title('Boxplot for rain')
plt.show()

#same process as above
data.boxplot(column='moisture',grid=False)
plt.ylabel('moisture(in %)')
plt.title('Boxplot for moisture')
plt.show()





