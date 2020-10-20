# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:44:17 2020

@author: Pooja Patidar
"""


#******************************question6***************************

import pandas as pd 
import matplotlib.pyplot as plt
#import numpy as np

#read data
data=pd.read_csv("landslide_data3.csv").fillna(value=0)

a=data.boxplot(column='rain',grid=False)

plt.ylabel('rain(in mm)')

plt.title('Boxplot for rain')
plt.show()