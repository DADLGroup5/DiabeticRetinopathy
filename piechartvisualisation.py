# -*- coding: utf-8 -*-
"""piechartvisualisation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KDBBvrHTzO_PRs5gXV6z3u0zdPCdinla
"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
# %matplotlib inline

from google.colab import files
dff = files.upload()

df = pd.read_csv('piechart.csv')

print(df)

#data
category = df["category"]
num = df["num"]
color = ['red','lightskyblue']
#plot 
result = plt.pie(num,labels=category,autopct='%1.1f%%',colors=color)
plt.show()