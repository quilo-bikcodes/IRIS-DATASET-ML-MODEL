# Data sorting
import numpy as np
import pandas as pd

# Graph Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Model Making
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class pltgraph:

    def __init__(self,data):
        self.data = data
        self.labels = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

    def get_stats(self):
        return self.data.describe()

     
    def showlineplt(self,i):
        plt.figure(figsize=(15,5))
        plt.plot(self.data[i])
        plt.title('Iris Dataset')
        plt.xlabel('Flower No. Index' )
        plt.ylabel(i)
        plt.show()
    
    def showViolinPlt(self,category):
        g = sns.violinplot(y='Species', x=category, data=self.data, inner='quartile')
        plt.show()
   
    def get_linepltSLcm(self):
        self.showlineplt(self.labels[0])
    def get_linepltSWcm(self):
        self.showlineplt(self.labels[1])
    def get_linepltPLcm(self):
        self.showlineplt(self.labels[2])
    def get_linepltPWcm(self):
        self.showlineplt(self.labels[3])


    def get_pairplt(self):
        tmp = self.data.drop('Id', axis=1)
        g = sns.pairplot(tmp, hue='Species', markers='+')
        plt.show()
    def get_pairpltreg(self):
        tmp = self.data.drop('Id', axis=1)
        g = sns.pairplot(tmp, hue='Species', markers='+',kind="reg")
        plt.show()
    
    def get_violinpltSLcm(self):
        self.showViolinPlt(self.labels[0])
    def get_violinpltSWcm(self):
        self.showViolinPlt(self.labels[1])
    def get_violinpltPLcm(self):
        self.showViolinPlt(self.labels[2])
    def get_violinpltPWcm(self):
        self.showViolinPlt(self.labels[3])

    

    

            
            

