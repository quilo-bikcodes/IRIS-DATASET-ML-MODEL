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
from sklearn.preprocessing import StandardScaler

class pltgraph:

    def __init__(self,data):
        self.data = data
        self.params = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

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
        self.showlineplt(self.params[0])
    def get_linepltSWcm(self):
        self.showlineplt(self.params[1])
    def get_linepltPLcm(self):
        self.showlineplt(self.params[2])
    def get_linepltPWcm(self):
        self.showlineplt(self.params[3])


    def get_pairplt(self):
        tmp = self.data.drop('Id', axis=1)
        g = sns.pairplot(tmp, hue='Species', markers='+',kind="reg")
        plt.show()
    def get_pairpltreg(self):
        tmp = self.data.drop('Id', axis=1)
        g = sns.pairplot(tmp, hue='Species', markers='+',kind="reg")
        plt.show()
    
    def get_violinpltSLcm(self):
        self.showViolinPlt(self.params[0])
    def get_violinpltSWcm(self):
        self.showViolinPlt(self.params[1])
    def get_violinpltPLcm(self):
        self.showViolinPlt(self.params[2])
    def get_violinpltPWcm(self):
        self.showViolinPlt(self.params[3])

class MlPrediction:
    def __init__(self,data):
        self.data = data
        self.params = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
    
    def KNNalgo(self, K):
        Scaler = StandardScaler()
        
        
        X = self.data.drop(['Id', 'Species'], axis=1) # Indeependent variable
        Y = self.data['Species'] # Dependent variable
        Scaled_X = Scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        classifier = KNeighborsClassifier(n_neighbors=K)
        classifier.fit(X_train.values,y_train)
        y_pred = classifier.predict([[10,50,355,1]])
        return y_pred
        # return metrics.accuracy_score(y_test, y_pred)
    

    def LogRegalgo(self):
        X = self.data.drop(['Id', 'Species'], axis=1) # Independent variable
        Y = self.data['Species'] # Dependent variable
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        classifier = LogisticRegression()
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        print(y_pred)
        return metrics.accuracy_score(y_test, y_pred)

iris_df = pd.read_csv('./data/iris.csv')
obj = MlPrediction(iris_df)
a = obj.KNNalgo(35)
print(a)
# b = obj.LogRegalgo()
# print(str(a*100)+"%")
# print(str(b*100)+"%")

    

    

            
            

