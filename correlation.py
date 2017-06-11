# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

combine = [train_df,test_df]
combine_df = pd.merge(train_df,test_df)

#Data type and distribution for each catagory,
# print(train_df.describe())
# train_df.info()
# print('_'*40)
# test_df.info()

#finding the correlation between each catabories and survival rate
# train_cor = train_df.corr()
# print(train_cor)

# distribution of catagory
#print(train_df.describe(include=['O']))

#Comparing each catatory with survival rate
# print(train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending = False))
# print('_'*40)
# print(train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending = False))
# print('_'*40)
# print(train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending = False))
# print('_'*40)
# print(train_df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending = False))
# print('_'*40)

#plotting age against survival
# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20)


#ploting age and Pclass against survival
# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend();#

#Female vs male survival rate, against embark and Pclass
# grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid.add_legend()



sns.plt.show()
