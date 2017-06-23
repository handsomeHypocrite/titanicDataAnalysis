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

# print(train_df.head())

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

#dropping irrelevent data
# print("before", train_df.shape, test_df.shape, combine_df.shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df,test_df]
# print("after", train_df.shape, test_df.shape)


#extracting title using regular expression
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# print(pd.crosstab(train_df['Title'], train_df['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
# print(pd.crosstab(train_df['Title'],train_df['Sex']))

#making titles catagory ordinal
title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5 }

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
# print(train_df.head())

#drop name and cabin number
train_df.drop(['Name', 'PassengerId'], axis=1, inplace=True)
test_df.drop(['Name', 'PassengerId'], axis=1, inplace=True)
combine = [train_df,test_df]

#relace Sex catagory with int data
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)
# print(train_df.head())

#fill in empty data with estimations
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

#create empty array
guess_ages = np.zeros((2,3))
# print(guess_ages)

#guess missing values
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()
            # print(dataset.head())
            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

# print(train_df["Age"][train_df["Age"]<15].head())
# print('_'*40)

#Slice age into 5 groups (graph it)
train_df['AgeBand']=pd.cut(train_df['Age'], 5)
# print(train_df[['AgeBand','Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))
# print(train_df['AgeBand'].head())

#make age ordinal
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.drop(['AgeBand'],axis=1,inplace = True)

#combine sibsp and parch
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# print(train_df[['FamilySize','Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False))


#creating a column name IsAlone
for dataset in combine:
    dataset['IsAlone']=0
    dataset.loc[dataset['FamilySize']==1, 'IsAlone']=1
    # dataset[dataset['FamilySize']==1]['IsAlone']=1

# print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=True).mean())


#choosing is alone feature over family size, sibsp and parch
train_df.drop(['SibSp','Parch', 'FamilySize'],axis=1, inplace=True)
test_df.drop(['SibSp','Parch', 'FamilySize'],axis=1, inplace=True)
combine=[train_df, test_df]

for dataset in combine:
    # dataset['Age*Class']=dataset['Age']*dataset['Pclass']
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

#print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))
# print(train_df[['Age*Class','Age','Pclass']].head())


freq_port = train_df.Embarked.dropna().mode()[0]
print(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived',ascending=False))

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



# print(train_df['Embarked'])

#completing fare
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

#creating fare band
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
# print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df.drop(['FareBand'], axis=1, inplace=True)
combine = [train_df, test_df]

# print(train_df.head(10))
# print(test_df.head(10))

#Seperating data into datasets to accomodate sklearn funtion structure
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.copy()
# print(X_train.shape, Y_train.shape, X_test.shape)


# using logistic regression model. Resulting in accuracy score of 80.36
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# print(Y_pred)
print('the accuracy of Logistic Regression model is:', acc_log)
# sns.plt.show()

# prediction using KNN model with respect to 5 nearest neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
# print(Y_pred)
acc_log = round(knn.score(X_train, Y_train) * 100, 2)
print("the accuracy of KNN model is: ",acc_log)







#using KNN model

