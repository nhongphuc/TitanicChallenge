#!/usr/bin/python

#Use the sklearn random forest method to predict Titanic survival.

#Import useful packages
import numpy as np
import pandas
from matplotlib import pyplot
import csv as csv
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from copy import deepcopy

#Read in the test and train files

train = pandas.read_csv("train.csv")
test = pandas.read_csv("test.csv")

#Message data

#Create columns for each point of departure
train['EmbarkedS'] = (train['Embarked']=='S').astype(int)
train['EmbarkedQ'] = (train['Embarked']=='Q').astype(int)
train['EmbarkedC'] = (train['Embarked']=='C').astype(int)

test['EmbarkedS'] = (test['Embarked']=='S').astype(int)
test['EmbarkedQ'] = (test['Embarked']=='Q').astype(int)
test['EmbarkedC'] = (test['Embarked']=='C').astype(int)

#Split Name

train['Surname'] = train.apply(lambda x: x['Name'].split(',')[0], axis=1)
test['Surname'] = test.apply(lambda x: x['Name'].split(",")[0], axis=1)

train['Title'] = train.apply(lambda x: x['Name'].split(',')[1].split(".")[0], axis=1)
test['Title'] = test.apply(lambda x: x['Name'].split(',')[1].split(".")[0], axis=1)

train['First_Name'] = train.apply(lambda x: x['Name'].split(',')[1].split(".")[1], axis=1)
test['First_Name'] = test.apply(lambda x: x['Name'].split(',')[1].split(".")[1], axis=1)

#Create title columns
train['Miss'] = (train.apply(lambda x: 'Miss.' in x['Name'] or 'Ms' in x['Name'] or 'Mlle' in x['Name'], axis=1)).astype(int)
train['Mrs'] = (train.apply(lambda x: 'Mrs.' in x['Name'] or 'Mme' in x['Name'], axis=1)).astype(int)
train['Mr'] = (train.apply(lambda x: 'Mr.' in x['Name'], axis=1)).astype(int)
train['Rev'] = (train.apply(lambda x: 'Rev.' in x['Name'], axis=1)).astype(int)
train['Dr'] = (train.apply(lambda x: 'Dr.' in x['Name'], axis=1)).astype(int)
train['Master'] = (train.apply(lambda x: 'Master' in x['Name'], axis=1)).astype(int)
train['Military'] = (train.apply(lambda x: 'Capt' in x['Name'] or 'Col' in x['Name'] or 'Major' in x['Name'], axis=1)).astype(int)
train['Titled'] = (train.apply(lambda x: 'Don' in x['Name'] or 'Donna' in x['Name']  or 'Sir' in x['Name'] or 'Countess' in x['Name'] or 'Lady' in x['Name'] or ' Jonkheer' in x['Name'], axis=1)).astype(int)


test['Miss'] = (test.apply(lambda x: 'Miss.' in x['Name'] or 'Ms' in x['Name'] or 'Mlle' in x['Name'], axis=1)).astype(int)
test['Mrs'] = (test.apply(lambda x: 'Mrs.' in x['Name'] or 'Mme' in x['Name'], axis=1)).astype(int)
test['Mr'] = (test.apply(lambda x: 'Mr.' in x['Name'], axis=1)).astype(int)
test['Rev'] = (test.apply(lambda x: 'Rev.' in x['Name'], axis=1)).astype(int)
test['Dr'] = (test.apply(lambda x: 'Dr.' in x['Name'], axis=1)).astype(int)
test['Master'] = (test.apply(lambda x: 'Master' in x['Name'], axis=1)).astype(int)
test['Military'] = (test.apply(lambda x: 'Capt' in x['Name'] or 'Col' in x['Name'] or 'Major' in x['Name'], axis=1)).astype(int)
test['Titled'] = (test.apply(lambda x: 'Don' in x['Name'] or 'Donna' in x['Name']  or 'Sir' in x['Name'] or 'Countess' in x['Name'] or 'Lady' in x['Name'] or ' Jonkheer' in x['Name'], axis=1)).astype(int)

#Get surname counts
train['SurnameCount'] = train.groupby('Surname')['Surname'].transform('count')
test['SurnameCount'] = test.groupby('Surname')['Surname'].transform('count')

#Create age categories
train['Child'] = train.apply(lambda x: int(x['Age']<16), axis=1)
train['YoungAdult'] = train.apply(lambda x: int(x['Age']>=16 and x['Age']<31), axis=1)
train['Elderly'] = train.apply(lambda x: int(x['Age']>=60), axis=1)

test['Child'] = test.apply(lambda x: int(x['Age']<16), axis=1)
test['YoungAdult'] = test.apply(lambda x: int(x['Age']>=16 and x['Age']<31), axis=1)
test['Elderly'] = test.apply(lambda x: int(x['Age']>=60), axis=1)

#Motherhood
train['Mother']= train.apply(lambda x: int(x['Age']>16 and x['Parch']>0 and not x['Sex']=='male'), axis=1)
test['Mother']= test.apply(lambda x: int(x['Age']>16 and x['Parch']>0 and not x['Sex']=='male'), axis=1)

#Replace sex by an int, male=1, female=-1
train['Gender'] = train.apply(lambda x: 2*int(x['Sex']== 'male')-1, axis=1)
test['Gender'] = test.apply(lambda x: 2*int(x['Sex']== 'male')-1, axis=1)

#Fill missing age data
MedianMaleAge = test[test['Sex']=='male'].Age.median()
MedianFemaleAge = test[test['Sex']=='female'].Age.median()
test['DefaultAge'] = test.apply(lambda x: MedianMaleAge if x['Sex']=='male' else MedianFemaleAge, axis=1)
test.Age.fillna(test['DefaultAge'], inplace=True)

MedianMaleAge = train[train['Sex']=='male'].Age.median()
MedianFemaleAge = train[train['Sex']=='female'].Age.median()
train['DefaultAge'] = train.apply(lambda x: MedianMaleAge if x['Sex']=='male' else MedianFemaleAge, axis=1)
train.Age.fillna(train['DefaultAge'], inplace=True)

#Fill missing fare data
MediumFare = train.Fare.median()
train.Fare.fillna(MediumFare, inplace=True)

MediumFare = test.Fare.median()
test.Fare.fillna(MediumFare, inplace=True)

#And other data just in case
train.Pclass.fillna(3, inplace=True)
train.SibSp.fillna(0, inplace=True)
train.Parch.fillna(0, inplace=True)

test.Pclass.fillna(3, inplace=True)
test.SibSp.fillna(0, inplace=True)
test.Parch.fillna(0, inplace=True)

#Save modified csv files
train.to_csv("trainDecorated.csv")
test.to_csv("testDecorated.csv")

#Save the passenter IDs to a later submission dataframe
submission_df = pandas.DataFrame()
submission_df['PassengerId'] = deepcopy(test['PassengerId'])

#Drop columns not to be used by the model
train = train.drop(['Name','Sex','PassengerId','Ticket','Cabin','Embarked','Surname','Title','First_Name','DefaultAge'],axis=1)
test = test.drop(['Name','Sex','PassengerId','Ticket','Cabin','Embarked','Surname','Title','First_Name','DefaultAge'],axis=1)
#Seperate into features and target values

train_features = deepcopy(train).drop('Survived', axis=1)
train_targets = train['Survived']

#Make a random forest and a perceptron
my_forest = RandomForestClassifier(n_estimators=25)
perceptron_bag = BaggingClassifier(Perceptron(),n_estimators=100, max_samples=0.5, max_features=1.0)
KN_bag = BaggingClassifier(KNeighborsClassifier(),n_estimators=25, max_samples=0.5, max_features=1.0)
NB_bag = BaggingClassifier(GaussianNB(),n_estimators=100, max_samples=0.5, max_features=1.0)

#Train the classifiers
my_forest.fit(train_features,train_targets)
perceptron_bag.fit(train_features,train_targets)
KN_bag.fit(train_features,train_targets)
NB_bag.fit(train_features,train_targets)

#Predict Probabillities
forest_prob_train = my_forest.predict_proba(train_features)
perceptron_prob_train = perceptron_bag.predict_proba(train_features)
KN_prob_train = KN_bag.predict_proba(train_features)
NB_prob_train = NB_bag.predict_proba(train_features)

forest_prob_test = my_forest.predict_proba(test)
perceptron_prob_test = perceptron_bag.predict_proba(test)
KN_prob_test = KN_bag.predict_proba(test)
NB_prob_test = NB_bag.predict_proba(test)

#Make a prediction
forest_predict = my_forest.predict(test)
forest_score = my_forest.score(train_features,train_targets)

#make some new data frames
train2 = pandas.DataFrame()
test2 = pandas.DataFrame()

#Create additional columns based on classifier predictions
train['RndForest'] = forest_prob_train[:,1]
train['Perceptron'] = perceptron_prob_train[:,1]
train['KNeighbors'] = KN_prob_train[:,1]
train['NaiveBayes'] = NB_prob_train[:,1]

test['RndForest'] = forest_prob_test[:,1]
test['Perceptron'] = perceptron_prob_test[:,1]
test['KNeighbors'] = KN_prob_test[:,1]
test['NaiveBayes'] = NB_prob_test[:,1]

train2['RndForest'] = forest_prob_train[:,1]
train2['Perceptron'] = perceptron_prob_train[:,1]
train2['KNeighbors'] = KN_prob_train[:,1]
train2['NaiveBayes'] = NB_prob_train[:,1]

test2['RndForest'] = forest_prob_test[:,1]
test2['Perceptron'] = perceptron_prob_test[:,1]
test2['KNeighbors'] = KN_prob_test[:,1]
test2['NaiveBayes'] = NB_prob_test[:,1]

#Make and train a perceptron
perceptron = Perceptron()
perceptron.fit(train2,train_targets)
perc_score = perceptron.score(train2,train_targets)
perc_predictions = perceptron.predict(test2)

train_features = deepcopy(train).drop('Survived', axis=1)

#Make and train a new random forest
meta_forest = RandomForestClassifier(n_estimators=25)
meta_forest.fit(train_features,train_targets)
final_score = meta_forest.score(train_features,train_targets)
final_predictions = meta_forest.predict(test)

print
print perc_score
print
print forest_score
print
print final_score.mean()


#Save modified csv files
train.to_csv("trainBare.csv")
test.to_csv("testBare.csv")

#Make predictions with the classifiers
submission_df['Survived'] = final_predictions

#Save modified csv files
submission_df.to_csv("submission.csv")

print '\nDone\n'
