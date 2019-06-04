import numpy as np
import pandas as pd



dataset = pd.read_csv('train.csv')
dataset_ = dataset[['PassengerId', 'Survived', 'Pclass', 'Sex','Age', 'SibSp','Parch','Ticket','Fare', 'Cabin', 'Embarked']]

print(dataset_['Cabin'])