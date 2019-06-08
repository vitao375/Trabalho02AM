import numpy as np
import pandas as pd
import re
from numpy import log2 as log
import pprint


# import csv

# table = []
# with open('train.csv','r') as f:
# 	data = csv.DictReader(f)
# 	for row in data:
# 		table.append(row)

# dataset = pd.DataFrame(table, columns=['Name', 'Age', 'Sex', 'Cabin', 'Fare', 'Embarked','Parch' ,'SibSp','FamilySize','Survived'])
# print(dataset)



dataset = pd.read_csv('train.csv')
dataset = dataset[['PassengerId', 'Name','Pclass', 'Sex','Age', 'SibSp','Parch','Ticket','Fare', 'Cabin', 'Embarked', 'Survived']]


# Feature engineering steps taken from Sina and Anisotropic, with minor changes to avoid warnings
full_data = [dataset]


# Feature that tells whether a passenger had a cabin on the Titanic
dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
#test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
	dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']+ 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
	dataset['IsAlone'] = 0
	dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column
for dataset in full_data:
	dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column
for dataset in full_data:
	dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())

# Remove all NULLS in the Age column
for dataset in full_data:
	age_avg = dataset['Age'].mean()
	age_std = dataset['Age'].std()
	age_null_count = dataset['Age'].isnull().sum()
	age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
	# Next line has been improved to avoid warning
	dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
	dataset['Age'] = dataset['Age'].astype(int)

# Define function to extract titles from passenger names
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

for dataset in full_data:
	dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
	dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

	dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
	# Mapping Sex
	#dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
	
	# Mapping titles
	#title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
	#dataset['Title'] = dataset['Title'].map(title_mapping)
	dataset['Title'] = dataset['Title'].fillna(0)

	# Mapping Embarked
	dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
	
	# Mapping Fare
	dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 								= 0
	dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
	dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
	dataset.loc[ dataset['Fare'] > 31, 'Fare'] 									= 3
	dataset['Fare'] = dataset['Fare'].astype(int)
	
	# Mapping Age
	dataset.loc[ dataset['Age'] <= 16, 'Age'] 						   = 0
	dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
	dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
	dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
	dataset.loc[ dataset['Age'] > 64, 'Age']						   = 4


drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
dataset = dataset.drop(drop_elements, axis = 1)
#test  = test.drop(drop_elements, axis = 1)
#dataset = pd.DataFrame(dataset, columns=['Survived', 'Title', 'Sex', 'Age', 'Fare', 'Embarked', 'Has_Cabin','FamilySize', 'IsAlone'])
#print(dataset)
target = 'Survived'


def calc_entropia_survived(dataset):
	Survived = dataset.keys()[0]   #To make the code generic, changing target variable class name
	entropia = 0
	values = dataset[Survived].unique()
	for value in values:
		fraction = dataset[Survived].value_counts()[value]/len(dataset[Survived])
		entropia+= -fraction*log(fraction)
	return entropia


#probabilidade dele sobreviver baseado em um atributo especifico
def calc_entropy_atributo(dataset, atributo):
	colun_survived = dataset.keys()[0]
	values = dataset[colun_survived].unique()
	variaveis = dataset[atributo].unique()
	entropia_atributos=0

	for variable in variaveis:
		entropia = 0
		for i in values:

			num = len(dataset[atributo][dataset[atributo]==variable][dataset[colun_survived] == i])
			den = len(dataset[atributo][dataset[atributo]==variable])
			fraction = num/(den + 0.000001)
			entropia += -fraction*log(fraction+ 0.000001)
		fraction2 = den/len(dataset)
		entropia_atributos+= -fraction2*entropia
	return abs(entropia_atributos)



#print(calc_entropy_atributo(dataset, 'Fare'))
#print(dataset.keys()[:-1])




def calc_ganho_info(dataset):
	list_ganho_info = []

	for i in dataset.keys()[:-1]:
		list_ganho_info.append(calc_entropia_survived(dataset)-calc_entropy_atributo(dataset, i))
	
	return  dataset.keys()[:-1][np.argmax(list_ganho_info)]


def get_subtable(df, node, value):
	df2 = df[df[node] == value].reset_index(drop = True)
	return df2
	
	

def buildTree(df,tree=None): 
	i =0
	Class = df.keys()[-1] 
	print(Class)
	node = calc_ganho_info(df)
	print(node)

	attValue = np.unique(df[node])
	
	#Create an empty dictionary to create tree	
	if tree is None:					
		tree={}
		tree[node] = {}
	
   
	for value in attValue:

		#print(value)
		
		subtable =get_subtable(df, node, value)
		
		clValue,counts = np.unique(subtable[target],return_counts=True)	
		#print(np.unique(subtable[target],return_counts=True))
		#print(subtable[target])				
		
		if (len(counts)== 1 or (counts[1]/(counts[0]+ counts[1])>0.55)):#Checking purity of subset
			tree[node][value] = counts	
													
		else:	
			subtable = df.drop(columns=[node])
			tree[node][value] = buildTree(subtable) #Calling the function recursively 
			
			print("Entrou no else")
			
	#print(df)

	return tree


#print(len(dataset.columns))
tree = buildTree(dataset)
pprint.pprint(tree)


