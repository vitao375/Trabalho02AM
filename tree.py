# TRABALHO DE ÁRVORES DE DESCISÃO  - APRENDIZADO DE MÁQUINAS
# PROF. MARCELO THIELO
# UNIPAMPA 2019/01
#
#
# AUTORES:  GUILHERME SOUZA SANTOS          152020028
#		    VITOR HUGO MARCIEL DOS SANTOS   161150706
#
#
#
#




#              ALGORITOMO
# A PARTE DO TRATAMENTO DE DADOS FOI ADPTADA DO ALGORITOMO DISPONIVÉL NO LINK
# https://www.kaggle.com/dmilla/introduction-to-decision-trees-titanic-dataset/comments
#  



import numpy as np
import pandas as pd
import re
from numpy import log2 as log
import pprint





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
	Survived = dataset.keys()[0]   
	entropia = 0
	values = dataset[Survived].unique()
	for value in values:
		fraction = dataset[Survived].value_counts()[value]/len(dataset[Survived])
		entropia+= -fraction*log(fraction)
	return entropia


#cALCULA ENTROPIA DO ATRIBUTO X DO COJUNTO DE DADOS
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


#GANHO DE INFORMAÇÃO, USANDO ENTROPIA CONFORME DISCUTIDO EM AULA
def calc_ganho_info(dataset):
	list_ganho_info = []

	for i in dataset.keys()[:-1]:
		list_ganho_info.append(calc_entropia_survived(dataset)-calc_entropy_atributo(dataset, i))
	
	return  dataset.keys()[:-1][np.argmax(list_ganho_info)]



def get_subtable(df, node, value):
	df2 = df[df[node] == value].reset_index(drop = True)
	return df2
	
	

#CONSTRUINDO A ARVORE POR MEIO DE RECURÇÃO USANDO UM DICIONARIO
def buildTree(df,tree=None): 
	
	node = calc_ganho_info(df)
	#print(node)

	attValue = np.unique(df[node])
	
		
	if tree is None:					
		tree={}
		tree[node] = {}
	
   
	for value in attValue:

		#print(value)
		
		subtable =get_subtable(df, node, value)
		
		clValue,counts = np.unique(subtable[target],return_counts=True)	
		#print(np.unique(subtable[target],return_counts=True))
		#print(subtable[target])
		#print(counts)
		#print(clValue)

		#TESTA SE O NO É PURO, E VERIFICA SE ESTA VIVO OU MORTO				
		if(len(counts)== 1):
			if(clValue[0] == 0):
				tree[node][value] = 'morto'
				#pprint.pprint(tree)
				#print('de cima morto')
			else:
				#pprint.pprint(tree)
				tree[node][value] = 'vivo'

		elif((counts[0]/(counts[0]+  counts[1]))>0.80):
			tree[node][value] = 'morto'
		elif((counts[1]/(counts[0]+ counts[1]))>0.80):
			tree[node][value] = 'vivo'	
													
						
		else:	
			subtable = df.drop(columns=[node])
			tree[node][value] = buildTree(subtable) 


	return tree





#TENTA PREDIZER SE O INDIVIDUO SOBREVEVEL OU NÃO AO TITANIC
def predict(inst,tree):

	for nodes in tree.keys():		
		
		value = inst[nodes]
		tree = tree[nodes][value]
		prediction = 0
			
		if type(tree) is dict:
			prediction = predict(inst, tree)
		else:
			prediction = tree
			break;							
		
	return prediction



# TESTES DA TAXA DE ACERTO SE ESTÁ BATENDO COM AS TAXA NA HORA DE CRIAR A ARVORE.
def result(tree):
	n = 0
	for i in range(100):
		test = dataset.sample(1).iloc[0]
		pre = predict(test, tree)
		if(pre=='morto' and test['Survived']==0):
			n+=1
		elif(pre=='vivo' and test['Survived']==1):
			n+=1
		
	return n/100

# Only to see

#print(len(dataset.columns))
tree = buildTree(dataset)
print('Press ENTER para ver a árvore!!')
input()
pprint.pprint(tree)
#print('<><><><><><><<<<><<<<<<<<<<<<<><><>>>>>>>>>>>>>>>>>')
#print(test['Survived'])
print('Press ENTER para ver a taxaa de acerto!!')
input()
print(result(tree))

#print(pre)