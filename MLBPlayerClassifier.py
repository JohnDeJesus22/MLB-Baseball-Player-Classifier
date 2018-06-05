#ANN Classifier built to assign positions to a baseball player using their stats.

#Part 1: Import Libraries and Data--------------------------------------------
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import graphviz

os.chdir('D:\\')
#Import data
dataset=pd.read_csv('MLB2016PlayerStats.txt')

os.chdir('C:\\Users\\Sabal\\Anaconda3\\Lib\\site-packages\\graphviz')

#Split "Pos Summary" column to obtain Original Position
dataset[['OriginalPosition',
	'2ndPosition',
	'3rdPosition',
	'4thPosition',
	'5thPosition',
	'6thPosition']] = dataset['Pos Summary'].str.split('-',expand=True)

#Drop "Pos Summary" column
dataset=dataset.drop(['Pos Summary'],axis=1)

#Part 2: Prep Data for ANN-----------------------------------------------------
#Check to see if any values are missing
dataset.isnull().any()

#Filling in missing values of columns from 'Fld%' 
#to 'Rdrs/yr' with zeros based on domain knowledge
for column in dataset.columns[14:19]:
    dataset[column].fillna(0,inplace=True)

#Check to make sure all values desired columns are filled
dataset.isnull().any()

#Data exploration for missing values in Field Percentage
field_nulls=dataset[dataset['Fld%'].isnull()]

#Determine positions under Field Percentage null criteria
field_nulls.OriginalPosition.unique()

#Count the number of instances of each position
field_nulls.OriginalPosition.value_counts()

#Split data into independent and dependent variables
X=dataset.iloc[:,5:21].values
y=dataset.OriginalPosition

#Convert positions into number values and create binary columns
#for each position
from sklearn.preprocessing import LabelBinarizer
labelbinarizer_y=LabelBinarizer()
y=labelbinarizer_y.fit_transform(y)

#Spliting the dataset into the training set and the test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Part 3: Building, Training, and Testing ANN-----------------------------------
#Importing keras and classes
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier=Sequential()

#Initial Layer and First Hidden Layer
classifier.add(Dense(input_dim=16, kernel_initializer ='uniform', activation ='relu', units=11))

#Second Hidden Layer
classifier.add(Dense(units=11, kernel_initializer ='uniform', activation ='relu'))

#Output Layer
classifier.add(Dense(units=7, kernel_initializer ='uniform', activation ='softmax'))

#visualizing ann with ann visualizer
from ann_visualizer.visualize import ann_viz

ann_viz(classifier, view=True,filename='MLBClassifier.gv',title='MLB ANN Classifier')

#Compile ANN
classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#Fit Training Set to ANN
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

#Apply Test Set to ANN
y_pred=classifier.predict(X_test)
y_pred=(y_pred>.5)

#Testing a Entry from 2017 Baseball Season (Jose Abreu)
player=[139,138,130,1197,1221,1135,78,8,130,.993,3,3,0,0,9.12,8.73]

#Transforming player entry and running through the ANN
new_entry=classifier.predict(sc_X.transform(np.array([player])))

#Inverse transforming the resulting prediction to string class
new_entry=labelbinarizer_y.inverse_transform(new_entry)
print(new_entry)


#Part 4: K-Fold Cross-Validation-----------------------------------------------
#import libraries
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#Create function from our ANN for wrapper
def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(units=11,kernel_initializer ='uniform' ,activation ='relu',input_dim=16))
    classifier.add(Dense(units=11,kernel_initializer ='uniform' ,activation ='relu'))
    classifier.add(Dense(units=7,kernel_initializer ='uniform' ,activation ='softmax'))
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
    return classifier

#Wrap ANN in KerasClassifier
classifier=KerasClassifier(build_fn=build_classifier, batch_size=10,epochs=100)

#Running the K-fold Cross-Validation
accuracies=cross_val_score(estimator=classifier, X=X_train, y=y_train, n_jobs=1, cv=10)

#Check Accuracy(bias)
mean=accuracies.mean()

#Check Variance
variance=accuracies.std()

#Part 5: Grid Search-----------------------------------------------------------
#Selecting parameters to improve and the options
parameters={'batch_size': [25,32],
            'epochs':[100,500],
            'optimizer':['adam','rmsprop']}

#Import Grid Search and build function for our ANN
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(units=11,kernel_initializer ='uniform' ,activation ='relu',input_dim=16))
    classifier.add(Dense(units=11,kernel_initializer ='uniform' ,activation ='relu'))
    classifier.add(Dense(units=7,kernel_initializer ='uniform' ,activation ='sigmoid'))
    classifier.compile(optimizer=optimizer,
    loss='categorical_crossentropy',metrics=['accuracy'])
    return classifier
    
#Assign KerasClassifier    
classifier=KerasClassifier(build_fn=build_classifier)

#Selecting parameters to improve and the options
parameters={'batch_size': [25,32],
            'epochs':[100,500],
            'optimizer':['adam','rmsprop']}

#Import Grid Search and build function for our ANN
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(units=11,kernel_initializer ='uniform' ,activation ='relu',input_dim=16))
    classifier.add(Dense(units=11,kernel_initializer ='uniform' ,activation ='relu'))
    classifier.add(Dense(units=7,kernel_initializer ='uniform' ,activation ='sigmoid'))
    classifier.compile(optimizer=optimizer,
    loss='categorical_crossentropy',metrics=['accuracy'])
    return classifier
    
#Assign KerasClassifier    
classifier=KerasClassifier(build_fn=build_classifier)

#Setting up Grid Search
grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10)

#Fit and initiate Grid Search on our ANN
grid_search=grid_search.fit(X_train,
labelbinarizer_y.inverse_transform(y_train))

#Get best parameters
best_parameters=grid_search.best_params_

#Get best accuracy
best_accuracy=grid_search.best_score_

