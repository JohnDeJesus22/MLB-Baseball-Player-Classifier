#ANN Classifier built to assign positions to a baseball player using their stats.

#Part 1: Import Libraries and Data--------------------------------------------
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import data
dataset=pd.read_csv('MLB2016PlayerStats.txt')

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

   
Let's Classify Baseball Players Using Deep Learning! Final Part
JAN
18
Let's Classify Baseball Players Using Deep Learning! Final Part
Welcome to the final installment of our MLB baseball classifier! Thank you for coming along on the creation ride for the ANN. Now we will see if we can find the best parameters using Grid Search.
If you need to recap/ catch up, you can find all four of the original parts below:
Understanding the Data and Importing into Python

Prepping Data for ANN

Building and Testing ANN

Assessing our ANN with K-Fold Cross Validation

In the last chapter of this series we will:

Explaining what Grid Search is.
The Parameters we will improve upon.
Setup the Grid Search Code
Review the results and the new accuracy
What is Grid Search?
Grid Search is an algorithm used to determine the best parameter choices for a machine learning model. Doing this will allow you to find the best parameters to improve the model's performance. This includes the model's accuracy, precision, the number of epochs, and other metrics that assess the model's efficiency and value. Grid Search is done after K-Fold Cross-Validation since we need to assess the performance of our model first before you search for the optimal set of parameter settings.

The Parameters to Improve
#Selecting parameters to improve and the options
parameters={'batch_size': [25,32],
            'epochs':[100,500],
            'optimizer':['adam','rmsprop']}
In our case, I will model the metrics improved from the Deep Learning Course I took. The parameters are created as a dictionary where the key is the parameter and the output is the list of options of the respective parameter.

First is batch size. Remember that we want to keep the batch size small but not too small. We will provide the batch sizes of 25 and 32 to test. For the number of epochs, we will assess if 100 epochs or 500 epochs to see if more training runs will improve the ANN's performance. 

Lastly, we will see if our original optimizer is still the best, or should we switch to the Adam optimizer function. Adam stands for Adaptive Moment Estimation. According to this post by Sebastian Ruder, Adam does the same actions as Rmsprop, but also keeps track of the "exponentially decaying averages of past gradients". You should also check out Sebastian's blog post to get an overview of other Optimization Algorithms. Perhaps there are others you would like to try if they are appropriate for this model.

Setup the Grid Search Code
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
First, we will need to import the GridSearchCV class from scikit-learn. Since we are utilizing a class from scikit-learn learn and our model is made with Keras, we will need to encase our model into a function to apply into the KerasClassifier (import this from Keras if you didn't do so yet). It is the same function as we did for the K-Fold Cross-Validation except we will have an optimizer parameter. This allows us to change the optimizer function for our model while we do the Grid Search. Finally, we will assign our KerasClassifier with our function to the classifier variable.

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

