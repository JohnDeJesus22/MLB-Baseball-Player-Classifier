# MLB Classifier with Dropout
#Add the 3 new (*) lines of code with comments attached to the original ANN in
#MLBPlayerClassifier.py to create dropout or this block of code with original.

#Importing keras and classes
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout #class to create Dropouts*

# Initializing the ANN
classifier=Sequential()

#Initial Layer and First Hidden Layer
classifier.add(Dense(input_dim=16, kernel_initializer ='uniform', activation ='relu', units=11))
classifier.add(Dropout(p=0.1))#For first hidden node layer p is percentage of Nodes dropped.*

#Second Hidden Layer
classifier.add(Dense(units=11, kernel_initializer ='uniform', activation ='relu'))
classifier.add(Dropout(p=0.1))#For second hidden node layer. p is percentage of Nodes dropped.*

#Output Layer
classifier.add(Dense(units=7, kernel_initializer ='uniform', activation ='softmax'))

#Compile ANN
classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#Fit Training Set to ANN
classifier.fit(X_train,y_train, batch_size=10, epochs=100)