"""
This Python script is for predicting car purchase amounts based on customer features.

The script begins by importing necessary Python libraries for data handling, visualization and machine learning.

It then loads a CSV dataset of car purchasing data into a DataFrame and examines the first and last five records.

Some preliminary data analysis is conducted using seaborn's pairplot to observe the relationships among variables.

The features and target variables are then prepared by selecting relevant columns from the DataFrame.

The script then scales the feature and target variables using MinMaxScaler from sklearn.preprocessing, which transforms the data such that its distribution will have a range from 0 to 1.

The data is then split into training and testing sets for model validation.

A sequential neural network model is built using keras with two hidden layers and one output layer. The model is compiled with an 'adam' optimizer and 'mean_squared_error' loss function, appropriate for regression problems.

The model is trained on the training set for 100 epochs, with a batch size of 50 and validation split of 0.2, then evaluated using the model's loss history.

Finally, a sample test data is used to predict a purchase amount with the trained model, which is then printed out.
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Data Loading
# Loading the dataset using pandas and specifying the encoding type
df = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')

# Checking the first 5 records of the dataframe
df.head(5)

# Checking the last 5 records of the dataframe
df.tail(5)

# Descriptive Statistics
print(df.describe())

# Checking for Null Values
print(df.isnull().sum())

# Correlation Analysis
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Outlier Detection - boxplot for Age
sns.boxplot(df['Age'])
plt.show()

# Distribution Analysis - histogram for Annual Salary
df['Annual Salary'].hist(bins=30)
plt.show()

# Pairwise relationships
sns.pairplot(df)
plt.show()

# Data Visualization
# Plotting pairplot for the entire dataframe using seaborn for initial data analysis
sns.pairplot(df)

# Data Preparation
# Creating the feature matrix X by dropping unnecessary columns from the dataframe
X = df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)

# Creating the target variable y which is the 'Car Purchase Amount' column
y = df['Car Purchase Amount']

# Checking the shape of X and y
X.shape
y.shape

# Data Scaling
# Scaling the feature matrix using MinMaxScaler for better model performance
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Checking the shape of the scaled feature matrix
X_scaled.shape

# Checking the maximum and minimum values identified by the scaler
scaler.data_max_
scaler.data_min_

# Reshaping the target variable to make it suitable for the scaler
y = y.values.reshape(-1,1)

# Checking the shape of the reshaped target variable
y.shape

# Scaling the target variable
y_scaled = scaler.fit_transform(y)

# Checking the shape of the scaled target variable
y_scaled.shape

# Model Preparation
# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.15)

# Checking the shapes of the training and testing sets
X_train.shape
X_test.shape

# Importing necessary keras modules
import tensorflow.keras
from keras.models import Sequential 
from keras.layers import Dense 

# Model Building
# Building a sequential model with two hidden layers and one output layer
model = Sequential()
model.add(Dense(15, input_dim = 5, activation ='relu')) # First hidden layer with 15 neurons
model.add(Dense(15, activation = 'relu')) # Second hidden layer with 15 neurons
model.add(Dense(1, activation = 'linear')) # Output layer with 1 neuron (since it's a regression problem)

# Checking the model summary
model.summary()

# Compiling the model with Adam optimizer and mean squared error loss function (suitable for regression problems)
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Model Training
# Training the model for 100 epochs with a batch size of 50 and a validation split of 0.2
epochs_hist = model.fit(X_train , y_train, epochs = 100, batch_size = 50 , verbose = 1, validation_split = 0.2)

# Model Evaluation
# Checking the keys in the history of the model
epochs_hist.history.keys()

# Plotting the model's loss progress during training
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress during Training')
plt.ylabel('Training and Validation Loss')
plt.xlabel('Epoch number')
plt.legend(['Training Loss', 'Validation Loss'])

# Model Testing
# Testing the model with a sample test data
X_test = np.array([[1, 50, 50000, 10000, 600000]])
y_predict = model.predict(X_test)

# Printing the predicted purchase amount
print('Expected Purchase amount', y_predict)
