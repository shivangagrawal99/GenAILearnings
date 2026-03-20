# Gen AI Application Developer (Level-3 to level-5)-Evening Slot
# Shivang

# 1. Can you build a Multivariate(Multiple) Linear Regression model that can predict the product sales
# based on the advertising budget allocated to different channels.
# The features are TV Budget ($), Radio Budget ($), Newspaper Budget ($)
# and the output is Sales (units)

# Dataset: https://www.kaggle.com/datasets/bumba5341/advertisingcsv (Datasets/Advertising.csv)
#	"","TV","Radio","Newspaper","Sales"
#	"1",230.1,37.8,69.2,22.1
#	...
#	"200",232.1,8.6,8.7,13.4


# step 1: import the libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# step 2: reading the data from the Advertising.csv file
df = pd.read_csv('Datasets/Advertising.csv')

# inputX = all rows and 3 columns (TV, Radio, Newspaper). Note the first column is the row index so we skip it.
input_X = df.iloc[:, 1:4].values
print(input_X[:5])

# outputY = all rows and last column (Sales)
output_y = df.iloc[:, 4].values
print(output_y[:5])

# step 3: select one fourth of the data for testing and three fourths for training.
input_train, input_test, output_train, output_test = train_test_split(input_X, output_y, test_size = 1/4, random_state = 42)

# step 4: select the simple Linear Regression Model and train on the training data.
# As inputX has 3 columns, below will make it Multivariate Linear Regression.
mlr_model = LinearRegression()
mlr_model.fit(input_train, output_train)

# step 5: testing/model prediction
print()
tvBudget = float(input("Enter the TV budget: "))
radioBudget = float(input("Enter the Radio budget: "))
paperBudget = float(input("Enter the Newspaper budget: "))

testInput = [[tvBudget, radioBudget, paperBudget]]
testPredictedOutput = mlr_model.predict(testInput)

print()
print("The Budgets(TV, Radio,Newspaer) are: ", testInput)
print("The predicted Sales is: %s units" % testPredictedOutput)
