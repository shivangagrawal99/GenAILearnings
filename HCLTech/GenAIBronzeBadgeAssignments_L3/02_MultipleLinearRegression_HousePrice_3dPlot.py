# Gen AI Application Developer (Level-3 to level-5)-Evening Slot
# Shivang

# 2. For the dataset (Datasets/HousePrice2Feature.csv) which is multi variate liner regression problem,
# give me the python program that does multi variate linear regresssion taking the above dataset and predicts the price of the house.
# Input is number of bed rooms and sq feet, output is price of the house.
# Also draw a plot of the model which is multi variate linear regrsssion. Predict the price of a 2000 sq feet 3 bed room house

# python implementation of multivariate linear Regression on using house price prediction dataset 

# step 1: import the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# step 2: reading the data and splitting it to input and output.
# Shivang - Added header=None as first row is not the header but the actual data
# home.csv contains 3 columns (sq feet, rooms, price)
dataset = pd.read_csv('Datasets/HousePrice2Feature.csv', header=None)
dataset.columns = ['Sq.Feet', 'Bedrooms', 'Price'] # Naming columns for easier plotting

# inputx = all rows, and first 2 columns (sq feet and rooms)
inputx = dataset.iloc[:, 0:2].values

# outputx = all rows, and last column (price)
outputy = dataset.iloc[:, 2].values

# print(outputy.size)
# print(outputy)

# step 3: select one fourth of the data for testing and two thirds for training.
# Shivang - random_state=42 as it is the "Answer to the Ultimate Question of Life, the Universe, and Everything".
input_train, input_test, output_train, output_test = train_test_split(inputx, outputy, test_size = 1/4, random_state = 42)

# step 4: selecting the simple Linear Regression model
model = LinearRegression()
print("\nThe parameters of the model are:\n\n",model.get_params())

# Shivang - since inputx as two columns, below will make is multivariate liner regression. model.fit() actually start training the model on the data.
print("\nThe model we are using is: ", model.fit(input_train, output_train))

# step 5: testing or model prediction using testinput
squarefeet = float(input("\nGive square feet of the house: "))
bedrooms = float(input("\nGive the number of bed rooms in the house: "))
testinput = [[squarefeet,bedrooms]]
testpredicted_output = model.predict(testinput)
print('\nThe test input is as follows: ',testinput) 
print('\nThe predicted house price is: ',testpredicted_output) 

# waiting for user input to proceed with plotting. actual inout does not matter, just pausing the program
yes = input("\nCan I proceed with plotting?\n")

# step 6: Printing the testing results
print("\nThe test input (square feet and the number of bed rooms) is as follows: \n")
print(input_test)

# model predicting the Test set results
predicted_output = model.predict(input_test)
print("\nThe predicted price of the house for the test input is as follows: \n")
print(predicted_output)

# Shivang
# Calculate metrics
r2 = r2_score(output_test, predicted_output)
mae = mean_absolute_error(output_test, predicted_output)

print("\n--- Model Correctness Report ---")
print(f"Model Accuracy (R2 Score): {r2*100:.2f}%")
print(f"Average Error (in Price): {mae:,.2f}")

# Compare a few side-by-side
print("\nActual vs Predicted (First 5):")
for actual, pred in zip(output_test[:5], predicted_output[:5]):
    print(f"Actual: {actual:,.2f} | Predicted: {pred:,.2f}")

# Shivang
# Visualize 3-d plot

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')

# 1. Plot the actual data points
ax.scatter(inputx[:, 0], inputx[:, 1], outputy, color='red', alpha=0.5, label='Actual Data', marker='o')

# 2. Highlight the user input (Large Yellow Star)
ax.scatter(squarefeet, bedrooms, testpredicted_output, 
           color='yellow', s=100, marker='*', edgecolor='black', 
           label=f'Prediction{squarefeet, bedrooms}: {testpredicted_output}')

# 3. Create a meshgrid to represent the Model (the Plane)
# We create a range of values for SqFt and Bedrooms
x_range = np.linspace(inputx[:, 0].min(), inputx[:, 0].max(), 50)
y_range = np.linspace(inputx[:, 1].min(), inputx[:, 1].max(), 50)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)

# 4. Predict the price for every point on that grid
# Flatten the grid to pass it to the model
flat_grid = np.c_[x_mesh.ravel(), y_mesh.ravel()]
z_predict = model.predict(flat_grid).reshape(x_mesh.shape)

# 5. Plot the surface
surf = ax.plot_surface(x_mesh, y_mesh, z_predict, alpha=0.4, cmap='viridis')

# 6. Labels and Legend
plt.title(f'Shivang House Price Prediction\nModel Accuracy (R2): {r2*100:.2f}%')
ax.set_xlabel('Square Feet')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')
ax.legend()
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.show()
