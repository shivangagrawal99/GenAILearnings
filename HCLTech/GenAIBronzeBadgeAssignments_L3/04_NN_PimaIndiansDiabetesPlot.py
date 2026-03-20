# Gen AI Application Developer (Level-3 to level-5)-Evening Slot
# Shivang

# 4. Write a python program to draw the neural network for the the Pima Indians Diabetes prediction problem
# Hint: Use Keras

# Dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database (Datasets/PimaIndiansDiabetes.csv)
#    Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
#    6,148,72,35,0,33.6,0.627,50,1
#    ...
#    1,93,70,31,0,30.4,0.315,23,0

# Install required packages:
# pip install numpy tensorflow-cpu matplotlib graphviz
# Install https://graphviz.gitlab.io/download/ and add to PATH

# step 1: import the libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# step 2: read the data
# fix random seed for reproducibility
np.random.seed(7)

dataset = np.loadtxt("Datasets/PimaIndiansDiabetes.csv", delimiter=",", skiprows=1)
input_X = dataset[:, 0:8]
input_y = dataset[:, 8]

# step 3: define the model
model = Sequential()
model.add(Input(shape=(8,), name="Input_Layer"))
model.add(Dense(12, activation='relu', name="Hidden_Layer_1"))
model.add(Dense(8, activation='relu', name="Hidden_Layer_2"))
model.add(Dense(1, activation='sigmoid', name="Output_Layer"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# step 4: Fit(Train) the model
model.fit(input_X, input_y, epochs=150, batch_size=10)

#evaluate the model
scores = model.evaluate(input_X, input_y, verbose=0)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# step 5: print model summary
model.summary()

# step 6: draw a simple schematic with matplotlib
# Save model diagram
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "04_NN_PimaIndiansDiabetesPlot.png")

plot_model(
    model,
    to_file=model_path,
    show_shapes=True,
    show_layer_names=True,
    show_layer_activations=True,
    show_trainable=True
)

print(f"Model diagram saved as: {model_path}")

img = mpimg.imread(model_path)
plt.imshow(img)
plt.axis('off')
plt.show()
