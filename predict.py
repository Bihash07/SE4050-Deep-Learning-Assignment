# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('synthetic_student_data.csv')  # Replace 'your_dataset.csv' with the actual dataset filename

# Data Exploration and Visualization
# Perform data exploration, visualization, and analysis here

# Data Preprocessing
# Encode categorical variables (e.g., Gender, CourseType)
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['CourseType'] = le.fit_transform(data['CourseType'])

# Split the data into features (X) and the target (y)
X = data.drop(columns=['AcademicPerformance'])
y = data['AcademicPerformance']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (optional, but often beneficial for deep learning)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode the target variable
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Define the deep learning model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 classes: 'Good', 'Average', 'Poor'

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=50, batch_size=32, validation_split=0.1, verbose=2)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(np.argmax(y_test_encoded, axis=1), y_pred_classes)
print(f'Accuracy: {accuracy}')

# Generate classification report and confusion matrix
report = classification_report(np.argmax(y_test_encoded, axis=1), y_pred_classes)
conf_matrix = confusion_matrix(np.argmax(y_test_encoded, axis=1), y_pred_classes)
print(report)
print(conf_matrix)

# Save the trained model (optional)
model.save('academic_performance_model.h5')
