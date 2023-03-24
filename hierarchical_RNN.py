import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv('contacts_mock_vs_reak,csv')

# Group the data by nurse ID
grouped = df.groupby('activity_ID')

# Create a list to hold the sequences for each nurse
sequences = []

# Loop through each nurse
for name, group in grouped:
    # Get the surface values for the nurse
    surfaces = group['surface'].values
    
    # Add the surfaces to the sequences list
    sequences.append(surfaces)

# Pad the sequences with zeros
padded_sequences = pad_sequences(sequences, padding='post')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

#In that case, you can modify the code to create an LSTM layer for each nurse. Here's how you can do it:


import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Dropout, Concatenate
from tensorflow.keras.models import Model

# Define the input shape
input_shape = (None, 1)

# Define the input layer
inputs = Input(shape=input_shape)

# Define the LSTM layers for each nurse
lstms = []
for i in range(400):
    lstms.append(LSTM(64, return_sequences=True)(inputs))

# Concatenate the LSTM outputs
concatenated = Concatenate()(lstms)

# Define the output layer
outputs = Dense(1, activation='sigmoid')(concatenated)

# Define the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

#This code creates an LSTM layer for each nurse using a for loop, and concatenates the outputs of the LSTM layers. The concatenated output is then passed through a Dense layer with a sigmoid activation function. The model is compiled with the Adam optimizer and binary cross-entropy loss function. The model is then trained on the training data for 10 epochs with a batch size of 32 and validated on the test data.
