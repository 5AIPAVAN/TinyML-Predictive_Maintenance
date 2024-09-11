import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load dataset from CSV file
df = pd.read_csv('predictive_maintenance.csv')

# 2. Drop unnecessary attributes (e.g., UDI, Product ID)
df = df.drop(columns=['UDI', 'Product ID'])

# 3. Split into features (X) and target variable (y)
X = df.drop(columns=['Target', 'Failure Type'])
y = df['Target']

# 4. Handle failure types and drop ambiguous data
# Portion of data where RNF=1
idx_RNF = df.loc[df['Failure Type'] == 'Random Failures'].index
df.drop(index=idx_RNF, inplace=True)

# Portion of data where Machine failure=1 but no failure cause is specified
idx_ambiguous = df.loc[(df['Target'] == 1) & (df['Failure Type'] == 'No Failure')].index
df.drop(index=idx_ambiguous, inplace=True)

# 5. Perform one-hot encoding for the 'Type' column
X_encoded = pd.get_dummies(X, columns=['Type'])

# 6. Perform train-test split
from sklearn.model_selection import train_test_split
X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 7. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# 8. Define and train the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs=120, batch_size=64, validation_data=(X_test_scaled, y_test))

# 9. Convert the trained model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# 10. Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# 11. Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# 12. Test the model with new input data

# Example new input data (assumed input without 'Type')
new_data = np.array([[298.1, 308.6, 1551, 42.8, 0]], dtype=np.float32)

# Perform one-hot encoding for 'Type' column (example for 'M')
encoded_type = np.array([[1, 0, 0]])  # Reshaped to 2D array for concatenation

# Concatenate new_data with encoded_type (now both are 2D arrays)
new_data = np.concatenate([new_data, encoded_type], axis=1)

# Standardize the input using the same scaler
new_data_scaled = scaler.transform(new_data)

# Reshape data to match input shape for TensorFlow Lite model
new_data_scaled = np.reshape(new_data_scaled, input_details[0]['shape'])

# Set the tensor to point to the input data
interpreter.set_tensor(input_details[0]['index'], new_data_scaled)

# Run the interpreter
interpreter.invoke()

# Extract output predictions
output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"Predicted output: {output_data}")

# Threshold to determine failure
threshold = 0.5
if output_data >= threshold:
    print("Failure predicted.")
else:
    print("No failure predicted.")

# 13. Optionally, you can visualize the training history

# Plot accuracy and loss
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()
