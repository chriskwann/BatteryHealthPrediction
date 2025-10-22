# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 2: Input the Dataset
# Training Data
train_data = np.array([
    [25, 100, 0.05, 98, 0.98],
    [30, 150, 0.06, 96, 0.96],
    [35, 200, 0.07, 94, 0.93],
    [40, 250, 0.09, 91, 0.89],
    [45, 300, 0.10, 89, 0.85],
    [25, 120, 0.055, 97, 0.97],
    [30, 170, 0.065, 95, 0.94],
    [35, 220, 0.075, 93, 0.91],
    [40, 270, 0.095, 90, 0.88],
    [45, 320, 0.11, 87, 0.83],
    [25, 140, 0.06, 96, 0.95],
    [30, 190, 0.07, 94, 0.93],
    [35, 240, 0.08, 92, 0.90],
    [40, 290, 0.10, 89, 0.86],
    [45, 340, 0.12, 85, 0.80]
])

# Testing Data
test_data = np.array([
    [30, 160, 0.06, 95, 0.95],
    [35, 210, 0.07, 93, 0.92],
    [40, 260, 0.08, 90, 0.88],
    [45, 310, 0.10, 87, 0.84],
    [50, 360, 0.11, 84, 0.80]
])

# Step 3: Split Features and Targets
X_train = train_data[:, :-1]
y_train = train_data[:, -1]

X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Step 4: Normalize the Data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Visualize Training Data (Optional)
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y_train)), y_train, color='blue', label='Train Health Score')
plt.title("Training Battery Health Scores")
plt.xlabel("Sample Index")
plt.ylabel("Health Score")
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Build the Neural Network Model
model = Sequential([
    Dense(16, input_shape=(4,), activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')  # Output layer
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 7: Train the Model
history = model.fit(X_train_scaled, y_train, epochs=100, verbose=0)

# Step 8: Evaluate the Model
y_pred = model.predict(X_test_scaled).flatten()
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Step 9: Plot Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Step 10: Plot True vs Predicted
plt.figure(figsize=(8, 5))
plt.plot(y_test, label='True', marker='o')
plt.plot(y_pred, label='Predicted', marker='x')
plt.title('True vs Predicted Health Score')
plt.xlabel('Sample Index')
plt.ylabel('Health Score')
plt.legend()
plt.grid(True)
plt.show()
