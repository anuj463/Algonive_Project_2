import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Task_2_airbnb_sample_dataset.csv")

print(df.head())
print(df.info())

# -----------------------------
#  Data Cleaning
# -----------------------------
df.dropna(subset=['price'], inplace=True)

# Fill missing values
df['reviews_per_month'].fillna(0, inplace=True)
df['name'].fillna("Unknown", inplace=True)
df['host_name'].fillna("Unknown", inplace=True)

# Remove extreme prices (outliers)
df = df[df['price'] < 500]

# -----------------------------
#  EDA
# -----------------------------

# Price distribution
plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=50)
plt.title("Price Distribution")
plt.show()

# Room type vs price
plt.figure(figsize=(8,5))
sns.boxplot(x='room_type', y='price', data=df)
plt.title("Room Type vs Price")
plt.show()

# Neighborhood analysis
plt.figure(figsize=(10,6))
df.groupby('neighbourhood_group')['price'].mean().plot(kind='bar')
plt.title("Average Price by Area")
plt.show()

# -----------------------------
#  Feature Engineering
# -----------------------------
df['availability_365'] = df['availability_365']

# Convert categorical to numeric
df = pd.get_dummies(df, columns=['room_type','neighbourhood_group'], drop_first=True)

# -----------------------------
#  Price Prediction Model
# -----------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Select features
features = ['minimum_nights','number_of_reviews','reviews_per_month','availability_365']
features += [col for col in df.columns if 'room_type_' in col or 'neighbourhood_group_' in col]

X = df[features]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# -----------------------------
#  Visualization (Actual vs Predicted)
# -----------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()