import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
housing = fetch_california_housing()

# Create dataframe
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add target column
df['Price'] = housing.target

# Show first rows
print(df.head())

# Features (X)
X = df.drop('Price', axis=1)

# Target (y)
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("R2 Score:", r2)

# Predict one example house
sample_house = [8.3252, 41, 6.9841, 1.0238,
                322, 2.5556, 37.88, -122.23]

predicted_price = model.predict([sample_house])

print("Predicted Price:", predicted_price[0])