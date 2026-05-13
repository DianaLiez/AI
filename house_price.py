import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample data
data = {
    'size': [1000, 1500, 1800, 2400, 3000],
    'bedrooms': [2, 3, 3, 4, 4],
    'age': [10, 5, 8, 2, 1],
    'price': [200000, 300000, 340000, 450000, 540000]
}

# Create dataframe
df = pd.DataFrame(data)

# Features and target
X = df[['size', 'bedrooms', 'age']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict new house price, 2000sqft, 3 bedrooms, 4 years old
new_house = [[2000, 3, 4]]

predicted_price = model.predict(new_house)

print("Predicted price:", predicted_price[0])