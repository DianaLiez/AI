from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Load dataset
X, y = load_iris(return_X_y=True)

# Create model
model = DecisionTreeClassifier()

# Perform 5-fold cross validation
scores = cross_val_score(model, X, y, cv=5)

print("Scores:", scores)
print("Average Accuracy:", scores.mean())