from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np

# Load dataset
X, y = load_iris(return_X_y=True)

# Initialize LOOCV
loo = LeaveOneOut()
model = LogisticRegression(max_iter=200)

scores = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))  # accuracy for the test sample

print("Mean accuracy:", np.mean(scores))
