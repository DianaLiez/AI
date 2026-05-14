from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Define model
model = RandomForestClassifier()

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions={
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None]
    },
    n_iter=5,
    cv=5,
    random_state=42
)

random_search.fit(X, y)

print(random_search.best_params_)