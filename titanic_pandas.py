import pandas as pd

# Load dataset (assuming CSV format)
titanic = pd.read_csv("titanic.csv")

# Quick survival rate by gender
print(titanic.groupby("Sex")["Survived"].mean())

# Survival rate by class
print(titanic.groupby("Pclass")["Survived"].mean())
