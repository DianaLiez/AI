import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
titanic = pd.read_csv("titanic.csv")

# Create a pivot table of survival rates
pivot = titanic.pivot_table(values="Survived", index="Sex", columns="Pclass", aggfunc="mean")

# Plot heatmap
plt.figure(figsize=(6,4))
sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Titanic Survival Rates by Gender and Class")
plt.ylabel("Gender")
plt.xlabel("Passenger Class")
plt.show()
