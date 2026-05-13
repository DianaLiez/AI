import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic dataset (replace with your file path)
titanic = pd.read_csv("titanic.csv")

# Group survival rates by gender and class
survival_rates = titanic.groupby(["Sex", "Pclass"])["Survived"].mean().unstack()

# Plot
survival_rates.plot(kind="bar")
plt.title("Titanic Survival Rates by Gender and Class")
plt.ylabel("Survival Rate")
plt.xlabel("Gender")
plt.legend(title="Class")
plt.show()
