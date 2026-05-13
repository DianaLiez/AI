import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic passenger data from a local CSV file.
# The file is expected to include at least these columns:
# - Survived: 0 for passengers who did not survive, 1 for passengers who survived
# - Sex: passenger gender
# - Age: passenger age in years
# - Pclass: ticket class, where 1 is first class, 2 is second class, and 3 is third class
titanic = pd.read_csv("titanic.csv")

# Convert continuous age values into readable age categories.
# pd.cut places each Age value into one of the bins below:
# - 0 to 12: Child
# - 12 to 18: Teen
# - 18 to 50: Adult
# - 50 to 80: Senior
# Passengers with missing Age values will keep a missing AgeGroup value.
titanic["AgeGroup"] = pd.cut(
    titanic["Age"],
    bins=[0, 12, 18, 50, 80],
    labels=["Child", "Teen", "Adult", "Senior"]
)

# Build a summary table for the heatmap.
# Because Survived is stored as 0 or 1, the mean of Survived is the survival rate.
# Rows are grouped by both Sex and AgeGroup, while columns are split by passenger class.
# Each cell answers: "What fraction of this gender/age/class group survived?"
pivot = titanic.pivot_table(
    values="Survived",
    index=["Sex", "AgeGroup"],
    columns="Pclass",
    aggfunc="mean"
)

# Create the heatmap figure.
# annot=True prints each survival rate directly inside its heatmap cell.
# cmap="YlOrRd" uses lighter colors for lower values and darker red/orange for higher values.
# fmt=".2f" displays survival rates with two decimal places, such as 0.74.
plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot=True, cmap="YlOrRd", fmt=".2f")

# Add chart labels so the plotted dimensions are clear to the reader.
plt.title("Titanic Survival Rates by Gender, Class, and Age Group")
plt.ylabel("Gender + Age Group")
plt.xlabel("Passenger Class")

# Display the chart in a window or notebook output, depending on where the script is run.
plt.show()
