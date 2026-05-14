import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LeaveOneOut

# Example dataset with 6 samples
X = np.arange(6).reshape(-1, 1)
y = np.array([0, 1, 0, 1, 0, 1])

loo = LeaveOneOut()

# Plotting setup
plt.figure(figsize=(8, 6))

for i, (train_index, test_index) in enumerate(loo.split(X)):
    # Create an array for visualization
    mask = np.zeros(len(X))
    mask[train_index] = 1  # training samples = 1
    mask[test_index] = 2   # test sample = 2
    
    plt.scatter(range(len(X)), [i]*len(X), 
                c=mask, cmap="coolwarm", s=100, marker="s")

plt.yticks(range(len(X)), [f"Iteration {i+1}" for i in range(len(X))])
plt.xticks(range(len(X)), [f"x{i+1}" for i in range(len(X))])
plt.title("Leave-One-Out Cross-Validation Splits")
plt.xlabel("Samples")
plt.ylabel("Iterations")
plt.colorbar(label="Train/Test (1=Train, 2=Test)")
plt.show()
