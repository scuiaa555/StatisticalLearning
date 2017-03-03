# Load data from sklearn datasets.
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Split dataset into training and test datasets for cross validation.
from sklearn.cross_validation import train_test_split

# Parameter random_state is the seed of random number generator and is set for reproduction purpose.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardization procedure.
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  # Initialize StandardScaler class
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

# Default multiclass classification in Perceptron is One-vs-Rest.
ppn = OneVsOneClassifier(Perceptron(eta0=0.1, random_state=0, n_iter=40))
ppn2 = OneVsRestClassifier(Perceptron(eta0=0.1, random_state=0, n_iter=40))
ppn.fit(X_train_std, y_train)  # Train the model.
ppn2.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)  # Predict the data.
y_pred2 = ppn2.predict(X_test_std)
print (y_test != y_pred2).sum()
print (y_test != y_pred).sum()

# Plot the decision surface
import sys
# Add the parent folder to path.
sys.path.append('../ch02')
from plot_decision_regions import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn2, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('./figures/iris_perceptron_scikit.png', dpi=300)
plt.show()
