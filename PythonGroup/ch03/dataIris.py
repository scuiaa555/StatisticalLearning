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
