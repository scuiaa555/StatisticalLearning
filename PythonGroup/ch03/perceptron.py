from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler  # StandardScaler is a class

sc = StandardScaler()  # initialization
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

ppn = OneVsOneClassifier(Perceptron(eta0=0.1, random_state=0, n_iter=40))
ppn2 = OneVsRestClassifier(Perceptron(eta0=0.1, random_state=0, n_iter=40))
ppn.fit(X_train_std, y_train)
ppn2.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
y_pred2 = ppn2.predict(X_test_std)
print (y_test != y_pred2).sum()
print (y_test != y_pred).sum()

# import matplotlib.pyplot as plt
# plt.scatter(X_train_std[:,0],X_train_std[:,1],color='red')

aaaa = 0
