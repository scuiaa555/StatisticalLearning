import numpy as np
import pandas as pd

# url = "https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.digits/train.3"
# s = requests.get(url).content
# c = pd.read_csv(io.StringIO(s.decode('utf-8')))
# data = np.array(c,dtype='float32');
df = pd.read_csv('./Handwriting/train.0.txt')
data0 = np.array(df, dtype='float32')
df = pd.read_csv('./Handwriting/train.1.txt')
data1 = np.array(df, dtype='float32')
df = pd.read_csv('./Handwriting/train.2.txt')
data2 = np.array(df, dtype='float32')
df = pd.read_csv('./Handwriting/train.3.txt')
data3 = np.array(df, dtype='float32')
df = pd.read_csv('./Handwriting/train.4.txt')
data4 = np.array(df, dtype='float32')
df = pd.read_csv('./Handwriting/train.5.txt')
data5 = np.array(df, dtype='float32')
df = pd.read_csv('./Handwriting/train.6.txt')
data6 = np.array(df, dtype='float32')
df = pd.read_csv('./Handwriting/train.7.txt')
data7 = np.array(df, dtype='float32')
df = pd.read_csv('./Handwriting/train.8.txt')
data8 = np.array(df, dtype='float32')
df = pd.read_csv('./Handwriting/train.9.txt')
data9 = np.array(df, dtype='float32')

# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=200)
# data=pca.fit_transform(data3)
# data=pca.inverse_transform(data)
#
# import matplotlib.pyplot as plt

# plt.plot(pca.explained_variance_ratio_, "o", linewidth=2)
# plt.axis('tight')
# plt.xlabel('n_components')
# plt.ylabel('explained_variance_ratio_')
# plt.show()

# img1 = np.reshape(data[2, :], (16, 16));
# imgshow = plt.imshow(img1, cmap='gray')

X = np.concatenate((data0, data1, data2, data3, data4, data5, data6, data7, data8, data9), axis=0)
y = np.concatenate((0 * np.ones((data0.shape[0])), 1 * np.ones((data1.shape[0])), 2 * np.ones((data2.shape[0])),
                    3 * np.ones((data3.shape[0])), 4 * np.ones((data4.shape[0])), 5 * np.ones((data5.shape[0])),
                    6 * np.ones((data6.shape[0])), 7 * np.ones((data7.shape[0])), 8 * np.ones((data8.shape[0])),
                    9 * np.ones((data9.shape[0])),), axis=0)

# Split dataset into training and test datasets for cross validation.
from sklearn.cross_validation import train_test_split

# Parameter random_state is the seed of random number generator and is set for reproduction purpose.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print "Num of training:", X_train.shape[0]
print "Num of test:", X_test.shape[0]

# Standardization procedure.
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  # Initialize StandardScaler class
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

import time

# Logistic regression
from sklearn.linear_model import LogisticRegression

start = time.time()
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
end = time.time()
y_pred = lr.predict(X_test_std)  # Predict the data.
print "Logistic Regression:", (y_test != y_pred).sum()
print "Time elapsed:", end - start

# SVM
from sklearn.svm import SVC

start = time.time()
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
end = time.time()
y_pred = svm.predict(X_test_std)
print "SVM:", (y_test != y_pred).sum()
print "Time elapsed:", end - start

# Kernel SVM

start = time.time()
svm2 = SVC(kernel='rbf', C=1.0, random_state=0)
svm2.fit(X_train_std, y_train)
end = time.time()
y_pred = svm2.predict(X_test_std)
print "SVM with rbf kernel:", (y_test != y_pred).sum()
print "Time elapsed:", end - start

# Decision tree
from sklearn.tree import DecisionTreeClassifier

start = time.time()
tree = DecisionTreeClassifier(criterion='entropy', max_depth=7, random_state=0)
tree.fit(X_train_std, y_train)
end = time.time()
y_pred = tree.predict(X_test_std)
print "Decision tree:", (y_test != y_pred).sum()
print "Time elapsed:", end - start

# KNN
from sklearn.neighbors import KNeighborsClassifier

start = time.time()
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
end = time.time()
y_pred = knn.predict(X_test_std)
print "KNN:", (y_test != y_pred).sum()
print "Time elapsed:", end - start