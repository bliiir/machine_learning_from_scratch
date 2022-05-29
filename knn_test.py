import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, y  = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# print(X_train.shape)
# print(X_train[0])
#
# print(y_train.shape)
# print(y_train)
#
plt.figure()
plt.scatter(X[:,2], X[:, 3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

# a = [1,1,1,2,2,3,4,5,5,5,5]
# from collections import Counter
# most_common = Counter(a).most_common(1)
# print(most_common[0][0])

# Classifier
clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Accuracy
acc = np.sum(predictions == y_test) / len(y_test)
breakpoint()
