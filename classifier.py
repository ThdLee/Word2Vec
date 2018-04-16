
import numpy as np
from sklearn.linear_model import LogisticRegression

data = np.loadtxt('vector.csv', delimiter=',')
train_data = data[:25000, 1:]
train_label = np.append(np.ones(12500, dtype=np.int), np.zeros(12500, dtype=np.int))

test_data = data[25000:, 1:]
test_label = np.append(np.ones(12500, dtype=np.int), np.zeros(12500, dtype=np.int))

print('Fitting')
classifier = LogisticRegression()
classifier.fit(train_data, train_label)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

print(classifier.score(train_data, train_label))