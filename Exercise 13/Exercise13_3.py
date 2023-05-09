# # logistic regression for multi-class classification using built-in one-vs-rest
# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# # define dataset
# X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=4, random_state=1)
# # define model
# model = LogisticRegression(multi_class='ovr')
# # fit model
# model.fit(X, y)
# # make predictions
# yhat = model.predict(X)


# logistic regression for multi-class classification using a one-vs-rest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=4, random_state=1)
# define model
model = LogisticRegression()
# define the ovr strategy
ovr = OneVsRestClassifier(model)
# fit model
ovr.fit(X, y)
# make predictions
yhat = ovr.predict(X)