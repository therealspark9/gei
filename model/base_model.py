#! encoding=utf8
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

class Model(object):

    def __init__(self):
        self.model = None

    def fit(self, x_train, y_train, x_test):
#       print('fit in model class')
        regressor.fit(x_train, y_train)
        pass

    def predict(self, x_test):
        pred = regressor.predict(x_test)
        pass