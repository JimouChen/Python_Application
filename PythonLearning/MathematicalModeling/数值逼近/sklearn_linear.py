from sklearn import linear_model
import numpy as np

if __name__ == '__main__':
    X = np.array([8.19, 2.72, 6.39, 8.71, 4.7, 2.66, 3.78])
    Y = np.array([7.01, 2.78, 6.47, 6.71, 4.1, 4.23, 4.05])
    X = X.reshape(-1, 1)
    model = linear_model.LinearRegression()
    model.fit(X, Y)
    print('k = ', model.coef_)
    print('b = ', model.intercept_)
