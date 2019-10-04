
from sklearn.metrics import mean_squared_error
from math import sqrt

def rms_data(y_actual,y_predicted):
    X = sqrt(mean_squared_error(y_actual, y_predicted))
    return X