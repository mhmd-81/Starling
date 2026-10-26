import numpy as np 
from sklearn.linear_model import LinearRegression


def make_windows(x,lag):
    x=np.asanyarray(x,dtype=float)
    if len(x) <= lag:
        raise ValueError(f'length {len(x)} is too short for {lag}')
    idx = np.arange(lag)[None, :] + np.arange(len(x) - lag)[:, None]
    return x[idx], x[lag:]


class Linear_model():
    def __init__(self,lag=32):
        self.lag = lag
        self.model = LinearRegression()

    def fit(self,x):
        X, y = make_windows(x, self.lag)
        self.model.fit(X,y)
        return self
    def score(self,x):
        X, y = make_windows(x, self.lag)
        err = np.abs(y - self.model.predict(X))
        return np.concatenate([np.full(self.lag, np.nan), err])