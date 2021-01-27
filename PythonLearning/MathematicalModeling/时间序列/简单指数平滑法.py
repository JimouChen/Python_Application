from statsmodels.tsa.api import SimpleExpSmoothing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    train = df[0:10392]
    test = df[10392:]
    pred = test.copy()
    fit = SimpleExpSmoothing(np.asarray(train['列名1'])).fit(smoothing_level=0.6, optimized=False)
    pred['列名2'] = fit.forecast(len(test))  # 需要预测多长

    # 画出来
    plt.figure(figsize=(16, 8))
    plt.plot(train['列名1'], label='Train')
    plt.plot(test['列名1'], label='Test')
    plt.plot(pred['列名2'], label='列名2')
    plt.legend(loc='best')
    plt.show()

    # 评估
    rms = sqrt(mean_squared_error(test['列名1'], pred))
    print(rms)
