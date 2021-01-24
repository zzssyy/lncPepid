from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
def lasso(datas, features_name, label):
    X = np.array(datas)
    y = np.array(label)

    lasso = Lasso(alpha=.0001, random_state=1, max_iter=2000)
    lasso.fit(X, y)
    # print(lasso.coef_)
    ranks = {}
    result = [(x, y) for x, y in zip(features_name, lasso.coef_)]
    result = sorted(result, key=lambda x: abs(x[1]), reverse=True)

    return [x[0] for x in result if abs(x[1]) > 0.0000000000001]

# def run(csvfile,logger):
#     logger.info('lasso start...')
#     feature_list = lasso(csvfile)
#
#     logger.info('lasso end.')
#     return feature_list

def run(datas, features_name, labels):
    feature_list = lasso(datas, features_name, labels)
    return feature_list
#
# filepath =  r'J:\多设备共享\work\MRMD2.0-github\mixfeature_frequency_DBD.csv'
# result = lasso(filepath)