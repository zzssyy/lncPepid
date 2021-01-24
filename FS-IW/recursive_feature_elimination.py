from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def ref_(datas, features_name, label):
    X = np.array(datas)
    y = np.array(label)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    estimator = LinearSVC(max_iter=2000)
    selector = RFE(estimator=estimator)
    selector.fit(X, y)
    sup = selector.support_.tolist()
    t = list()
    for i in sup:
        if i is True:
            t.append(i)
    l = len(t)
    result = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name))
    return [result[x][1] for x in range(len(result)) if x <= l]

# def run(csvfile,logger):
#     logger.info('ref start...')
#     feature_list = ref_(csvfile)
#     logger.info('ref end.')
#     return feature_list

def run(datas, features_name, labels):
    feature_list = ref_(datas, features_name, labels)
    return feature_list

if __name__ == '__main__':
    res = ref_('../mixfeature_frequency_DBD.csv')
    print(res)