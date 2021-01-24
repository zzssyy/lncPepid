from sklearn.feature_selection import VarianceThreshold
import numpy as np

def vm(datas, features_name, label):
    X = np.array(datas)
    y = np.array(label)
    vt = VarianceThreshold()
    vt.fit(X)
    result = [(x, y) for x, y in zip(features_name, vt.variances_)]
    result = sorted(result, key=lambda x: abs(x[1]), reverse=True)
    return [x[0] for x in result if abs(x[1]) > 0.1]

def run(datas, features_name, labels):
    feature_list = vm(datas, features_name, labels)
    return feature_list

