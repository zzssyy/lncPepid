from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def fvalue(datas, features_name, label):
    X = np.array(datas)
    y = np.array(label)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model1 = SelectKBest(f_regression, k=2)  # 选择k个最佳特征
    model1.fit_transform(X, y)
    result = [(x, y) for x, y in zip(features_name, model1.scores_)]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return [x[0] for x in result if abs(x[1]) > 0.0000000000001]

# def run(csvfile,logger):
#     logger.info('mut_inf_cla start...')
#     feature_list = mut_inf_cla(csvfile)
#     logger.info('mut_inf_cla end.')
#     return feature_list

def run(datas, features_name, labels):
    feature_list = fvalue(datas, features_name, labels)
    return feature_list

if __name__ == '__main__':
    res = mut_inf_cla('G:\MRMD2.0\experimental_data\dna.csv')
    print(res)