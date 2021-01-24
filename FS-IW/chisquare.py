from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def chi2_(datas, features_name, label):
    X = np.array(datas)
    y = np.array(label)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model1 = SelectKBest(chi2, k=2)  # 选择k个最佳特征
    model1.fit_transform(X,y)
    result = [(x, y) for x, y in zip(features_name, model1.scores_)]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return [x[0] for x in result if abs(x[1]) > 0.0000000000001]

# def run(csvfile,logger):
#     logger.info('chi2 start...')
#     feature_list = chi2_(csvfile)
#     logger.info('chi2 end.')
#     return feature_list

def run(datas, features_name, labels):
    feature_list = chi2_(datas, features_name, labels)
    return feature_list

if __name__ == '__main__':
    res = chi2_('../mixfeature_frequency_DBD.csv')
    print(res)