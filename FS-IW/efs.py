import numpy as np
import random
import mut_inf_cla
import recursive_feature_elimination
import lasso
import mic
import chisquare
import ANOVA
import variancemaximum
import f_value
from sklearn.model_selection import train_test_split
import logging
import scipy.io as scio
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from M3V import oemmmcrowd
import os
import time

def get_data(rf):
    datas = list()
    labels = list()
    features = list()

    with open(rf, "r") as lines:
        for line in lines:
            data = line.strip().split(",")
            if data[0] == 'class':
                features = data[1:]
            else:
                datas.append(data[1:])
                labels.append(data[0])
    pos_l = len(np.array(labels)[np.where(np.array(labels) == '1')].tolist())
    pos = datas[:pos_l]
    neg = datas[pos_l:len(datas)]
    return pos, neg, features, labels

def split_data(pos, neg, m):
    p, g = len(pos), len(neg)
    n = int(len(pos)/m) + 1
    random.shuffle(pos)
    spos = [pos[i:i+n] for i in range(0, len(pos), n)]
    # for i in range(0, len(spos)):
    #     spos[i].insert(0, features)
    random.shuffle(neg)
    sneg = [neg[i:i+n] for i in range(0, len(neg), n)]
    return spos, sneg, n, p, g

def single_ranking_result(pos, neg, features_name, n, p ,g, logger):
    pos = np.array(pos, dtype='object')
    neg = np.array(neg, dtype='object')
    data = list()
    for i in range(0, len(pos)):
        a = np.concatenate((pos[i], neg[i]))
        data.append(a.tolist())
    #define different feature selection result matrix
    results = list()
    fre = 0 #record the frequency
    for i in range(0, len(data)):
        logger.info("dealing with " + str(i+1) + "-th partition data")
        if i != len(data)-1:
            fre += 1
            labels = np.array([1 if i < n else 0 for i in range(2 * n)], dtype=int).tolist()
            result = feature_rank(data[i], features_name, labels, logger)
            results.append(result)
        else:
            p1 = p - fre*n
            g1 = g - fre*n
            labels = np.array([1 if i < p1 else 0 for i in range(p1+g1)], dtype=int).tolist()
            result = feature_rank(data[i], features_name, labels, logger)
            results.append(result)
    return results

def feature_rank(datas, features_name, labels, logger):
    logger.info("MIC start...")
    mic_data = mic.run(datas, features_name, labels)
    logger.info("MIC end.")
    logger.info("Lasso start...")
    lasso_data = lasso.run(datas, features_name, labels)
    logger.info("Lasso end.")
    logger.info("chi2 start...")
    chi2_data = chisquare.run(datas, features_name, labels)
    logger.info("chi2 end.")
    logger.info("REF start...")
    ref_data = recursive_feature_elimination.run(datas, features_name, labels)
    logger.info("REF end.")
    logger.info("Mutual_information start...")
    mut_data = mut_inf_cla.run(datas, features_name, labels)
    logger.info("Mutual_information end.")
    logger.info("ANOVA start...")
    ano_data = ANOVA.run(datas, features_name, labels)
    logger.info("ANOVA end.")
    logger.info("VarianceThreshold start...")
    vm_data = variancemaximum.run(datas, features_name, labels)
    logger.info("VarianceThreshold end.")
    logger.info("f_value start...")
    fv_data = f_value.run(datas, features_name, labels)
    logger.info("f_value end.")

    result = [mic_data, lasso_data, chi2_data, ref_data, mut_data, ano_data, vm_data, fv_data]
    return result

#convert feature selection results to a matrix with mat type
def initial_labels(result, features):
    L = list()
    true_labels = list()
    for i in range(0, len(result)):
        mr = list()
        for j in range(0, len(result[i])):
            r = np.ones(len(features))
            for z in range(0, len(result[i][j])):
                if result[i][j][z] in features:
                    r[features.index(result[i][j][z])] = 2
            mr.append(r.tolist())
        L.append(mr)
    for i in L:
        T = np.sum((np.array(i)), axis=0)
        T[np.where(T < 2)] = 1
        T[np.where(T >= 2)] = 2
        true_labels.append(T.tolist())
    return L, true_labels

def save_mat_file(L, wf, true_labels):
    name1 = 'L'
    name2 = 'true_labels'
    for i in range(0, len(L)):
        result = dict()
        path = wf
        result[name1] = np.array(L[i]).T.tolist()
        true_labels[i] = np.array(true_labels[i])
        true_labels[i].shape = (len(true_labels[i]), 1)
        result[name2] = true_labels[i].tolist()
        path += str(i) + '.mat'
        scio.savemat(path, result)

def ensemble_feature_selection(path):
    results = list()
    file_name = list()
    dir = os.listdir(path)
    for i in dir:
        if os.path.splitext(i)[1] == '.mat':
            file_name.append(path + '\\' + i)
        else:
            continue
    for i in file_name:
        mm = oemmmcrowd.oemmmcrowd()
        mm.loadData(i)
        results = results + mm.train().tolist()
    return results

def get_final_result(fs, datas, feature_name):
    results = list()
    features = list()
    datas = np.array(datas)
    for j in range(len(fs)):
        x = list()
        feature = list()
        for i in range(len(fs[j])):
            if fs[j][i] == 2:
                x.append(datas[:, i].tolist())
                feature.append(feature_name[i])
            else:
                continue
        features.append(feature)
        x = np.array(x).T.tolist()
        y = [1] * int(len(x)/2) + [0] * int(len(x)/2)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        rf = RandomForestClassifier(random_state=1, n_estimators=100)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
        results.append([accuracy, recall, precision, f1_score])

    index = np.argmax(np.array(results), axis=0)
    print(results[index[3]])
    optimal_feature = features[index[3]]
    return results, optimal_feature

def run(rf, m, wf, path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_path = os.getcwd() + os.sep + 'Logs' + os.sep
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = log_path + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # logging.basicConfig(level=logging.INFO,
    #                     format='[%(asctime)s]: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置
    formatter = logging.Formatter('[%(asctime)s]: %(message)s')
    # 文件
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # 控制台
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    pos, neg, features, labels = get_data(rf)
    spos, sneg, n, p, g = split_data(pos, neg, m)
    result = single_ranking_result(spos, sneg, features, n, p, g, logger)
    L, true_labels = initial_labels(result, features)
    save_mat_file(L, wf, true_labels)
    fs = ensemble_feature_selection(path)
    results, optimal_feature = get_final_result(fs, pos+neg, features)
    logger.info(fs)
    logger.info(results)
    logger.info(optimal_feature)


if __name__ == '__main__':
    rf = "G:\赵思远\miRNA-encoded peptides\ILPEFS\特征\seq.csv"
    wf = "G:\py-workspace\\test\M3V\dna"
    path = 'G:\py-workspace\\test\M3V'
    m = 15
    run(rf, m, wf, path)