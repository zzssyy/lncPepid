from minepy import MINE
from multiprocessing import Pool, Manager
import psutil
import datetime
import numpy as np

def multi_processing_mic(datas, features_name, label):
    datas = np.array(datas).T.tolist()
    datas.insert(0, label)
    datas = np.array(datas).T
    n = psutil.cpu_count(logical=False)
    n = 1
    pool = Pool()
    manager = Manager()
    dataset = datas
    mic_score = manager.dict()
    features_and_index = manager.dict()
    features_queue = manager.Queue()
    i = 1

    for name in np.array(features_name):
        features_and_index[name] = i
        features_queue.put(name)
        i+=1
    for i in range(n):
        pool.apply_async(micscore, (dataset, features_queue, features_and_index, mic_score))
    pool.close()
    pool.join()
    mic_score =[(a, b) for a, b in mic_score.items()]
    mic_score = sorted(mic_score, key=lambda x:x[1], reverse=True)
    # mic_features =[x[0] for x in mic_score]
    mic_features = [x[0] for x in mic_score if abs(x[1]) > 0.0000000000001]
    return mic_features

def micscore(dataset, features_queue, features_and_index, mic_score):
    #print('. ',end='')

    mine = MINE(alpha=0.6, c=15)
    Y = dataset[:,0]

    while not features_queue.empty():
        name = features_queue.get()
        i = features_and_index[name]
        X=dataset[:,i]

        mine.compute_score(X, Y)
        score = mine.mic()
        mic_score[name] = score

    return mic_score

# def run(filecsv,logger):
#     logger.info('mic start...')
#     datas = readData(filecsv)
#     'mic,features_name = micscore(datas)'
#     mic, features_name=multi_processing_mic(datas)
#     #print()
#
#     logger.info('mic end.')
#     return mic,list(features_name)

def run(datas, features_name, labels):
    mic = multi_processing_mic(datas, features_name, labels)
    return mic

if __name__ == '__main__':
    a = datetime.datetime.now()
    mic = run('G:\MRMD2.0\experimental_data\dna.csv')
    print(mic)
    b = datetime.datetime.now()
    print((b-a).seconds)   #427



