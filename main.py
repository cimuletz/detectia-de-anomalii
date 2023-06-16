# Implementation of pyOD classes.

# Class initialization and call of the evaluate functions taken from 
# pyOD public examples https://github.com/yzhao062/pyod/tree/master/examples
from pyod.models.deep_svdd import DeepSVDD
from data_generator import generate_data_clusters
from pyod.models.ecod import ECOD
from pyod.models.loda import LODA
from pyod.models.rod import ROD
from pyod.models.iforest import IForest
from pyod.models.lunar import LUNAR
from pyod.models.so_gaal import SO_GAAL
from pyod.models.xgbod import XGBOD
from pyod.models.inne import INNE
from pyod.models.lscp import LSCP
from pyod.models.anogan import AnoGAN
from pyod.models.hbos import HBOS
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.cd import CD
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from my_data_generator import generateData, generateDataRotated, generateCovarianceMatrix, generateDataAvg
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tsne import TSNEGen

# constants
dimensions = None
dataset_size = None
dataset_size_test = None
no_classes = None
no_classes_test = None
train_anomalies = None
test_anomalies = None
path = "data_config.yml"

def evaluate(clf_name, rotation = None):
    roc = []

    contamination = train_anomalies * int(use_train_anomalies)
    print(contamination)
    if clf_name == 'DeepSVDD':
        clf = DeepSVDD(batch_size = 32, contamination = contamination, verbose=0, hidden_neurons=[64])
    if clf_name =='ECOD':
        clf = ECOD(contamination = contamination)
    if clf_name =='INNE':
        clf = INNE(contamination = contamination)
    if clf_name == 'LSCP':
        clf = LSCP(contamination = contamination)
    if clf_name == 'XGBOD':    
        clf = XGBOD(silent = False)
    if clf_name == 'SO_GAAL':
        clf = SO_GAAL(stop_epochs=30, lr_d=0.01, lr_g=0.0001, momentum=0.9, contamination=contamination)
    if clf_name == 'AnoGAN':
        clf = AnoGAN(contamination = contamination)
    if clf_name == 'LODA':
        clf = LODA(contamination=contamination)
    if clf_name == 'LOF':
        clf = LOF(contamination = contamination)
    if clf_name == 'kNN':
        clf = kNN(contamination = contamination)
    if clf_name == 'HBOS':
        clf = HBOS(contamination = contamination)
    if clf_name == 'OCSVM':
        clf = OCSVM(contamination = contamination, verbose = True)
    if clf_name == 'CD':
        clf = CD(contamination = contamination)

    x_train = []
    y_train = []
    if configs_dict['random'][0] == 0:
        for i in range(no_classes):
            x, y= generateData(ratio = train_anomalies, dimensions = dimensions, dataset_size = dataset_size, use_train_anomalies= use_train_anomalies
                            ,config = configs_dict)
            x = x.reshape((dataset_size, dimensions * dimensions))

            x_train.append(x)
            y_train.append(y)

        # tsne_train = torch.stack(x_train)
        # tsne = TSNEGen(tsne_train, y_train)
        # tsne.generateTSNE()
        x_train = torch.cat(x_train).squeeze()
        y_train = np.array([np.array(tensor) for tensor in y_train]).flatten()
        clf.fit(x_train, y_train)

        for i in range(no_classes_test):
            x, y= generateData(ratio = test_anomalies, dimensions = dimensions, dataset_size = dataset_size_test, use_train_anomalies = 1\
                            ,config = configs_dict)
            x = x.reshape((dataset_size_test, dimensions * dimensions))


            if clf_name == 'CD':
                y_test_pred = clf.predict(np.append(x, y.reshape(-1,1), axis=1))  # outlier labels (0 or 1)
                y_test_scores = clf.decision_function(np.append(x, y.reshape(-1,1), axis=1))  # outlier scores

            # get the prediction on the test data
            else:
                y_test_pred = clf.predict(x)  # outlier labels (0 or 1)
                y_test_scores = clf.decision_function(x)  # outlier scores
            print("\nOn Test Data:")
            evaluate_print(clf_name, y, y_test_scores)
            roc.append(roc_auc_score(y, y_test_scores))
    
    else:
        x_train, x_test, y_train, y_test = generateDataRotated(ratio = test_anomalies, dimensions = dimensions, dataset_size = dataset_size, use_train_anomalies = 1\
                            ,config = configs_dict)

        for i in range(len(x_train)):
            for j in range(i, len(x_train)):
                print(i, j)
                tsne_train = torch.stack((x_train[i], x_train[j]))
                tsne = TSNEGen(tsne_train)
                tsne.generateTSNE()
        x_train = torch.cat(x_train).squeeze()
        y_train = np.array([np.array(tensor) for tensor in y_train]).flatten()
        clf.fit(x_train, y_train)
           
        for i in range(no_classes_test):
            x, y= x_test[i], y_test[i]
            x = x.reshape((dataset_size, dimensions * dimensions))
            
            if clf_name == 'CD':
                y_test_pred = clf.predict(np.append(x, y.reshape(-1,1), axis=1))  # outlier labels (0 or 1)
                y_test_scores = clf.decision_function(np.append(x, y.reshape(-1,1), axis=1))  # outlier scores

            # get the prediction on the test data
            else:
                y_test_pred = clf.predict(x)  # outlier labels (0 or 1)
                y_test_scores = clf.decision_function(x)  # outlier scores
            print("\nOn Test Data:")
            evaluate_print(clf_name, y, y_test_scores)
            roc.append(roc_auc_score(y, y_test_scores))
    
    return np.array(roc)

if __name__ == "__main__":

    configs_dict = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    # train_data, _, train_labels, _ = generate_data_clusters(n_train=dataset_size, n_test = dataset_size_test, n_features=dimensions ** 2, contamination=0.001, random_state = 42)
    # _, test_data, _, test_labels = generate_data_clusters(n_train=dataset_size, n_test = dataset_size_test, n_features=dimensions ** 2, contamination=0.45, random_state = 44, density='different')

    dimensions = configs_dict['dimensions'][0]
    dataset_size = configs_dict['dataset_size'][0]
    dataset_size_test = configs_dict['dataset_size_test'][0]
    no_classes = configs_dict['no_classes'][0]
    no_classes_test = configs_dict['no_classes_test'][0]
    use_train_anomalies = configs_dict['use_train_anomalies'][0]
    train_anomalies = configs_dict['train_anomalies'][0]
    test_anomalies = configs_dict['test_anomalies'][0]

    roc = []
    clf_list = ['CD','LODA', 'ECOD', 'INNE', 'HBOS', 'DeepSVDD', 'OCSVM']
    #clf_list = ['OCSVM']
    for clf in clf_list:
        roc.append(evaluate(clf))
        # # Convert the reshaped array to a PyTorch tensor
        # x_test.append(x)
        # y_test.append(y)

    print("Train anomalies:", train_anomalies)
    print("Test anomalies:", test_anomalies)
    print("Use anomalies:", use_train_anomalies)

    for i in range(0, len(roc)):
        print(clf_list[i], ":", np.mean(roc[i]), "std:", np.std(roc[i]))

    # x_test = torch.stack(x_test).squeeze()
    # y_test = np.array([np.array(tensor) for tensor in y_test]).flatten()
    # print(y_test.shape)
