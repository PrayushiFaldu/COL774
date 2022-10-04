import pandas as pd
from libsvm.svmutil import *
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore")

class Classifiers:
    def __init__(self):
        self.model = None
        self.class1 = None
        self.class2 = None

def train_classifiers(data_df):
    for key in classifier_dict:
        # print(f"Traning for classifier {key}")
        data_subset = data_df[data_df["label"].isin([key[0], key[1]])]

        data_subset.sort_values("label", inplace=True)
        data_subset["y_label"] = data_subset["label"].apply(lambda x: 1 if (x == key[0]) else -1)

        columns = [f"col_{i}" for i in range(28 * 28)]
        x_features = data_subset[columns]
        y_labels = data_subset["y_label"]

        problem = svm_problem(y_labels.tolist(), (x_features.values / 255).tolist())

        options = " ".join([f"-q -c 1", "-g 0.05", "-t 2"])
        model = svm_train(problem, options)

        exec(f"class_{key[0]}_{key[1]} = Classifiers()")

        classifier_dict[key].model = model
        classifier_dict[key].class1 = key[0]
        classifier_dict[key].class2 = key[1]

def predict_libsvm(X,Y, model, is_normalised=False):
    if (not is_normalised):
        X = X / 255

    sp_label, p_acc, p_val = svm_predict(Y, X, model)
    test_acc = p_acc[0]

    return sp_label, p_val, test_acc

def accuracy(y_actual, y_pred):
    correct = 0
    for i in range(len(y_actual)):
        if(y_pred[i] == y_actual[i]):
            correct += 1
    return round(correct*100/len(y_actual),3)

if __name__ == "__main__":
    start_time = time.time()

    import sys

    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]

    classifier_dict = {}
    for i in range(0, 10):
        for j in range(i + 1, 10):
            classifier_dict.update({(i, j): Classifiers()})

    data_df = pd.read_csv(train_data_path, header=None)
    data_df.columns = [f"col_{i}" for i in data_df.columns]
    data_df.rename(columns={"col_784": "label"}, inplace=True)

    train_classifiers(data_df)

    test_data_df = pd.read_csv(test_data_path, header=None)
    test_data_df.columns = [f"col_{i}" for i in test_data_df.columns]
    test_data_df.rename(columns={"col_784": "label"}, inplace=True)

    columns = [f"col_{i}" for i in range(28 * 28)]
    x_test_features = test_data_df[columns]
    y_test_labels = test_data_df["label"]

    XX = np.array(x_test_features)
    YY = np.array(y_test_labels)

    classifier_prediction = []
    classifier_score = []
    classifier_acc = []
    classifier_map = {}
    count = 0
    for classifier in classifier_dict:
        classifier_map.update({count: classifier})
        model = classifier_dict[classifier].model
        c1 = classifier_dict[classifier].class1
        c2 = classifier_dict[classifier].class2
        y_pred, scores, acc = predict_libsvm(XX,YY,model)
        y_pred_actual = [c1 if i == 1 else c2 for i in y_pred]
        classifier_prediction.append(y_pred_actual)
        classifier_score.append(scores)
        classifier_acc.append(acc)

    predictions = []
    cp = np.array(classifier_prediction).T
    for i in range(len(XX)):
        predictions.append(np.argmax(np.bincount(cp[i])))

    test_acc = accuracy(y_pred=predictions, y_actual=YY.tolist())
    print(f"Test Accuracy : {test_acc}, and {sum(classifier_acc)/len(classifier_acc)}")

    conf_mat = np.zeros((10, 10))
    y_actual = YY.tolist()
    for i, p in enumerate(predictions):
        conf_mat[y_actual[i]][p] += 1

    print(conf_mat)

    print(f"Execution time {time.time()-start_time}")

# /home/prayushi/Desktop/IITD/Assignments/ML/Assignment_2/Data/mnist/train.csv /home/prayushi/Desktop/IITD/Assignments/ML/Assignment_2/Data/mnist/test.csv
