
from tqdm import tqdm
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
import numpy as np
from libsvm.svmutil import *
import time

import warnings
warnings.filterwarnings("ignore")
solvers.options['show_progress'] = False

def gaussian_kernel(X,x,gamma=0.05):
    return np.exp(-gamma*(np.linalg.norm(X-x, axis=1)**2))

def predict_gaussian(X,filtered_alpha_gaussian, x_t_gaussian, y_t_gaussian, is_normalised=False):
    
    if is_normalised:
        pass
    else:
        X = X/255

    y_pred = []
    mul_cache = filtered_alpha_gaussian*y_t_gaussian.reshape(-1,1)
    t=0
    for i in range(len(X)):
        kernel = gaussian_kernel(x_t_gaussian,X[i]).reshape(-1,1)
        t = np.sum(mul_cache*kernel)
        y_pred.append(np.sign(t + b_guassian))
    
    return y_pred

def accuracy(y_actual, y_pred):
    correct = 0
    for i in range(len(y_actual)):
        if(y_pred[i] == y_actual[i]):
            correct += 1
    return round(correct*100/len(y_actual),3)

def libsvm(x_features, y_labels, kernel=LINEAR, C=1.0, gamma=0.05):
    model_fit = svm_problem(list(y_labels), list(x_features))
    parameters = svm_parameter()
    parameters.kernel_type = kernel
    parameters.C = 1
    parameters.gamma = 0.05

    svm_trained_model=svm_train(model_fit, parameters)
    return svm_trained_model

if __name__=="__main__":


    start_time = time.time()

    import sys
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]

    data_df = pd.read_csv(train_data_path, header=None)
    data_df.columns = [f"col_{i}" for i in data_df.columns]
    data_df.rename(columns={"col_784":"label"}, inplace=True)

    # entry number : csy217548 so : class 8 and class 9
    data_subset = data_df[data_df["label"].isin([8,9])]
    data_subset.sort_values("label", inplace=True)
    data_subset["y_label"] = 0
    data_subset.loc[data_subset["label"] ==8, "y_label"] = 1
    data_subset.loc[data_subset["label"] ==9, "y_label"] = -1

    columns = [f"col_{i}" for i in range(28*28)]
    x_features = data_subset[columns]
    y_labels = data_subset["y_label"]

    X = np.array(x_features)
    Y = np.array(y_labels)

    gk_libsvm = libsvm(X/255,Y, kernel=RBF)
    lk_libsvm = libsvm(X,Y)

    test_data_df = pd.read_csv(test_data_path, header=None)
    test_data_df.columns = [f"col_{i}" for i in test_data_df.columns]
    test_data_df.rename(columns={"col_784":"label"}, inplace=True)

    # entry number : csy217548 so : class 8 and class 9
    test_data_subset = test_data_df[test_data_df["label"].isin([8,9])]
    test_data_subset.sort_values("label", inplace=True)
    test_data_subset["y_label"] = 0
    test_data_subset.loc[test_data_subset["label"] ==8, "y_label"] = 1
    test_data_subset.loc[test_data_subset["label"] ==9, "y_label"] = -1

    columns = [f"col_{i}" for i in range(28*28)]
    x_test_features = test_data_subset[columns]
    y_test_labels = test_data_subset["y_label"]

    XX = np.array(x_test_features)
    YY = np.array(y_test_labels)

    gk_sp_label, gk_p_acc, gk_p_val = svm_predict(YY,XX/255,gk_libsvm)
    lk_sp_label, lk_p_acc, lk_p_val = svm_predict(YY,XX,lk_libsvm)


    print(f"Test accuracy  Gaussian Kernel: {gk_p_acc[0]} and Linear Kernel : {lk_p_acc[0]}")


# /home/prayushi/Desktop/IITD/Assignments/ML/Assignment_2/Data/mnist/train.csv /home/prayushi/Desktop/IITD/Assignments/ML/Assignment_2/Data/mnist/test.csv

