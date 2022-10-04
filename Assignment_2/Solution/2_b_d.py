from libsvm.svmutil import *
import time
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
def compute_validation_accuracy(problem):
    k_fold_acc = []
    C = [0.00001, 0.001, 1, 5, 10]
    for c in C:
        # print(f"Checking for {c}")
        options = " ".join(["-q -v 5", f"-c {c}", "-g 0.05", "-t 2"])
        model = svm_train(problem, options)
        k_fold_acc.append(model)
    return k_fold_acc

def compute_test_accuracy(problem, XX, YY, is_normalised=False):
    if(is_normalised == False):
        XX = XX/255
    test_acc = []
    C = [0.00001, 0.001, 1, 5, 10]
    for c in C:
        options = " ".join([f"-q -c {c}", "-g 0.05", "-t 1"])
        model = svm_train(problem, options)
        sp_label, p_acc, p_val = svm_predict(YY, (XX), model)
        test_acc.append(p_acc[0])
    return test_acc

if __name__ == "__main__":
    start_time = time.time()

    import sys
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]

    data_df = pd.read_csv(train_data_path, header=None)
    data_df.columns = [f"col_{i}" for i in data_df.columns]
    data_df.rename(columns={"col_784": "label"}, inplace=True)

    data_subset = data_df[data_df["label"].isin([8, 9])]
    data_subset.sort_values("label", inplace=True)
    data_subset["y_label"] = 0
    data_subset.loc[data_subset["label"] == 8, "y_label"] = 1
    data_subset.loc[data_subset["label"] == 9, "y_label"] = -1

    columns = [f"col_{i}" for i in range(28 * 28)]
    x_features = data_subset[columns]
    y_labels = data_subset["y_label"]

    problem = svm_problem(y_labels.tolist(), (x_features.values / 255).tolist())
    k_fold_acc = compute_validation_accuracy(problem)

    test_data_df = pd.read_csv(test_data_path, header=None)
    test_data_df.columns = [f"col_{i}" for i in test_data_df.columns]
    test_data_df.rename(columns={"col_784": "label"}, inplace=True)

    # entry number : csy217548 so : class 8 and class 9
    test_data_subset = test_data_df[test_data_df["label"].isin([8, 9])]
    test_data_subset.sort_values("label", inplace=True)
    test_data_subset["y_label"] = 0
    test_data_subset.loc[test_data_subset["label"] == 8, "y_label"] = 1
    test_data_subset.loc[test_data_subset["label"] == 9, "y_label"] = -1

    columns = [f"col_{i}" for i in range(28 * 28)]
    x_test_features = test_data_subset[columns]
    y_test_labels = test_data_subset["y_label"]

    XX = np.array(x_test_features)
    YY = np.array(y_test_labels)

    test_acc = compute_test_accuracy(problem,XX,YY)
    print(f"Test Accuracy : {test_acc}")

    print(f"Execution time {time.time()-start_time}")

# /home/prayushi/Desktop/IITD/Assignments/ML/Assignment_2/Data/mnist/train.csv /home/prayushi/Desktop/IITD/Assignments/ML/Assignment_2/Data/mnist/test.csv
