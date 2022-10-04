import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore")

solvers.options["show_progress"] = False
class GuassianKernel:
    def __init__(self, X, Y):
        self.X = np.array(X) / 255
        self.Y = np.array(Y)
        self.gamma = 0.05
        self.C = 1
        self.m, self.n = self.X.shape
        self.H = np.zeros((self.m, self.m))
        self.W = None
        self.b = None

    def compute_solver_input(self):
        self.H = np.zeros((self.m, self.m))

        X_squared = np.reshape(np.sum((self.X * self.X), axis=1), (self.m, 1))
        X_norm = X_squared + X_squared.T - (2 * np.dot(self.X, self.X.T))
        self.H = np.exp(-0.05 * X_norm)

        # compute P
        P = matrix(np.outer(self.Y, self.Y) * self.H)

        # compute G
        const1 = np.eye(self.m) * -1
        const2 = np.eye(self.m)
        G = matrix(np.vstack((const1, const2)))  # 2m*2m

        # compute h
        h = matrix(np.array([0.] * self.m + [self.C * 1.] * self.m))  # 2m*1

        # compute q
        q = matrix(np.ones((self.m, 1)) * -1.)  # m*1

        # compute A and b
        A = matrix(self.Y.reshape(1, -1) * 1.)  # m*1
        b = matrix(np.zeros(1))  # scalar

        return P, G, h, q, A, b

    def fit_model(self):
        P, G, h, q, A, b = self.compute_solver_input()
        solver_solution = solvers.qp(P, q, G, h, A, b)

        alphas = np.array(solver_solution['x'])

        return alphas, self.H

class Classifiers:
    def __init__(self):
        self.filtered_alpha_gaussian = None
        self.x_t_gaussian = None
        self.y_t_gaussian = None
        self.b_gaussian = None
        self.alphas_gaussian = None
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

        gk = GuassianKernel(np.array(x_features), y_labels.tolist())
        alphas_gaussian, kernel_matrix = gk.fit_model()

        indices = ((alphas_gaussian > 0.001) & (alphas_gaussian < 1.0)).flatten()
        ind = np.arange(len(alphas_gaussian))[indices]
        X = np.array(x_features)
        Y = np.array(y_labels.tolist())
        X = X / 255
        x_t_gaussian = X[indices]
        y_t_gaussian = Y[indices]
        filtered_alpha_gaussian = alphas_gaussian[indices]
        print(f"#support vectors: {filtered_alpha_gaussian.shape}")
        # print(filtered_alpha_gaussian.shape, x_t_gaussian.shape, y_t_gaussian.shape)

        b_gaussian = 0
        mul_cache = filtered_alpha_gaussian * y_t_gaussian.reshape(-1, 1)
        for i in range(len(filtered_alpha_gaussian)):
            b_gaussian += y_t_gaussian[i] - np.sum(mul_cache * kernel_matrix[ind[i], indices].reshape(-1, 1))

        b_gaussian = b_gaussian / len(filtered_alpha_gaussian)
        exec(f"class_{key[0]}_{key[1]} = Classifiers()")

        classifier_dict[key].x_t_gaussian = x_t_gaussian
        classifier_dict[key].y_t_gaussian = y_t_gaussian
        classifier_dict[key].b_gaussian = b_gaussian
        classifier_dict[key].filtered_alpha_gaussian = filtered_alpha_gaussian
        classifier_dict[key].alphas_gaussian = alphas_gaussian
        classifier_dict[key].class1 = key[0]
        classifier_dict[key].class2 = key[1]


def guassian_kernel_2(X, x, gamma=0.05):
    return np.exp(-gamma * (np.linalg.norm(X - x, axis=1) ** 2))


def predict_gaussian(X, filtered_alpha_gaussian, x_t_gaussian, y_t_gaussian, b_gaussian, is_normalised=False):
    if (not is_normalised):
        X = X / 255

    m = x_t_gaussian.shape[0]
    n = X.shape[0]
    x_training_squared = np.sum((x_t_gaussian * x_t_gaussian), axis=1).reshape(m, 1)
    x_test_squared = np.sum((X * X), axis=1).reshape(n, 1)
    H1 = np.repeat(x_training_squared, n, axis=1)
    H2 = np.repeat(x_test_squared.T, m, axis=0)
    X_norm = H1 + H2 - 2 * np.dot(x_t_gaussian, X.T)
    gaus = np.exp(-0.05 * X_norm)
    B = filtered_alpha_gaussian * y_t_gaussian.reshape(-1, 1) * gaus
    scores = np.sum(B, axis=0) + b_gaussian
    y_pred = np.sign(scores)
    return y_pred, scores

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
    classifier_map = {}
    count = 0
    for classifier in classifier_dict:
        classifier_map.update({count: classifier})
        alpha = classifier_dict[classifier].filtered_alpha_gaussian
        y_t = classifier_dict[classifier].y_t_gaussian
        x_t = classifier_dict[classifier].x_t_gaussian
        b = classifier_dict[classifier].b_gaussian
        c1 = classifier_dict[classifier].class1
        c2 = classifier_dict[classifier].class2
        y_pred, scores = predict_gaussian(XX, alpha, x_t, y_t, b)
        y_pred_actual = [c1 if i == 1 else c2 for i in y_pred]
        classifier_prediction.append(y_pred_actual)
        classifier_score.append(scores)

    predictions = []
    cp = np.array(classifier_prediction).T
    for i in range(len(XX)):
        predictions.append(np.argmax(np.bincount(cp[i])))

    test_acc = accuracy(y_pred=predictions, y_actual=YY.tolist())
    print(f"Test Accuracy : {test_acc}")

    conf_mat = np.zeros((10, 10))
    y_actual = YY.tolist()
    for i, p in enumerate(predictions):
        conf_mat[y_actual[i]][p] += 1

    print(conf_mat)

    print(f"Execution time {time.time()-start_time}")

# /home/prayushi/Desktop/IITD/Assignments/ML/Assignment_2/Data/mnist/train.csv /home/prayushi/Desktop/IITD/Assignments/ML/Assignment_2/Data/mnist/test.csv
