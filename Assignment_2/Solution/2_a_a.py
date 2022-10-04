import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore")
solvers.options['show_progress'] = False

def predict_linear(x_features, W, b_linear):
    y_pred = []
    for feature in x_features:
        if(feature.dot(W) + b_linear > 0):
            y_pred.append(1)
        else:
            y_pred.append(-1)
    return y_pred

def accuracy(y_actual, y_pred):
    correct = 0
    for i in range(len(y_actual)):
        if(y_actual[i]==y_pred[i]):
            correct += 1
    return round(correct*100/len(y_actual),3)

class LinearKernel:
    def __init__(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.C = 1
        self.m, self.n = self.X.shape
        self.W = None
        self.b = None
        
    def compute_solver_input(self):
        # compute P
        X_prime = (self.X.T*self.Y).T
        P = matrix(np.dot(X_prime, X_prime.T)*1.) #m*m

        #compute G
        const1 = np.eye(self.m)*-1
        const2 = np.eye(self.m)
        G = matrix(np.vstack((const1, const2))) #2m*2m

        # compute h
        h = matrix(np.array([0.]*self.m + [self.C*1.]*self.m)) #2m*1

        #compute q
        q = matrix(np.ones((self.m, 1))*-1.) # m*1

        #compute A and b
        A = matrix(self.Y.reshape(1, -1)*1.) # m*1
        b = matrix(np.zeros(1)) # scalar

        return P,G,h,q,A,b
        
    def fit_model(self):
        P,G,h,q,A,b = self.compute_solver_input()
        solver_solution = solvers.qp(P, q, G, h, A, b)

        alphas = np.array(solver_solution['x'])
        return alphas


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

    lk = LinearKernel(X,Y)
    alphas_linear = lk.fit_model()

    indices = ((alphas_linear > 0.00000001) & (alphas_linear < 1.0)).flatten()
    x_t_linear = X[indices]
    y_t_linear = Y[indices]
    filtered_alpha_linear = alphas_linear[indices]

    print(f"#support vectors: {filtered_alpha_linear.shape}")
    W = ((filtered_alpha_linear*y_t_linear.reshape(-1,1)).T@x_t_linear).reshape(-1,1)

    B = np.dot(x_t_linear,W)
    b_linear = -np.mean((np.amin(B[(y_t_linear==1).flatten()]) + np.amax(B[(y_t_linear == -1).flatten()])))

    print(f"Total training time : {time.time()-start_time} seconds")

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

    y_pred = predict_linear(XX, W, b_linear)
    test_acc = accuracy(y_pred=y_pred,y_actual=YY.tolist())

    print(f"Test accuracy : {test_acc}")


# /home/prayushi/Desktop/IITD/Assignments/ML/Assignment_2/Data/mnist/train.csv /home/prayushi/Desktop/IITD/Assignments/ML/Assignment_2/Data/mnist/test.csv

