
from tqdm import tqdm
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore")
solvers.options['show_progress'] = False

def gaussian_kernel(X,x,gamma=0.05):
    return np.exp(-gamma*(np.linalg.norm(X-x, axis=1)**2))

def predict_gaussian(X, is_normalised=False):
    
    if(not is_normalised):
        X = X/255
        
    y_pred = []
    mul_cache = filtered_alpha_gaussian*y_t_gaussian.reshape(-1,1)
    t=0
    for i in tqdm(range(len(X))):
        kernel = gaussian_kernel(x_t_gaussian,X[i]).reshape(-1,1)
        t = np.sum(mul_cache*kernel)
        y_pred.append(np.sign(t + b_guassian))
    
    return y_pred

def accuracy(y_actual, y_pred):
    correct = 0
    for i in range(len(y_actual)):
        if(y_actual[i]==y_pred[i]):
            correct += 1
    return round(correct*100/len(y_actual),3)

class GuassianKernel:
    def __init__(self, X, Y):
        self.X = np.array(X)/255
        self.Y = np.array(Y)
        self.gamma = 0.05
        self.C = 1
        self.m, self.n = self.X.shape
        self.H = np.zeros((self.m, self.m))
        self.W = None
        self.b = None
        
    def compute_solver_input(self):

        self.H = np.zeros((self.m, self.m))
#         for i in tqdm(range(self.m)):
#             for j in range(self.m):
#                 self.H[i][j] = np.exp(-self.gamma*(np.linalg.norm(self.X[i] - self.X[j]) ** 2) )
        
#         for i in tqdm(range(self.m)):
#             self.H[i][:] = np.exp(-self.gamma*(np.linalg.norm(self.X[i]-self.X, axis=1)**2))

        X_squared = np.reshape(np.sum((self.X*self.X),axis = 1),(self.m,1))
        X_norm = X_squared + X_squared.T - (2 * np.dot(self.X ,self.X.T))
        self.H = np.exp(-0.05 * X_norm)
        
        # compute P
        P = matrix(np.outer(self.Y,self.Y)*self.H)
        
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
        
        return alphas, self.H



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

    gk = GuassianKernel(np.array(x_features),y_labels.tolist())
    alphas_gaussian,kernel_matrix = gk.fit_model()

    indices = ((alphas_gaussian > 0.01) & (alphas_gaussian < 1.0)).flatten()
    ind = np.arange(len(alphas_gaussian))[indices]
    X = np.array(x_features)
    Y = np.array(y_labels.tolist())
    X = X/255
    x_t_gaussian = X[indices]
    y_t_gaussian = Y[indices]
    filtered_alpha_gaussian = alphas_gaussian[indices]
    print(f"#support vectors: {filtered_alpha_gaussian.shape}")

    b_guassian = 0
    mul_cache = filtered_alpha_gaussian * y_t_gaussian.reshape(-1,1)
    for i in tqdm(range(len(filtered_alpha_gaussian))):
        b_guassian += y_t_gaussian[i] - np.sum( mul_cache * kernel_matrix[ind[i],indices].reshape(-1,1))

    b_guassian = b_guassian/len(filtered_alpha_gaussian)
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

    y_pred = predict_gaussian(XX)
    test_acc = accuracy(y_pred=y_pred,y_actual=YY.tolist())

    print(f"Test accuracy : {test_acc}")


# /home/prayushi/Desktop/IITD/Assignments/ML/Assignment_2/Data/mnist/train.csv /home/prayushi/Desktop/IITD/Assignments/ML/Assignment_2/Data/mnist/test.csv

