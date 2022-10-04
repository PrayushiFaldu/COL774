# Imports

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# Preprocess input 
class Preprocess:
    def __init__(self, ip):
        self.input_features = ip
        
    def normalize(self):
        mean = np.mean(self.input_features, axis=0)
        std = np.std(self.input_features, axis=0)
        standardize_features = np.divide(np.subtract(self.input_features, mean),std)
        return standardize_features


# Computes sigmoid
def compute_sigmoid(theta,X):
    z = X.dot(theta)
    h = 1.0 / (1.0 + np.exp(-z)) 
    return h.reshape(-1,1)

# Computes gradient
def compute_gradient(theta,X,Y):
    g_z = compute_sigmoid(theta,X)
    del_l = X.T.dot(Y - g_z)
    return del_l

# Computes Hessian
def compute_hessian(theta, X, Y):
    g_z = compute_sigmoid(theta,X)
    base = g_z - np.square(g_z)
    x_1 =X.T[1].reshape(len(X),1)
    x_2 =X.T[2].reshape(len(X),1)
    a11 = np.sum(base)
    a12 = np.sum(base*x_1)
    a13 = np.sum(base*x_2)
    a22 = np.sum(base*x_1*x_1)
    a23 = np.sum(base*x_1*x_2)
    a33 = np.sum(base*x_2*x_2)
    return np.array([a11, a12, a13, a12, a22, a23, a13, a23, a33]).reshape(3,3)

# Compute log likelihood (LL(theta))
def compute_log_likelihood(theta, X, Y):                                                                
    g_z = compute_sigmoid(theta,X)  
    return np.sum(Y * np.log(g_z)+ (1 - Y) * np.log(1 - g_z)).to_list()[0]

# Draw plot
def draw_plot(theta, ax_obj):
    t0 = theta.reshape(1,3)[0][0]
    t1 = theta.reshape(1,3)[0][1]
    t2 = theta.reshape(1,3)[0][2]

    x1 = np.linspace(-2,2,100)
    x2 = -1 * (t0 + t1*x1)/t2
    ax_obj.plot(x1,x2,"g-")
    
# Main function - fit model using newtons optimisation
def newtons_optimisation(theta, X, Y, iterations=100, auto_convergence=False, plot=None):

    delta = 0.001
    loss_history = []
    i = 0
    while(True):
        i+= 1;
        if(auto_convergence == False and i>iterations): 
            return theta, loss_history
        elif((len(loss_history) > 2) and (loss_history[-1] ==  float('-inf') or abs(loss_history[-1]-loss_history[-2])<= delta)): 
             return theta, loss_history
        else:
            del_l = compute_gradient(theta, X, Y)
            H = compute_hessian(theta, X, Y)
            theta = theta - np.linalg.inv(H).dot(del_l) 
            loss_history.append(compute_log_likelihood(theta, X, Y))
             
    return theta, loss_history
    
if __name__ == "__main__":

    # Specify input files path here
    input_x = "/home/prayushi/Desktop/IITD/Asisgnments/ML/Assignment_1/Data/q3/logisticX.csv"
    input_y = "/home/prayushi/Desktop/IITD/Asisgnments/ML/Assignment_1/Data/q3/logisticY.csv"
    output_plots_dir_path ="/home/prayushi/Desktop/IITD/Asisgnments/ML/Solution/Submission/Q3"
    
    # Read files
    X = pd.read_csv(input_x, header=None, names=["x1","x2"])
    Y = pd.read_csv(input_y, header=None, names=["y"])

    # Preprocess data
    x1_normalized = Preprocess(X["x1"]).normalize()
    x2_normalized = Preprocess(X["x2"]).normalize()

    # Concatenate column wise
    X_normalized = np.c_[x1_normalized,x2_normalized]
    X_normalized = np.c_[np.ones(len(x1_normalized)),X_normalized]

    # intialise theta
    theta = np.array([0.001,0.001,0.001]).reshape(-1,1)

    # Fit Model
    theta_learned, loss = newtons_optimisation(theta, X_normalized, Y, auto_convergence=False, iterations=5, plot=plt)

    # 1(b)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2", rotation=90)
    ax.plot(x1_normalized[:50],x2_normalized[:50],'b+', label='0')
    ax.plot(x1_normalized[50:],x2_normalized[50:],'r*', label='1')
    ax.legend(loc ="upper right")
    
    draw_plot(theta_learned,ax_obj=ax)
    
    fig.savefig(f'{output_plots_dir_path}/3b_plot.png')

