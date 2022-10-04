# Imports
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
import matplotlib

# A - generation of data
def generate_data_distrib():

	data_distrib = [(3,2), (-1,2)] 
	x1 = np.random.normal(data_distrib[0][0], data_distrib[0][1], 1000000)
	x2 = np.random.normal(data_distrib[1][0], data_distrib[1][1], 1000000)

	noise = np.random.normal(0, 2**(1/2), 1000000).reshape(-1,1)
	
	theta = np.array([3,1,2]).reshape(-1,1)
	
	arr_1 = np.c_[np.ones(len(x1)),x1]
	X = np.c_[arr_1, x2]

	Y = X.dot(theta)+noise
	return X,Y

# C
def test_hypothesis(test_x_ip):
	
	test_x = pd.read_csv(test_x_ip)
	test_x_arr = np.c_[np.c_[np.ones(len(test_x)),test_x["X_1"]], test_x["X_2"]]
	Y_actual =  np.array(test_x["Y"]).reshape(-1,1)

	# Filled in the learned parameters
	theta = {"n_1000000": [2.95558351,1.01031417,1.99778227],"n_10000": [3.00242126,0.99995314, 2.0012511], "n_100" : [3.00523206, 0.99717407, 2.00254335], "n_1" : [2.9673345, 1.01952358, 2.00101839], "default" : [3,1,2]}

	print("----------------------------")
	print("Error in learning hypothesis\n")
	for i in theta:
	    y_hat = test_x_arr.dot(np.array(theta[i]).reshape(-1,1))
	    error = np.mean(np.square(y_hat-Y_actual))
	    print(f"Error for {i} is {error}")

class GradientDescent:
    def __init__(self, X, Y, learning_rate=0.001, iterations=100, auto_convergence_detection=False, stopping_threshold=0.001, k=100):
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.auto_convergence_detection = auto_convergence_detection
        self.stopping_threshold = stopping_threshold
        self.k = k
        
    def least_square_loss(self, theta, X, Y):
        Y_hat = X.dot(theta)
        least_square_loss = (1/(2*len(X))) * np.sum(np.square(Y_hat-Y))
        return least_square_loss
    
    def is_converged_batch_GD(self, loss_history):
        k = self.k
        delta = self.stopping_threshold   #k=100,r=1,delta=0.00001
        N = len(loss_history)
        if(N < k): return False;
        
        sum_last_k_batches_curr_iteration = np.mean(loss_history[max(0,N-k-1):N-1])
        sum_last_k_batches_prev_iteration = loss_history[-1]
        
#         print((str)(sum_last_k_batches_prev_iteration - sum_last_k_batches_curr_iteration))
        if(abs(sum_last_k_batches_prev_iteration - sum_last_k_batches_curr_iteration) <= delta): 
#             print("converged")
            return True
        return False

    def is_converged_batch_GD_2(self, loss_history):
        k = self.k
        delta = self.stopping_threshold #k=100,r=1,delta=0.00001
        N = len(loss_history)
        if(N < 2*k+1): return False;
        
        sum_last_k_batches_curr_iteration = np.mean(loss_history[N-k:N])
        sum_last_k_batches_prev_iteration = np.mean(loss_history[N-2*k:N-k])
        
        if(abs(sum_last_k_batches_prev_iteration - sum_last_k_batches_curr_iteration) <= delta): 
            return True
        return False
    
        
    def mini_batch_GD(self, theta, batch_size=64):
        
        random_shuffle = np.random.permutation(len(self.X))
        self.X = self.X[random_shuffle]
        self.Y = self.Y[random_shuffle]
        
        # stores iteration wise loss - to track convergence
        loss_history = list()
        # stores iteration wise theta - only for visualisation
        theta_history = list()
        
        data_len = len(self.X)
        itr = 0
        
        ls_index=0
        while(True):
            if(self.auto_convergence_detection==False and itr >= self.iterations): break
            else:
                i = 0
                iteration_wise_batch_wise_loss = 0
                batch_number=0
                
                while(i<data_len):
                    loss_history.append([])
                
                    # create data batch
                    X_batch = self.X[i:i+batch_size]
                    Y_batch = self.Y[i:i+batch_size]

                    # compute preicted value of y
                    Y_hat = X_batch.dot(theta)

                    # update theta parameters
                    delta = (1/batch_size)*( X_batch.T.dot((Y_hat - Y_batch)))*self.learning_rate
#                     print(f"Updating theta with gradient {delta}")
                    theta = theta - delta
                    # compute loss
                    iteration_wise_batch_wise_loss = self.least_square_loss(theta, X_batch, Y_batch)
                    loss_history[ls_index] = iteration_wise_batch_wise_loss
                    ls_index += 1
                    
                    i = i + batch_size
                    batch_number += 1
                
                    # self.auto_convergence_detection==True and 
                    if(self.auto_convergence_detection==True and self.is_converged_batch_GD_2(loss_history)):
                        theta_history.append([])
                        theta_history[itr] = theta
                        return theta_history, loss_history
                
                theta_history.append([])
                theta_history[itr] = theta
                
            itr += 1
        return theta_history, loss_history

if __name__=="__main__":

	test_x_ip = "/home/prayushi/Desktop/IITD/Asisgnments/ML/Assignment_1/Data/q2/q2test.csv"

	X,Y = generate_data_distrib()

	theta = np.zeros(3).reshape(-1,1)
	batch_size = 100
	# the value of k and stopping threshold varies based on different batch size - TURN auto_convergence_detection = True
	t, l = GradientDescent(X, Y, auto_convergence_detection=False, iterations=20, k=10, stopping_threshold=0.001).mini_batch_GD(theta, batch_size)

	print("Learned params")
	print(f'Batch_size,Theta0,Theta1,Theta2,Final cost/MSE,Total theta updations')
	print(f'{batch_size},{round(t[-1][0][0],4)},{round(t[-1][1][0],4)},{round(t[-1][2][0],4)},{round(l[-1],4)},{len(l)}')

	test_hypothesis(test_x_ip)