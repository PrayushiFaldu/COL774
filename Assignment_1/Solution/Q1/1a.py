# Imports
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Preprocess input 
class Preprocess:
    def __init__(self, ip):
        self.input_features = ip
        
    def normalize(self):
        mean = np.mean(self.input_features, axis=0)
        std = np.std(self.input_features, axis=0)
        standardize_features = np.divide(np.subtract(self.input_features, mean),std)
        return standardize_features

class GradientDescent:
    def __init__(self, X, Y, learning_rate=0.01, iterations=100, auto_convergence_detection=False, stopping_threshold=0.001):
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.auto_convergence_detection = auto_convergence_detection
        self.stopping_threshold = stopping_threshold
        
    def least_square_loss(self, theta, X, Y):
        Y_hat = X.dot(theta)
        least_square_loss = (1/2*(len(X))) * np.sum(np.square(Y_hat-Y))
        return least_square_loss
        
    def vanilla_GD(self, theta):
        random_shuffle = np.random.permutation(len(self.X))
        self.X = self.X[random_shuffle]
        self.Y = self.Y[random_shuffle]
        
        # stores iteration wise loss - to track convergence
        loss_history = list()
        # stores iteration wise theta - only for visualisation
        theta_history = list()
        
        itr = 0
        
        while(True):
            if(self.auto_convergence_detection==False and itr >= self.iterations): break
            else:
                i = 0
                iteration_wise_batch_wise_loss = 0
                batch_number=0
                
                Y_hat = self.X.dot(theta)
                theta = theta - (1/len(self.X))*( self.X.T.dot((Y_hat - self.Y)))*self.learning_rate
                iteration_wise_loss = self.least_square_loss(theta, self.X, self.Y)
                loss_history.append([])
                loss_history[itr] = iteration_wise_loss
                theta_history.append([])
                theta_history[itr] = theta
                if(self.auto_convergence_detection==True and self.is_converged(loss_history)):
                        return theta_history, loss_history
                
            itr += 1
        return theta_history, loss_history
    
    
    def stochastic_GD(self):
        pass
    
    def is_converged(self, loss_history):
        k = 1
        delta = self.stopping_threshold
        N = len(loss_history)
        if(N < k+1): return False;
        
        sum_last_k_batches_curr_iteration = loss_history[-1]
        sum_last_k_batches_prev_iteration = np.mean(loss_history[max(0,N-k-1):N-1])
        
        if(abs(sum_last_k_batches_prev_iteration - sum_last_k_batches_curr_iteration) <= delta): 
            return True
        return False
    
def least_square_loss(theta, X, Y):
    Y_hat = X.dot(theta)
    least_square_loss = (1/2*(len(X))) * np.sum(np.square(Y_hat-Y))
    return least_square_loss

def draw_3d_mesh_and_contour(loss_history, theta_history, output_plots_dir_path):
	Lx = -0.1
	Ly = 1.5
	Nx = 2.01
	Ny = -1.5

	x = np.linspace(Nx, Lx, 50)
	y = np.linspace(Ny, Ly, 50)

	T0, T1 = np.meshgrid(x, y)
	cost_mesh = np.array([least_square_loss(np.array([t0,t1]).reshape(-1,1), X_intercept, Y) for t0, t1 in zip(np.ravel(T0), np.ravel(T1)) ] )
	Z = cost_mesh.reshape(T0.shape)

	J_history = [np.array(i) for i in loss_history]
	theta_0 = list()
	theta_1 = list()

	for i in range(len(theta_history)):
	    theta_0.append(theta_history[i][0][0])
	    theta_1.append(theta_history[i][1][0])

	theta_result = np.array([theta_0[-1], theta_1[-1]])
	anglesx = np.array(theta_0)[1:] - np.array(theta_0)[:-1]
	anglesy = np.array(theta_1)[1:] - np.array(theta_1)[:-1]

	fig = plt.figure(figsize = (10,6))

	# Surface plot
	ax = fig.add_subplot(1, 1, 1, projection='3d')
	ax.plot_surface(T0, T1, Z, rstride = 5, cstride = 5, cmap = 'jet', alpha=0.5)
	ax.set_xlabel('theta 0')
	ax.set_ylabel('theta 1')
	ax.set_zlabel('Cost function')
	ax.set_title('Gradient descent: Root at {}'.format(theta_result.ravel()))
	ax.view_init(45, 30)

	for i in range(len(J_history)):
	    ax.plot(theta_0,theta_1,J_history, marker = '.', color = 'r', alpha = .4)
	    # fig.tight_layout()
	    # fig.canvas.draw()
	    # plt.pause(0.2)

	fig.savefig(f'{output_plots_dir_path}/1c_plot.png')

	fig = plt.figure(figsize = (10,6))

	# #Contour plot
	ax_1 = fig.add_subplot(1, 1, 1)
	ax_1.contour(T0, T1, Z, 150, cmap = 'jet', zorder=-10)
	ax_1.set_xlabel('theta 0')
	ax_1.set_ylabel('theta 1')
	for i in range(len(J_history)):
	    ax_1.plot(theta_0,theta_1, marker = '.', color = 'r', alpha = .4)
	    # fig.tight_layout()
	    # fig.canvas.draw()
	    # plt.pause(0.2)

	fig.savefig(f'{output_plots_dir_path}/1d_plot.png')


if __name__=="__main__":

	input_x = "/home/prayushi/Desktop/IITD/Asisgnments/ML/Assignment_1/Data/q1/linearX.csv"
	input_y = "/home/prayushi/Desktop/IITD/Asisgnments/ML/Assignment_1/Data/q1/linearY.csv"
	output_plots_dir_path = "/home/prayushi/Desktop/IITD/Asisgnments/ML/Solution/Submission/Q1"

	X = pd.read_csv(input_x, header=None)
	Y = pd.read_csv(input_y, header=None)

	X = np.array(X)
	
	Y = np.array(Y)
	X_normalised = Preprocess(X).normalize()
	X_intercept = np.c_[np.ones(len(X_normalised)),X_normalised]
	theta = np.zeros(2).reshape(-1,1)

	theta = np.zeros(2).reshape(-1,1)
	t, l = GradientDescent(X_intercept, Y, stopping_threshold=0.0001, learning_rate=0.025, auto_convergence_detection=True, iterations=10).vanilla_GD(theta)

	print("Parameters Based on experimentation")
	print(f"Stopping threshold: {0.0001} and learning rate : {0.025}")
	print("-----------------------------------")
	print("Learned Parameters")
	print(f"Theta0 : {t[-1][0][0]}\nTheta1 : {t[-1][1][0]}\nCost in last iteration : {l[-1]}\nTotal iterations : {len(l)}")


	# Plot
	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(X_normalised,Y,'b+')
	ax.plot(X_normalised,X_intercept.dot(t[-1]),'r-')
	ax.set_xlabel("acidity of wine")
	ax.set_ylabel("density of wine", rotation=90)
	ax.axis([-2,4,0.99,1])

	fig.savefig(f'{output_plots_dir_path}/1b_plot.png')

	draw_3d_mesh_and_contour(l, t, output_plots_dir_path)