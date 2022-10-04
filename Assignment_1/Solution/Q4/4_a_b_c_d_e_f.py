# Imports

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



class GDA():
    def __init__(self, X, Y):  
        self.X = X
        self.Y = Y
        self.p1 = None
        self.mu1 = None
        self.mu2 = None
    
    def linear_fit(self, class_1_data, class_2_data):
        
        c1 = class_1_data - self.mu1
        c2 = class_2_data - self.mu2
        sigma = ((c1.T).dot(c1) + (c2.T).dot(c2))/len(self.X)
        sigma_inv = np.linalg.inv(sigma)
        mu1 = self.mu1.reshape(-1,1)
        mu2 = self.mu2.reshape(-1,1)
        
        theta = (2 * sigma_inv).dot(mu1-mu2)
        theta0 = (mu1.T).dot(sigma_inv).dot(mu1) - (mu2.T).dot(sigma_inv).dot(mu2) + np.log(self.p1) - np.log(1-self.p1)
        
        print("\n---------------------")
        print("Params for Linear Fit")
        print(f" mu1 : {mu1},\n mu2 : {mu2},\n sigma : {sigma}")
        return np.array([theta0[0][0], theta[0][0], theta[1][0]])
    
    def quadratic_fit(self, class_1_data, class_2_data):
        
        c1 = class_1_data - self.mu1
        c2 = class_2_data - self.mu2
        
        sigma_1 = ((c1.T).dot(c1))/len(c1)
        sigma_1_inv = np.linalg.inv(sigma_1)
        sigma_2 = ((c2.T).dot(c2))/len(c2)
        sigma_2_inv = np.linalg.inv(sigma_2)
        
        mu1 = self.mu1.reshape(-1,1)
        mu2 = self.mu2.reshape(-1,1)
        
        theta2 = sigma_2_inv - sigma_1_inv
        theta1 = (2 * sigma_1_inv).dot(mu1) - (2 * sigma_2_inv).dot(mu2) 
        theta0 = -(mu1.T).dot(sigma_1_inv).dot(mu1) + (mu2.T).dot(sigma_2_inv).dot(mu2) + np.log(self.p1) - np.log(1-self.p1) - np.log(np.linalg.det(sigma_1_inv))/2 + np.log(np.linalg.det(sigma_2_inv))/2
        
        print("\n---------------------")
        print("Params for Quadratic Fit")
        print(f" mu1 : {mu1},\n mu2 : {mu2},\n sigma1 : {sigma_1},\n sigma2 : {sigma_2}")
        return np.array([theta2[0][0], theta2[0][1], theta2[1][1], theta1[0][0], theta1[1][0], theta0[0][0]])
        
    def fit_model(self, is_linear_fit = True):
                
        class_1_data = self.X[np.where(self.Y[:,0] == 1)]
        class_2_data = self.X[np.where(self.Y[:,0] == 0)]
        self.mu1 = np.mean(class_1_data, axis=0)
        self.mu2 = np.mean(class_2_data, axis=0)
        self.p1 = len(class_1_data)/(len(class_1_data)+len(class_2_data))
        params= None
        if(is_linear_fit):
            params = self.linear_fit(class_1_data, class_2_data)
            return params
        else:
            return self.quadratic_fit(class_1_data, class_2_data)

if __name__ =="__main__":

	# Input file paths
	inputx = "/home/prayushi/Desktop/IITD/Asisgnments/ML/Assignment_1/Data/q4/q4x.dat"
	inputy = "/home/prayushi/Desktop/IITD/Asisgnments/ML/Assignment_1/Data/q4/q4y.dat"
	output_plots_dir_path = "/home/prayushi/Desktop/IITD/Asisgnments/ML/Solution/Submission/Q4"

	# Read files
	X = np.loadtxt(inputx, unpack = True).T
	Y = np.loadtxt(inputy, unpack = True,dtype='str').reshape(-1,1)

	# Encode Output variables
	Y[Y=="Alaska"]=1
	Y[Y=="Canada"]=0
	Y = Y.astype("int")

	# Preprocess
	x1_normalized = Preprocess(X[:,0]).normalize()
	x2_normalized = Preprocess(X[:,1]).normalize()

	X_normalized = np.c_[x1_normalized,x2_normalized]

	# Fit Linear model
	linear_coeffs = GDA(X_normalized, Y).fit_model(is_linear_fit=True)

	# Fit Quadratic model
	quadratic_coeffs = GDA(X_normalized, Y).fit_model(is_linear_fit=False)

	# Draw Plots
	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.set_xlabel("x1")
	ax.set_ylabel("x2", rotation=90)
	ax.plot(x1_normalized[:50],x2_normalized[:50],'b+', label='Alaska')
	ax.plot(x1_normalized[50:],x2_normalized[50:],'r*', label='Canada')
	ax.legend(loc ="upper right")

	fig.savefig(f'{output_plots_dir_path}/4b_plot.png')

	# Draw linear boundary
	x1 = np.linspace(-1.5, 2.3, 100)
	x2 = (linear_coeffs[0] - linear_coeffs[1]*x1)/linear_coeffs[2]
	ax.plot(x1,x2,'y-')

	fig.savefig(f'{output_plots_dir_path}/4c_plot.png')


	# Draw quadratic boundary
	yplot = list()
	xplot = list()
	x = np.linspace(-1.5,2,100)
	for i in x:
	    c1 = quadratic_coeffs[0]*i*i + quadratic_coeffs[3]*i + quadratic_coeffs[5]
	    b1 = 2*quadratic_coeffs[1]*i + quadratic_coeffs[4]
	    a1 = quadratic_coeffs[2]
	    xplot.append(i)
	    yplot.append((-1*b1 + ((b1*b1 - 4*a1*c1)**(1/2)))/(2*a1))

	ax.plot(xplot,yplot,'g-')

	fig.savefig(f'{output_plots_dir_path}/4e_plot.png')
