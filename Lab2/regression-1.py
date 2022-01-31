import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    mean_vec = np.array([0,0])
    covariance = np.array([[beta,0], [0,beta]])

    x_range = np.linspace(-1,1,100)
    y_range = np.linspace(-1,1,100)
    [X, Y] = np.meshgrid(x_range, y_range)
    x_t = X[0].reshape(100,1)
    plot = []

    for i in range(100):
        y_t = Y[i].reshape(100,1)
        x_set = np.concatenate((x_t, y_t), 1)
        plot.append(util.density_Gaussian(mean_vec, covariance, x_set))
    # figure configuration
    plt.figure(1)
    plt.title("prior")
    plt.xlabel("a0")
    plt.ylabel("a1")
    # plot the contour for a
    plt.contour(X,Y,plot)
    # plot the true value point a = (-0.1, -0.5)
    plt.plot([-0.1], [-0.5], marker = 'o', markersize = 5, color = 'blue')
    plt.show()
    
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here
    n = len(x) # training sample size
    cov_a = np.array([[beta,0], [0,beta]])
    cov_a_inv = np.linalg.inv(cov_a)
    cov_w = np.array([sigma2])
    
    arr = np.ones(shape = (n,1))
    A = np.append(arr, x, axis = 1)
    # Cov:
    Cov_inv = cov_a_inv + (np.matmul(A.T, A)/cov_w)
    Cov = np.linalg.inv(Cov_inv)
    # mu:
    mu_sec = np.matmul(A.T, z)/cov_w
    mu = np.matmul(Cov, mu_sec).reshape(1,2).squeeze()

    #print(mu)
    #print(Cov)

    # x: a0; y: a1
    x_range = np.linspace(-1,1,100)
    y_range = np.linspace(-1,1,100)
    [X, Y] = np.meshgrid(x_range, y_range)
    x_t = X[0].reshape(100,1)
    plot = []
    
    for i in range(100):
        y_t = Y[i].reshape(100,1)
        x_set = np.concatenate((x_t, y_t), 1)
        plot.append(util.density_Gaussian(mu, Cov, x_set))
    
    plt.figure(2)
    plt.xlabel("a0")
    plt.ylabel("a1")
    # plot the contour for a
    plt.contour(X,Y,plot)
    
    if n == 1:
        plt.title('posterior1: p(a|x1, z1)')
        # plot the true value point a = (-0.1, -0.5)
        plt.plot([-0.1], [-0.5], marker = 'o', markersize = 5, color = 'blue')
        plt.show()
    if n == 5:
        plt.title('posterior5: p(a|x1, z1, ..., x5, z5)') 
        # plot the true value point a = (-0.1, -0.5)
        plt.plot([-0.1], [-0.5], marker = 'o', markersize = 5, color = 'blue')
        plt.show()
    if n == 100:
        plt.title('posterior100: p(a|x1, z1, ..., x100, z100)')
        # plot the true value point a = (-0.1, -0.5)
        plt.plot([-0.1], [-0.5], marker = 'o', markersize = 5, color = 'blue')
        plt.show()

    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here

    N = len(x_train) # Number of data sample
    A = np.append(np.ones([len(x),1]), np.expand_dims(x, 1), axis = 1) # A = [1 x]

    mu_z = np.dot(A, mu)

    Cov_w = np.array([sigma2])
    first_part = np.matmul(A, Cov)
    Cov_z = Cov_w + np.matmul(first_part, A.T)
    var_z = np.diag(Cov_z)
    std_z = np.sqrt(var_z)

    plt.figure()
    plt.xlabel('x (input)')
    plt.ylabel('z (output)')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    if N == 1 :
        plt.title('predict1: p(z|x,x1,z1)')
    if N == 5 :
        plt.title('predict5: p(z|x,x1,z1,...,x5,z5)')
    if N == 100 :
        plt.title('predict100: p(z|x,x1,z1,...,x100,z100)')
    
    # 1) & 2) Testing 
    plt.errorbar(x, mu_z, yerr = std_z, fmt = 'x', color = 'green')
    # 3) Training
    plt.scatter(x_train, z_train, color = 'red', s=3)

    plt.show()    

    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 100
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
