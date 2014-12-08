import numpy as np

def computeCost(X,y,theta):
    """
    The computeCost function calculates the Cost Function for Linear Regression
    for given values of x, y and theta. It is used to optimize the values
    of theta for given x's and y's.
    """
    
    """ 
    You need to return the correct calculation of variable cost
    which represents the total cost. Here, we initialize it to a
    default value of 0. Formally, the calculation should be:
    cost = 1/(2m) * sum (from i=0 to i=m) (h(x^i) - y^i)**2
    where h(x^i)=theta_0 * x^i_0 + theta_1 * x^i_1 + theta_2 * x^i_2
    NOTE: x^i does NOT  mean 'raised to the ith power', but
          only signifies the ith example
    """
    cost = 0

    # initialize a useful variable: the number of examples m
    m = X.shape[0]
    # below we calculate cost without vectorization
    # Go through the code line by line and make sure you understand
    # how the cost is calculated and that is as described above
    ##sum = 0 
   ## for i in range(0,m):
        ## h = theta[0,0] * X[i,0]+ theta[0,1] * X[i,1] + theta[0,2] * X[i,2]
         ##error = h - y[i]
         ##error = error**2
         ##sum = sum + error

   ## cost = sum / (2*m)
    h=x.dot(theta.transpose())
    error=h-y
    error=error**2
    sum=np.sum(error)
    cost = sum/(2*m)
    """ 
    ---  Question 3 (20%) ---
    You should comment out the above lines of code and 
    do the calculation of the cost using vectorization.
    You should return the same number as the above calculation does.
    Tip 1: You will want to use the np.sum function.
    Tip 2: A good vectorized version will do the above calculation in only 1 line of code!
    Tip 3: Start by vectorizing the h(x) - refer to assignment brief. You will calculate the
             h(x) for all the training examples in X by vectorizing this calculation. Mathematically,
             this calculation is: X * theta. 
             Then vectorize h(x)-y.
             Then add the (h(x)-y)**2. Then, incorporate the np.sum function to calculate the total error. Lastly, put the
             finishing touches with dividing by 2m.
    """

    return cost


def featureNormalize(X):
    """
    This function normalizes the features in X, stores them in X_norm and returns it. Normalized elements X_norm[i,j] should be calculated as:
    X_norm[i,j] = (X[i,j] - mean[i]) / sigma[i]
    where:
    mean[i] is the mean (e.g., arithmetic average) of all the numbers in column i (i=1,2)
    sigma[i] is the standard deviation of all the numbers in column i (i=1,2)
    ATTENTION: You should NOT normalize the 1st column of X, that contains 1's. This should remain unchanged!
    """

    m = X.shape[0]
    n = X.shape[1]
    # You need to set these values correctly. Here we initialize them.
    X_norm = np.ones((X.shape[0],X.shape[1]))
    mu = np.zeros(X.shape[1])      # will store the mean values of X columns 0,1,2
    sigma = np.zeros(X.shape[1])   # will store the standard deviation of values of X columns 0,1,2

    # To help you, we show you how to calculate the mu and sigma vectors using vectorization.
    mu = np.mean(X, axis=0)
    sigma = np.std(X,axis=0)
    
    # The lines below will help you in vectorizing the calculation of X_norm below
    # and avoiding changing the values of the 1st column.
    mu[0] = 0
    sigma[0] = 1              

    # In contrast, if we were to calculate mu without vectorization we'd have to code this:
    bad_mu = np.zeros(X.shape[1])
    for i in range(0,m):
        for j in range(1,n):
            bad_mu[j] = bad_mu[j] + X[i,j]
    
    for j in range(1,n):
        bad_mu[j] = bad_mu[j] / m
    # Obviously, the vectorized version is much more easy to code and less prone to bugs. We'll ignore bad_mu from now on.    


    """
    Next, we iterate through the X matrix and normalize features in columns 1 and 2.
    ---  Question 4 (20%) ---
    You should comment out the code below and refactor the process by vectorising it.
    Pay attention to NOT change the values of the 1st column of the X matrix (they should remain 1's).
    As always, a good vectorized version could be just 1 line. You'll need to use the mu and sigma
    that were calculated above.
    """
    for i in range(0,m):
        for j in range(1,n):
            X_norm[i,j] = (X[i,j] - mu[j]) / sigma[j]


    return X_norm, mu, sigma


def gradientDescent(X,y,alpha,num_iterations):

    # Initialize some useful values
    m = X.shape[0]                           # number of training examples
    theta = np.zeros((1,X.shape[1]))         # initial values of theta=[0,0,0]
    J_history = np.array([])                 # will store the values of the cost function for each iteration

    theta_temp = np.zeros((1,X.shape[1]))   
    for k in range(0,num_iterations):
        for j in range (0,theta.shape[1]):
            sum = 0
            for i in range(0,m):
                sum = sum + (X[i,:].dot(theta.transpose()) - y[i])*X[i,j]
            theta_temp[0,j] = theta[0,j] - (alpha / m) * sum
        theta = theta_temp.copy()
        J_history = np.append(J_history,computeCost(X,y,theta))

    return theta, J_history



