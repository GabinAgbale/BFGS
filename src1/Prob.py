import numpy as np
from numpy.linalg import norm


"""
 Define Logistic Regression Problem, with:
 A : feature matrix (np.array: n_samples x n_features)
 y : labels (np.array: n_samples)
"""

def train_test_split(A, y, rate=0.25):
    nsamples = int(rate * A.shape[0])
    indices_test = np.random.choice(A.shape[0], nsamples, replace=False)
    indices_train = np.array(list(set(range(A.shape[0])) - set(indices_test)))
    
    return A[indices_train], y[indices_train], A[indices_test], y[indices_test]


class LogRegPb(object):

    def __init__(self, A, y):
        
        self.A,self.y,self.A_test,self.y_test = train_test_split(A, y)
        
        self.n, self.d = self.A.shape

        # Lipschitz constant 
        self.L = norm(self.A, ord=2) ** 2 / (4. * self.n)

    # Partial objective function
    def f_i(self, i, x):
        return (self.y[i] - 1/(1 + np.exp(-np.dot(self.A[i].T, x)) ))**2

    def fun(self, x):
        #return(np.mean([self.f_i(i, x) for i in range(self.n)]))
        total = 0
        for i in range(self.n):
          total += self.f_i(i, x)
        return total/self.n

    # Partial gradient
    def grad_i(self, i, x):
        E = np.exp(np.dot(self.A[i].T, x))
        nom = 2 * E * (E * (self.y[i] - 1) + self.y[i])
        denom = (1 + E)**3
        return -  (nom / denom) * self.A[i]

    # full gradient
    def grad(self, x):
        total = 0
        for i in range(self.n):
          total += self.grad_i(i, x)
        return total/self.n