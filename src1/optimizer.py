import numpy as np
from tqdm import tqdm
from scipy.linalg import norm
import psutil

# Batch Stochastic gradient implementation
def sgd(x0,problem, xtarget, lr_choice=1,step0=1, n_iter=1000,
        batch_size=1,with_replace=False,verbose=False):
    """
        A code for gradient descent with various step choices.

        Inputs:
            x0: Initial vector
            problem: Problem structure
                problem.fun() returns the objective function, which is assumed to be a finite sum of functions
                problem.n returns the number of components in the finite sum
                problem.grad_i() returns the gradient of a single component f_i
                problem.cvxval() returns the strong convexity constant
                problem.lambda returns the value of the regularization parameter
            stepchoice: Strategy for computing the stepsize
                0: Constant step size equal to 1/L
                t>0: Step size decreasing in 1/(k+1)**t
            step0: Initial steplength (only used when stepchoice is not 0)
            n_iter: Number of iterations, used as stopping criterion
            nb: Number of components drawn per iteration/Batch size
                1: Classical stochastic gradient algorithm (default value)
            with_replace: Boolean indicating whether components are drawn with or without replacement
                True: Components drawn with replacement
                False: Components drawn without replacement (Default)
            verbose: Boolean indicating whether information should be plot at every iteration (Default: False)

        Outputs:
            x_output: Final iterate of the method (or average if average=1)
            objvals: History of function values (Numpy array of length n_iter at most)
            normits: History of distances between iterates and optimum (Numpy array of length n_iter at most)
    """
    ############
    # Initial step: Compute and plot some initial quantities

    # objective history
    objvals = []
    normits = []
    # Number of samples
    n = problem.n

    # Number of variables
    d = problem.d

    # Initial value of current iterate
    x = x0.copy()

    # Initialize iteration counter
    k=0

    #Set learning rate with Lipschitz constant
    lr = lr_choice/problem.L

    # Current objective
    obj = problem.fun(x)
    objvals.append(obj)

    ################
    # Main loop

    print("Stochastic Gradient, batch size=",batch_size,"/",n)

    for k in tqdm(range(n_iter)):
        # Draw the batch indices
        ik = np.random.choice(n,batch_size,replace=with_replace)# Batch gradient
        # Stochastic gradient calculation
        sg = np.zeros(d)
        for j in range(batch_size):
            gi = problem.grad_i(ik[j],x)
            sg = sg + gi
        sg = (1/batch_size)*sg
        x[:] = x - lr * sg

        # Plot quantities of interest at the end of every epoch only
        if k % (n // batch_size) == 0:
            obj = problem.fun(x)
            nmin = norm(x-xtarget)
    
            objvals.append(obj)
            normits.append(nmin)
            if verbose:
                print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8)]))


    x_output = x.copy()

    return x_output, np.array(objvals), np.array(normits)


#########################################


# Quasi-Newton batch stochastic gradient implementation


def bfgs(x0,problem, xtarget, lr_choice=1,step0=1, n_iter=1000,
        batch_size=1,with_replace=False,verbose=False):
    ############
    # Initial step: Compute and plot some initial quantities

    # objective history
    objvals = []
    normits = []

    # Number of samples
    n = problem.n

    # Number of variables
    d = problem.d

    # Initial value of current iterate
    x = x0.copy()


    # Initialize iteration counter
    k=0

    #Set learning rate with Lipschitz constant
    lr = lr_choice/problem.L

    # Current objective
    obj = problem.fun(x)
    objvals.append(obj)
    normits = []


    
    I = np.identity(problem.d)
    H = np.copy(I)
  
  
    print("Batch quasi-Newton method, batch size=",batch_size,"/",n)
    if verbose:
      # Plot initial quantities of interest
      print(' | '.join([name.center(8) for name in ["iter", "fval"]]))
      print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8)]))
    
    sg = np.zeros(d)
    ik = np.random.choice(n,batch_size,replace=with_replace)
    for j in range(batch_size):
        gi = problem.grad_i(ik[j],x)
        sg = sg + gi
    sg = (1/batch_size)*sg
    
    
    ################
    # Main loop
    for k in tqdm(range(n_iter)):
        pre_x = np.copy(x)
        x = x - lr * H @ sg
        if k % (n // batch_size) == 0:
              obj = problem.fun(x)
              objvals.append(obj)

              if verbose:
                  print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8)])) 
        
        # nx = norm(x) #Computing the norm to measure divergence
        # k += 1
        # if nx > 10**100:
        #     print('Gradient norm exploding')
        #     break

        # compute H_k
        pre_grad = np.copy(sg)
        sg = np.zeros(d)
        ik = np.random.choice(n,batch_size,replace=with_replace)
        for j in range(batch_size):
            gi = problem.grad_i(ik[j],x)
            sg = sg + gi
        sg = (1/batch_size)*sg
        v = sg - pre_grad
        s = x - pre_x
        
        sv = np.dot(s.T, v)

        # to avoid overflow (final value of H explode when sv is too small)
        # we modify the theoretical condition sv > 0 
        if sv > 10e-6:
          VS = np.dot(v.reshape(-1, 1), s.reshape(-1,1).T)
          M = I - VS
          H = np.dot(np.dot(M.T, H), M) + np.dot(s.reshape(-1, 1),s.reshape(-1, 1).T)/sv
     
    x_output = x.copy()
    return x_output, np.array(objvals)



#########################################


# Quasi-Newton batch stochastic gradient implementation, limited-memory variant

def l_bfgs(x0,problem, xtarget, lr_choice=1, n_iter=1000,
         batch_size=1, memory = 0, with_replace=False,verbose=False):
    ############
    # Initial step: Compute and plot some initial quantities

    # objective history
    objvals = []
    normits = []

    # Number of samples
    n = problem.n

    # Number of variables
    d = problem.d

    # Initial value of current iterate
    x = x0.copy()

    # Initialize iteration counter
    k=0

    #Set learning rate with Lipschitz constant
    lr = lr_choice/problem.L

    # Current objective
    obj = problem.fun(x)
    objvals.append(obj)
    

    

    print("Low Memory Batch quasi-Newton method, batch size=",batch_size,"/",n, "|| memory=",memory)
    #batch stochastic descent
    if verbose:
        # Plot initial quantities of interest
        print(' | '.join([name.center(8) for name in ["iter", "fval"]]))
        print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8)]))
    
    
    #Initialize gradient first value
    sg = np.zeros(d)
    ik = np.random.choice(n,batch_size,replace=with_replace)
    for j in range(batch_size):
        gi = problem.grad_i(ik[j],x)
        sg = sg + gi
    sg = (1/batch_size)*sg
   
    
    I = np.eye(problem.d)
    H = np.copy(I)
    S, V = np.zeros((memory, problem.d)), np.zeros((memory, problem.d))
   
    # Main loop
    for k in tqdm(range(n_iter)):
        pre_x = x
        x = x - lr * H @ sg

        if k % (n // batch_size) == 0:
              obj = problem.fun(x)
              objvals.append(obj)

        # compute H_k
        pre_grad = sg
        sg = np.zeros(d)
        ik = np.random.choice(n,batch_size,replace=with_replace)
        for j in range(batch_size):
            gi = problem.grad_i(ik[j],x)
            sg = sg + gi
        sg = (1/batch_size)*sg
        v = sg - pre_grad
        s = x - pre_x


        # update S and V memory vector
        V[1:] = V[:memory-1]
        S[1:] = S[:memory-1]
        V[0] = v
        S[0] = s

        H = np.copy(I)
        #print([np.dot(a.T, b) for a,b in zip(S, V)])
        for i in range(min(k, memory)):
            sv = np.dot(S[i].T, V[i])    
            if sv > 10e-7:
                VS = np.dot(V[i].reshape(-1, 1), S[i].reshape(-1,1).T)
                M = I - VS
                H = np.dot(np.dot(M.T, H), M) + np.dot(S[i].reshape(-1, 1),S[i].reshape(-1, 1).T)/sv

    x_output = x.copy()

    return x_output, np.array(objvals)
