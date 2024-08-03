from sklearn.datasets import load_svmlight_file
import numpy as np


# Extract data from svm file (.txt)
def get_sampled_data(filepath, sample_rate=0.25):
    # load dataset
    A, y = load_svmlight_file(filepath)
    print('Data loaded')
    #sample data
    nsamples = int(sample_rate * A.shape[0])
    indices = np.random.choice(A.shape[0], nsamples, replace=False)

    return A[indices], y[indices]




