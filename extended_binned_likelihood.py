#!/usr/bin/python3
# coding:utf-8
"""
Script to show the usecase and implemetation of an binned extended NNL estimator

.. author: Alexander Becker <nabla.becker@mailbox.org>
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
import statmeth as stm
import statplot as stp



#the model used to be fitted to the data is described
def signal(x, theta):
    """Gaussian function representing the signal with theta being the
    parameters of the signal"""
    return theta[0] * np.exp(-(theta[1]-x)**2)

def background(x, theta):
    """exponential decay representing the background of the measurement
    with theta being the parameters of the background"""
    return theta[0] * np.exp(-x*theta[1])

def model(x, theta1, theta2, theta3, theta4):
    """the model that is used to fit to the data using the ML estimator"""
    return background(x, (theta1, theta2)) + signal(x, (theta3, theta4))

#create data for experiment
def create_pseudo_experiment_data(N, bounds, theta, method='rejection'):
    """create random numbers according to the model distribution"""
    if method == 'rejection':
        data, rnum = stm.generate_pdf_rnums_rejection_method(N, model, bounds, *theta)
        return data
    raise NotImplementedError

def create_parameter_scan(params, scan_idx, scan_bounds, var_num):
    """create a list of parameters where the parameter indexed by scand_idx is varied within
    the bounds given by scan bounds with var_num values"""
    scan_params = []
    for i, param in enumerate(params):
        if i == scan_idx:
            scan_params.append(np.linspace(scan_bounds[0], scan_bounds[1], var_num, endpoint=True))
        else:
            scan_params.append(np.full(var_num, params[i]))
    return list(zip(*scan_params))

#the extended binned log likelihood function as defined in the lectures
def extended_binned_log_likelihood(hist, bin_edges, model_func, *params):
    """compute the extended binned log likelyhood given a histogram and its bin edges
    as well as the model function that along with a single variable takes parameters *params"""
    bin_centers = np.array(stm.bin_centers(bin_edges))
    #probability that a measurement gets sorted into the kth bin
    #approximated using the probability at the center of the bin.
    nu_k = model_func(bin_centers, *params)
    #the sps.gamma function is an way to write the factorial of a number
    #the stirling approximation could be used to calculate ln(n!) quicker
    #for large n but a test should probably be put in here then
    return sum(-nu_k -np.log(sps.gamma(hist+1)) + hist * np.log(nu_k))

if __name__ == "__main__":
    #set the parameters for the data
    BOUNDS = (0, 10)
    N = 100000
    params = (1, 0.25, 0.1, 5)
    BIN_COUNT = 25
    #create the histogram of the data
    data = create_pseudo_experiment_data(N, BOUNDS, params)
    hist, bin_edges = np.histogram(data, BIN_COUNT, density=True)

    #rescale model scaling to fit the data scaling
    model_integral = stm.numerical_integrate_1D(model, BOUNDS, *params)
    hist_integral = sum(hist)
    bin_width = bin_edges[1] - bin_edges[0] #assuming uniform spacing
    scale = hist_integral / model_integral * bin_width
    model_params = (params[0]*scale, params[1], params[2]*scale, params[3])

    #do a parameter scan of the signal strength with the ebnll
    scan_points = 100
    scan_bounds = (0.05, .8)
    scan_params = create_parameter_scan(model_params, 2, scan_bounds, scan_points)
    sig_strength_ebnnl = [-extended_binned_log_likelihood(hist, bin_edges, model, *scan_param)
                          for scan_param in scan_params]

    #plot the maximum likelihood scan
    scan_domain = list(zip(*scan_params))[2]
    stp.plot_likelihood_scan_1D(sig_strength_ebnnl, scan_domain, '\lambda')

    #plot the results
    bin_centers = stm.bin_centers(bin_edges)
    plt.hist(data, 100)
    plt.show()
