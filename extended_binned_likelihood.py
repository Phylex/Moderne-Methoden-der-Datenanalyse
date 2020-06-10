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
def extended_binned_log_likelihood(hist, bin_edges, model_func, expected_count, *params):
    """compute the extended binned log likelyhood given a histogram and its bin edges
    as well as the model function that along with a single variable takes parameters *params"""
    #here we use the model function to calculate the probability that
    #an event will be sortet into the ith bin
    bin_centers = np.array(stm.bin_centers(bin_edges))
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bin_area = model_func(bin_centers, *params) * bin_widths
    total_area = sum(bin_area)
    #probability that a measurement gets sorted into the kth bin
    #approximated using the probability at the center of the bin.
    prob_bin = bin_area / total_area
    #calculate the expected entries into the ith bin
    nu_i = expected_count * prob_bin
    return sum(-nu_i + hist * np.log(nu_i))

if __name__ == "__main__":
    #set the parameters for the data
    BOUNDS = (0, 10)
    N = 4500
    params = (1, 0.25, 0.1, 5)
    BIN_COUNT = 20
    #create the histogram of the data
    data = create_pseudo_experiment_data(N, BOUNDS, params)
    hist, bin_edges = np.histogram(data, BIN_COUNT, density=False)

    #plot the variation of the amplitude of the signal
    scan_vals = np.linspace(0, .5, 6, endpoint=True)
    scan_idx = 2
    scale_dependent_params = [0, 2]
    fitted_params = params
    stp.plot_parameter_variation_over_errorbar(hist, bin_edges, model, fitted_params,
                                               scale_dependent_params, scan_idx, scan_vals,
                                               color='blue-green', param_name=r'\lambda',
                                               title='Variation of signal amplitude', save=True)

    #plot the variation of the position of the signal
    scan_vals = np.linspace(4, 6, 5, endpoint=True)
    scan_idx = 3
    stp.plot_parameter_variation_over_errorbar(hist, bin_edges, model, fitted_params,
                                               scale_dependent_params, scan_idx, scan_vals,
                                               color='red-yellow', param_name=r'\eta',
                                               title='Variarion of signal position', save=True)

    #set common scan parameters
    scan_points = 100
    #do a parameter scan of the signal strength with the ebnll
    scan_bounds = (0.01, .2)
    scan_params = create_parameter_scan(params, 2, scan_bounds, scan_points)
    sig_strength_ebnnl = [-extended_binned_log_likelihood(hist, bin_edges, model, N, *scan_param)
                          for scan_param in scan_params]
    #plot the maximum likelihood scan for the amplitude
    scan_domain = list(zip(*scan_params))[2]
    stp.plot_likelihood_scan_1D(sig_strength_ebnnl, scan_domain, r'\lambda',
                                plot_delta=True, palette='blue-green', title='Signal amplitude scan')

    #plot the maximum likelihood scan for the position
    scan_bounds = (4, 6)
    scan_params = create_parameter_scan(params, 3, scan_bounds, scan_points)
    sig_position_ebnnl = [-extended_binned_log_likelihood(hist, bin_edges, model, N, *scan_param)
                          for scan_param in scan_params]
    scan_domain = list(zip(*scan_params))[3]
    stp.plot_likelihood_scan_1D(sig_position_ebnnl, scan_domain, r'\eta',
                                plot_delta=True, palette='red-yellow', title='Signal position scan')
