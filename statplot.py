# coding:utf-8
'''
Ploting module for automating plots for different scenarios

.. author: Alexander Becker <nabla.becker@mailbox.org>
'''
import itertools as itt
import matplotlib.pyplot as plt
import numpy as np
import statmeth as stm

def plot_likelihood_scan_1D(scan_vals, scan_domain, parameter_name, plot_delta=False):
    #convert the scan vals to an np.array
    scan_vals = np.array(scan_vals)
    theta_min, theta_max = min(scan_domain), max(scan_domain)
    min_nll = min(scan_vals)
    min_nll_idx = scan_vals.argmin()
    if plot_delta:
        scan_vals = scan_vals - min_nll
        min_nll = 0
    #split the array into the part left of the min and right of the min
    #so that it is possible to find the closest array vals to +/- sigma
    left_side = scan_vals[:min_nll_idx]
    right_side = scan_vals[min_nll_idx:]
    left_sigma, ls_idx = stm.find_nearest(left_side, min_nll+.5)
    right_sigma, rs_idx = stm.find_nearest(right_side, min_nll+.5)
    rs_idx = rs_idx + len(left_side)

    if plot_delta:
        scan_vals = scan_vals - min_nll

    #plot likelihood scan
    plt.plot(scan_domain, scan_vals, color='darkblue', marker='o',
             label=r'Likelihood Scan for $'+parameter_name+'$')
    #plot the lines indicating the differnt sections
    xmin, xmax, ymin, ymax = plt.axis()
    #horizontal lines for the minimum and the min+1/2 lines
    if not plot_delta:
        plt.hlines(y=min_nll, xmin=xmin, xmax=xmax, color='blue', label='NLL minimum')
    plt.hlines(y=min_nll+(1./2.), xmin=xmin, xmax=xmax, color='lightblue', linestyle='--',
               label='standard deviation')
    #vertical lines for the center and the left and right standard deviations
    plt.vlines(x=scan_domain[ls_idx], ymin=scan_vals[ls_idx], ymax=ymax, color='cyan',
               label=r'$\hat{'+parameter_name+'} \pm \sigma$')
    plt.vlines(x=scan_domain[rs_idx], ymax=ymax, ymin=scan_vals[rs_idx], color='cyan')
    plt.vlines(x=scan_domain[min_nll_idx], ymin=min_nll, ymax=ymax, color='gray',
               label=r'$\hat{ '+parameter_name+'}_{ ML } = '+str(np.round(scan_domain[min_nll_idx],4))+'$')
    #grid and legend
    plt.grid(linestyle=':')
    plt.legend()
    plt.savefig('{}_likelihood_scan.svg'.format(parameter_name))
    plt.show()

def plot_parameter_variation_over_errorbar(hist, bin_edges, model, fitted_params, scale_dependent_params, var_param_idx, var_vals, density=False):
    colors = ['navy', 'blue', 'royalblue', 'dodgerblue', 'deepskyblue', 'darkturquoise']
    #scale the scalable parameters to the scale of the histogramm
    #first determin the area unter the histogram as an estimate for area
    bin_width = bin_edges[1:] - bin_edges[:-1]
    hist_int = sum(hist*bin_width)
    #now determin the integral of the model
    bounds = (min(bin_edges), max(bin_edges))
    model_int = stm.numerical_integrate_1D(model, bounds, *fitted_params)
    #the scale is defined as the ratio of the areas
    print(fitted_params)
    scale = hist_int / model_int
    print(hist_int)
    print(model_int)
    print(scale)
    #prepare variables for the plot
    domain = np.linspace(bounds[0], bounds[1], 1000)
    bin_centers = stm.bin_centers(bin_edges)
    cyclic_colors = itt.cycle(colors)
    #do the parameter variation
    for var_val, color in zip(var_vals, cyclic_colors):
        #set the parameter to vary to the value of the current iteration
        plot_params = [var_val if i == var_param_idx else param for i, param in enumerate(fitted_params)]
        #apply the scaling
        plot_params = [scale*param if i in scale_dependent_params else param for i, param in enumerate(plot_params)]
        #plot
        plt.plot(domain, model(domain, *plot_params), color=color)
    #plot the
    yerr = np.sqrt(hist)
    plt.errorbar(bin_centers, hist, yerr=yerr, color=next(cyclic_colors), linestyle='')
    plt.show()
