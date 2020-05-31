# coding:utf-8
'''
Ploting module for automating plots for different scenarios

.. author: Alexander Becker <nabla.becker@mailbox.org>
'''
import matplotlib.pyplot as plt
import numpy as np
import statmeth as stm

def plot_likelihood_scan_1D(scan_vals, scan_domain, parameter_name):
    #convert the scan vals to an np.array
    scan_vals = np.array(scan_vals)
    theta_min, theta_max = min(scan_domain), max(scan_domain)
    min_nll = min(scan_vals)
    min_nll_idx = scan_vals.argmin()
    #split the array into the part left of the min and right of the min
    #so that it is possible to find the closest array vals to +/- sigma
    left_side = scan_vals[:min_nll_idx]
    right_side = scan_vals[min_nll_idx:]
    left_sigma, ls_idx = stm.find_nearest(left_side, min_nll+.5)
    right_sigma, rs_idx = stm.find_nearest(right_side, min_nll+.5)
    rs_idx = rs_idx + len(left_side)

    #plot likelihood scan
    plt.plot(scan_domain, scan_vals, color='darkblue', marker='o',
             label=r'Likelihood Scan for $'+parameter_name+'$')
    #plot the lines indicating the differnt sections
    xmin, xmax, ymin, ymax = plt.axis()
    #horizontal lines for the minimum and the min+1/2 lines
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
