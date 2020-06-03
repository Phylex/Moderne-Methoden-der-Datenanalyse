# coding:utf-8
'''
Ploting module for automating plots for different scenarios

.. author: Alexander Becker <nabla.becker@mailbox.org>
'''
import itertools as itt
import matplotlib.pyplot as plt
import numpy as np
import statmeth as stm

PALETTS = {'blue-green': ['navy', 'blue', 'royalblue', 'dodgerblue',
                          'deepskyblue', 'darkturquoise', 'turquoise', 'aqua', 'aquamarine'],
           'green-yellow': ['darkgreen', 'green', 'limegreen', 'lime', 'lawngreen',
                            'greenyellow', 'yellowgreen', 'olivedrap', 'darkolivegreen'],
           'red-yellow': ['firebrick', 'lightcoral', 'coral',
                          'orangered', 'red', 'darkorange', 'orange', 'gold', 'goldenrod'],
          }

def plot_likelihood_scan_1D(scan_vals, scan_domain, parameter_name, title='Likelihood scan', palette='blue-green', plot_delta=False):
    #convert the scan vals to an np.array
    scan_vals = np.array(scan_vals)
    min_nll = min(scan_vals)
    min_nll_idx = scan_vals.argmin()
    if plot_delta:
        scan_vals = scan_vals - min_nll
        min_nll = 0
    #split the array into the part left of the min and right of the min
    #so that it is possible to find the closest array vals to +/- sigma
    left_side = scan_vals[:min_nll_idx]
    right_side = scan_vals[min_nll_idx:]
    _left_sigma, ls_idx = stm.find_nearest(left_side, min_nll+.5)
    _right_sigma, rs_idx = stm.find_nearest(right_side, min_nll+.5)
    rs_idx = rs_idx + len(left_side)

    if plot_delta:
        scan_vals = scan_vals - min_nll

    #plot likelihood scan
    #set up color iterator
    color_it = iter(PALETTS[palette])
    color = next(color_it)
    plt.plot(scan_domain, scan_vals, color=color, marker='o',
             label=r'Likelihood Scan for $'+parameter_name+'$')
    #plot the lines indicating the differnt sections
    xmin, xmax, _ymin, ymax = plt.axis()
    #horizontal lines for the minimum and the min+1/2 lines
    color = next(color_it)
    if not plot_delta:
        plt.hlines(y=min_nll, xmin=xmin, xmax=xmax, color=color, label='NLL minimum')
    color = next(color_it)
    plt.hlines(y=min_nll+(1./2.), xmin=xmin, xmax=xmax, color=color, linestyle='--',
               label='standard deviation')
    #vertical lines for the center and the left and right standard deviations
    color = next(color_it)
    color = next(color_it)
    plt.vlines(x=scan_domain[ls_idx], ymin=scan_vals[ls_idx], ymax=ymax, color=color,
               label=r'$\hat{'+parameter_name+'} \pm \sigma$')
    plt.vlines(x=scan_domain[rs_idx], ymax=ymax, ymin=scan_vals[rs_idx], color=color)
    plt.vlines(x=scan_domain[min_nll_idx], ymin=min_nll, ymax=ymax, color='slategray',
               label=r'$\hat{ '+parameter_name+'}_{ ML } = '+str(np.round(scan_domain[min_nll_idx],4))+'$')
    #grid and legend
    plt.grid(linestyle=':')
    plt.legend()
    plt.title(title)
    plt.savefig('{}_likelihood_scan.svg'.format(parameter_name))
    plt.show()

def plot_parameter_variation_over_errorbar(hist, bin_edges, model, fitted_params,
                                           scale_dependent_params, var_param_idx, var_vals,
                                           param_name=r'\theta', color='blue-green',
                                           save=False, title='Variation of a parameter'):
    """plots a sequence of functions where the parameter indicated by `var_param_idx` is run
    through all values in `var_vals` while all other parameters stay unchanged (set to the
    value in fitted_params. They are underlayed with data to which the function params are
    supposed to be fitted. The scale is automatically adjusted to match the fit function and
    the data in the histogram given by hist and bin edges. The indicies of the scale
    dependent parameters are given in scale_dependent_parameters. The color of the curves can be
    selected from a range of pallets ('blue-green', 'green-yellow', and 'red-yellow').
    """
    colors = PALETTS[color]
    #scale the scalable parameters to the scale of the histogramm
    #first determin the area unter the histogram as an estimate for area
    bin_width = bin_edges[1:] - bin_edges[:-1]
    hist_int = sum(hist*bin_width)
    #now determin the integral of the model
    bounds = (min(bin_edges), max(bin_edges))
    model_int = stm.numerical_integrate_1D(model, bounds, *fitted_params)
    #the scale is defined as the ratio of the areas
    scale = hist_int / model_int
    #prepare variables for the plot
    domain = np.linspace(bounds[0], bounds[1], 1000)
    bin_centers = stm.bin_centers(bin_edges)
    cyclic_colors = itt.cycle(colors)
    #do the parameter variation
    for var_val, color in zip(var_vals, cyclic_colors):
        #set the parameter to vary to the value of the current iteration
        plot_params = [var_val if i == var_param_idx else param
                       for i, param in enumerate(fitted_params)]
        #apply the scaling
        plot_params = [scale*param if i in scale_dependent_params else param
                       for i, param in enumerate(plot_params)]
        #plot
        plt.plot(domain, model(domain, *plot_params), color=color,
                 label=r'$'+param_name+r'$ = '+str(np.round(var_val, 3)))
    #plot the errorbars
    yerr = np.sqrt(hist)
    plt.errorbar(bin_centers, hist, yerr=yerr, color='black', marker='o',
                 linestyle='', label='Events')
    #draw the supplemental information
    plt.grid(linestyle=':')
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig('variation_of_param_'+param_name+'.svg')
    plt.show()

def plot_param_variation_1D(func, param_sets, plot_bounds, legend=True, labels=None,
                            var_var_name=None, var_var_idx=None, palette='blue-green', title=None,
                            axis=None):
    """Plot the function given py func once for every parameter set given in param_sets.
    The domain of the plot is bounded by plot_bounds. The curves are assigned a label
    from labels if labels is not none. labels has to be a list of labels with the same
    length as the list of param sets. If a descriptive label is not needed the name of
    the parameter can be given that is varied (in param sets) and a label will be
    automatically generated (this only works when one parameter is varied. If both
    Labels and var_var_name are None and the length of the parameter set is smaller
    than 7 every parameter is given a generic name and printed in the legend with
    the value of the corresponding parameter. The legend can be disabled with
    legend=False.
    """
    #set up the axes if none where passed in, otherwis use the given one
    if axis is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
    else:
        ax = axis
    #color for the diffent curves
    color = itt.cycle(PALETTS[palette])
    #build the labels
    if legend and labels is None:
        #autogenerate labels
        if var_var_name is None:
            if len(param_sets[0]) < 7:
                labels = []
                for param_set in param_sets:
                    base_label = r'$\theta_{} = {:.3f}$ '
                    label = ''
                    for i, param in enumerate(param_set):
                        label += base_label.format(i, float(param))
                    labels.append(label)
            else:
                labels = [None for param_set in param_sets]
        #generate labels from the parameter name
        else:
            if var_var_idx is not None:
                base_label = r'$'+var_var_name+' = {:.3f}$'
                labels = []
                for param_set in param_sets:
                    label = base_label.format(param_set[var_var_idx])
                    labels.append(label)
            else:
                labels = [None for param_set in param_sets]
    #prepare the domain of the plot
    delta_bounds = plot_bounds[1]-plot_bounds[0]
    domain = np.linspace(plot_bounds[0], plot_bounds[1], delta_bounds*1000)
    #calculate the curves
    curves = [func(domain, *param_set) for param_set in param_sets]
    #plot the curves
    for curve, label, cur_col in zip(curves, labels, color):
        ax.plot(domain, curve, label=label, color=cur_col)
    #make the plot look nice
    if legend:
        ax.legend()
    if title is not None:
        ax.set_title(title)
    ax.grid()
    if axis is None:
        return fig, ax
