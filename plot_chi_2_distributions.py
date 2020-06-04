#!/usr/bin/python3
# coding:utf-8
"""A script that plots the chi_squared distribution vor different n
.. author: Alexander Becker <nabla.becker@mailbox.org>
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import statplot as stp

def chi_squared_distribution(x, n):
    """the chi squared distribution"""
    return 1/(2**(n/2)*gamma(n/2)) * np.exp(-x/2) * x**(n/2-1)

if __name__ == "__main__":
    n_param_set = [2, 3, 5, 10, 15, 20]
    n_param_set = [(x, ) for x in n_param_set]
    fig, ax = stp.plot_param_variation_1D(chi_squared_distribution, n_param_set, (0, 30),
                                var_var_name='n', var_var_idx=0, palette='green-yellow',
                                title=r'$\chi^2$ distribution for various n')
    fig.savefig('chi_squared_dist.svg')
    plt.show()

