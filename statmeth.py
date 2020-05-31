# coding:utf-8
"""
.. module:: statmeth
    :platform: Unix
    :synopsis: This module provides often used procedures for generating random data
        and other often used functions
.. moduleauthor:: Alexander Becker <nabla.becker@mailbox.org>
"""
import numpy as np

def bin_centers(bin_edges):
    """calculate the bin centers given the list of bin edges as returned by
    `np.histogram` or `plt.hist` functions"""
    if isinstance(bin_edges, np.ndarray):
        return (bin_edges[1:] + bin_edges[:-1])/2
    return [(bin_edges[i]+edge)/2 for i, edge in enumerate(bin_edges[1:])]


def generate_pdf_rnums_rejection_method(num, pdf, bounds, *args):
    """generate random numbers following the pdf given for
    pdf(*args) using the rejection method and transform the area according tho the pdf"""
    count = 0
    rgen = 0
    transformed_rnums = []
    delta = bounds[1]-bounds[0]
    maxval = max(pdf(np.linspace(bounds[0], bounds[1], int(delta*1000), endpoint=True), *args))
    while count < num:
        coordinate = np.random.rand(2) * np.array([delta, maxval+.1]) + bounds[0]
        rgen += 1
        if coordinate[1] < pdf(coordinate[0], *args):
            transformed_rnums.append(coordinate[0])
            count += 1
    return transformed_rnums, rgen


def transform_to_exp_decay(urnums, theta):
    """transforms uniformally distributed random numbers to random numbers
    with a distribution following an exponential decay"""
    return -theta * np.log(theta*urnums)

def numerical_integrate_1D(func, bounds, *params):
    """numerically integrate the function f(x, *params) inside the
    interval given by bounds using the trapezoidal method"""
    delta_b = np.abs(bounds[1]-bounds[0])
    domain = np.linspace(bounds[0], bounds[1], int(delta_b*1000))
    delta_x = domain[1:] - domain[:-1]
    return sum((func(domain[:-1], *params) + func(domain[1:], *params))/2 + delta_x)

def statistical_integrate_1D(func, bounds, *params):
    """use the statistical method to integrate f"""
    delta_b = bounds[1]-bounds[0]
    data, rlen = generate_pdf_rnums_rejection_method(int(delta_b*1000), func, bounds, *params)
    return len(data)/rlen
