#!/usr/bin/env python3

# SEIR model with social distancing
# adapted from https://github.com/jckantor/covid-19

import numpy as np

from graphing import plot_infection_rates, save_png
from seir_simulation import simulate_infection

def simulation_parameters():
    N = 14000
    n = 10
    t_incubation = 5.1
    t_infective = 3.3
    t_social_distancing = 2
    u_social_distancing = 40
    R0 = 2.4
    alpha = 1/t_incubation
    gamma = 1/t_infective
    beta = R0*gamma
    t = np.linspace(0, 210, 210)
    return {
        'R0': R0,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        't': t,
        't_social_distancing': t_social_distancing,
        'u_social_distancing': u_social_distancing,
        'N': N,
        'n': n,
    }

def initial_population(parameters):
    """initial number of infected and recovered individuals"""
    e_initial = parameters['n']/parameters['N']
    i_initial = 0.00
    r_initial = 0.00
    s_initial = 1 - e_initial - i_initial - r_initial
    return s_initial, e_initial, i_initial, r_initial

parameters = simulation_parameters()
x_initial = initial_population(parameters)
(with_distancing, without_distancing) = simulate_infection(x_initial, parameters)
figure = plot_infection_rates(parameters, with_distancing, without_distancing)
save_png(figure, 'covid-19')