#!/usr/bin/env python3

# SEIR model with social distancing
# adapted from https://github.com/jckantor/covid-19

import numpy as np
from scipy.integrate import odeint

from graphing import plot_infection_rates, save_png

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

def evolve_simulation_over_time(derivative_function, initial_population, times, parameters):
    return odeint(derivative_function, initial_population, times, args=(parameters,)).T

def simulate_infection(derivative_function, initial_population, parameters):
    times = parameters['t']
    zero_social_distance_params = dict(parameters)
    zero_social_distance_params['u_social_distancing'] = 0
    with_distancing = evolve_simulation_over_time(derivative_function, initial_population, times, parameters)
    without_distancing = evolve_simulation_over_time(derivative_function, initial_population, times, zero_social_distance_params)
    return with_distancing, without_distancing

# SEIR model differential equations.
def susceptible_becoming_exposed(social_distancing_effectiveness, t, t_social_distancing_start, beta, susceptible, infected):
    return -(1-social_distancing_effectiveness*(1 if t >= 7 * t_social_distancing_start else 0)/100)*beta * susceptible * infected

def deriv(x, t, params):
    s, e, i, r = x
    u = params['u_social_distancing']
    t_social_distancing = params['t_social_distancing']
    beta = params['beta']
    alpha = params['alpha']
    gamma = params['gamma']
    exposed_becoming_infected = alpha * e
    infected_becoming_recovered = gamma * i
    dsdt = susceptible_becoming_exposed(u, t, t_social_distancing, beta, s, i)
    dedt =  -1 * susceptible_becoming_exposed(u, t, t_social_distancing, beta, s, i) - exposed_becoming_infected
    didt = exposed_becoming_infected - infected_becoming_recovered
    drdt =  infected_becoming_recovered
    return [dsdt, dedt, didt, drdt]

parameters = simulation_parameters()
x_initial = initial_population(parameters)
(with_distancing, without_distancing) = simulate_infection(deriv, x_initial, parameters)
figure = plot_infection_rates(parameters, with_distancing, without_distancing)
save_png(figure, 'covid-19')