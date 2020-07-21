#!/usr/bin/env python3

# SEIR model with social distancing
# adapted from https://github.com/jckantor/covid-19

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns

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

def simulate_infection(derivative_function, initial_population, parameters):
    times = parameters['t']
    zero_social_distance_params = dict(parameters)
    zero_social_distance_params['u_social_distancing'] = 0
    with_distancing = odeint(derivative_function, initial_population, times, args=(parameters,)).T
    without_distancing = odeint(derivative_function, initial_population, times, args=(zero_social_distance_params,)).T
    return with_distancing, without_distancing

def plot_infection_rates(parameters, with_distancing, without_distancing):
    # plot the data
    N = parameters['N']
    t = parameters['t']
    u_social_distancing = parameters['u_social_distancing']

    fig = plt.figure(figsize=(12, 10))
    ax = [fig.add_subplot(311, axisbelow=True), 
        fig.add_subplot(312)]

    pal = sns.color_palette()
    (s,e,i,r) = with_distancing
    ax[0].stackplot(t/7, N*s, N*e, N*i, N*r, colors=pal, alpha=0.6)
    ax[0].set_title('Susceptible and Recovered Populations with {0:3.0f}% Effective Social Distancing'.format(u_social_distancing))
    ax[0].set_xlabel('Weeks following Initial Campus Exposure')
    ax[0].set_xlim(0, t[-1]/7)
    ax[0].set_ylim(0, N)
    ax[0].legend([
        'Susceptible', 
        'Exposed/no symptoms', 
        'Infectious/symptomatic',
        'Recovered'], 
        loc='best')

    t_social_distancing = parameters['t_social_distancing']
    R0 = parameters['R0']
    ax[0].plot(np.array([t_social_distancing, t_social_distancing]), ax[0].get_ylim(), 'r', lw=3)
    ax[0].plot(np.array([0, t[-1]])/7, [N/R0, N/R0], lw=3, label='herd immunity')
    ax[0].annotate("Start of social distancing",
        (t_social_distancing, 0), (t_social_distancing + 1.5, N/10),
        arrowprops=dict(arrowstyle='->'))
    ax[0].annotate("Herd Immunity without social distancing",
        (t[-1]/7, N/R0), (t[-1]/7 - 8, N/R0 - N/5),
        arrowprops=dict(arrowstyle='->'))

    (_, e0, i0, _) = without_distancing
    ax[1].stackplot(t/7, N*i0,N*e0, colors=pal[2:0:-1], alpha=0.5)
    ax[1].stackplot(t/7, N*i, N*e, colors=pal[2:0:-1], alpha=0.5)
    ax[1].set_title('Infected Population with no Social Distancing and with {0:3.0f}% Effective Social Distancing'.format(u_social_distancing))
    ax[1].set_xlim(0, t[-1]/7)
    ax[1].set_ylim(0, max(0.3*N, 1.05*max(N*(e + i))))
    ax[1].set_xlabel('Weeks following Initial Campus Exposure')
    ax[1].legend([
        'Infective/Symptomatic', 
        'Exposed/Not Sympotomatic'],
        loc='upper right')
    ax[1].plot(np.array([t_social_distancing, t_social_distancing]), ax[0].get_ylim(), 'r', lw=3)

    y0 = N*(e0 + i0)
    k0 = np.argmax(y0)
    ax[1].annotate("No social distancing", (t[k0]/7, y0[k0] + 100))

    y = N*(e + i)
    k = np.argmax(y)
    ax[1].annotate("With {0:3.0f}% effective social distancing ".format(u_social_distancing), (t[k]/7, y[k] + 100))

    for a in ax:
        a.xaxis.set_major_locator(plt.MultipleLocator(5))
        a.xaxis.set_minor_locator(plt.MultipleLocator(1))
        a.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
        a.grid(True)

    return plt

def save_png(plt, filename):
    plt.tight_layout()
    plt.savefig(filename)

# SEIR model differential equations.
def susceptible_population_gradient(social_distancing_effectiveness, t, t_social_distancing_start, beta, susceptible, infected):
    return -(1-social_distancing_effectiveness*(1 if t >= 7 * t_social_distancing_start else 0)/100)*beta * susceptible * infected

def deriv(x, t, params):
    s, e, i, r = x
    u = params['u_social_distancing']
    t_social_distancing = params['t_social_distancing']
    beta = params['beta']
    alpha = params['alpha']
    gamma = params['gamma']
    dsdt = susceptible_population_gradient(u, t, t_social_distancing, beta, s, i)
    dedt =  -1 * susceptible_population_gradient(u, t, t_social_distancing, beta, s, i) - alpha * e
    didt = alpha * e - gamma * i
    drdt =  gamma * i
    return [dsdt, dedt, didt, drdt]

parameters = simulation_parameters()
x_initial = initial_population(parameters)
(with_distancing, without_distancing) = simulate_infection(deriv, x_initial, parameters)
figure = plot_infection_rates(parameters, with_distancing, without_distancing)
save_png(figure, 'covid-19')