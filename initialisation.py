import numpy as np

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
