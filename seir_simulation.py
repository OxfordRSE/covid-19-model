from scipy.integrate import odeint

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

def evolve_simulation_over_time(derivative_function, initial_population, times, parameters):
    return odeint(derivative_function, initial_population, times, args=(parameters,)).T

def simulate_infection(initial_population, parameters):
    times = parameters['t']
    zero_social_distance_params = dict(parameters)
    zero_social_distance_params['u_social_distancing'] = 0
    with_distancing = evolve_simulation_over_time(deriv, initial_population, times, parameters)
    without_distancing = evolve_simulation_over_time(deriv, initial_population, times, zero_social_distance_params)
    return with_distancing, without_distancing

# SEIR model differential equations.
def susceptible_becoming_exposed(social_distancing_effectiveness, t, t_social_distancing_start, beta, susceptible, infected):
    return -(1-social_distancing_effectiveness*(1 if t >= 7 * t_social_distancing_start else 0)/100)*beta * susceptible * infected

