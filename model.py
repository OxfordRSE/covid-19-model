#!/usr/bin/env python3

# SEIR model with social distancing
# adapted from https://github.com/jckantor/covid-19

from initialisation import simulation_parameters
from graphing import plot_infection_rates, save_png
from seir_simulation import simulate_infection

parameters = simulation_parameters()
(with_distancing, without_distancing) = simulate_infection(parameters)
figure = plot_infection_rates(parameters, with_distancing, without_distancing)
save_png(figure, 'covid-19')