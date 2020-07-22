import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
