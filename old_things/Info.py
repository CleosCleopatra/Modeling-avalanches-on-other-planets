import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,          # base size
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13
})


labels = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

def plot_gravity_vs_avalanche(x, y):
    """
    data: list of dicts, each with keys:
          'x' (list or array),
          'y' (list or array),
          'label' (string)
    """


    fig, ax = plt.subplots(figsize=(6, 4))

    for i in range(len(x)):
        ax.scatter(x[i],y[i], label=labels[i])
    
    ax.set_xlabel("Gravity (m/s^2)")
    ax.set_ylabel("Mean avalanche area (cells affected)")
    ax.set_title("Effect of gravity on avalanche area")
    ax.legend()
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.grid(True)

    x = np.array(x)
    y = np.array(y)

    x_pad = 0.05 * (x.max() - x.min())
    y_pad = 0.05 * (y.max() - y.min())

    ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
    ax.set_ylim(y.min() - y_pad, y.max() + y_pad)

    ax.legend()

    # Remove empty space around data
    ax.autoscale(tight=True)
    ax.margins(x=0.02, y=0.05)

    plt.tight_layout()
    plt.show()



x = []
y = []

x=[3.7, 8.87, 9.81, 3.71, 24.79, 10.44, 8.69, 11.15]
y=[1.1770134141774196, 1.1043174248773466, 1.0985795937048852, 1.1749988966189466, 1.3852367392673597, 1.1277201424779175, 1.1071972949796094, 1.1285889613702191]
plot_gravity_vs_avalanche(x, y)






import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,          # base size
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13
})


labels = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

def plot_gravity_vs_avalanche(x, y):
    """
    data: list of dicts, each with keys:
          'x' (list or array),
          'y' (list or array),
          'label' (string)
    """


    fig, ax = plt.subplots(figsize=(6, 4))

    for i in range(len(x)):
        ax.scatter(x[i],y[i], label=labels[i])
    
    ax.set_xlabel("Gravity (m/s^2)")
    ax.set_ylabel("Mean avalanche area (cells affected)")
    ax.set_title("Effect of gravity on avalanche area")
    ax.legend()
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.grid(True)

    x = np.array(x)
    y = np.array(y)

    x_pad = 0.05 * (x.max() - x.min())
    y_pad = 0.05 * (y.max() - y.min())

    ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
    ax.set_ylim(y.min() - y_pad, y.max() + y_pad)

    ax.legend()

    # Remove empty space around data
    ax.autoscale(tight=True)
    ax.margins(x=0.02, y=0.05)

    plt.tight_layout()
    plt.show()



x = []
y = []

x=[3.7, 8.87, 9.81, 3.71, 24.79, 10.44, 8.69, 11.15]
y=[1.1770134141774196, 1.1043174248773466, 1.0985795937048852, 1.1749988966189466, 1.3852367392673597, 1.1277201424779175, 1.1071972949796094, 1.1285889613702191]
plot_gravity_vs_avalanche(x, y)



