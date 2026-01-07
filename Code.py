import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from scipy.stats import skew, kurtosis

np.random.seed(5) #Reproducibilitty

#Parameters
max_steps = 500
directions = ((0,1),(0, -1),(1, 0), (-1, 0))
min_runoff = 2.0
p = 0.02 
target_num_avalanches = 300
repititions = 100
size_of_terrain = 128
alpha = 1.0
beta = 0.5

#Planet names and gravities
planets = ["Mercury", "Mars", "Uranus", "Venus", "Earth", "Saturn", "Neptune", "Jupiter"]
planet_data = [("Mercury", 3.70), ("Mars", 3.71),  ("Uranus", 8.69), ("Venus", 8.87), ("Earth", 9.81), ("Saturn", 10.44),  ("Neptune", 11.15), ("Jupiter", 24.79)]

#Increase font size for the fonts
plt.rcParams.update({
    "font.size": 13,          
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13
})

plt.margins(y=0.05)



def slope_for_gravity(g, static = 6, dynamic = 3):
    """
    Computations of static and dynamic slope threshold.

    Args:
        g: gravity of the planet
        static: static slope for earth
        dynamic: dynamic slope for earth
    
    Returns:
        static: static slope for planet
        dynamic: dynamic slope for earth
    """ 
    f = np.sqrt(g / 9.81)

    static = static * (1 + alpha * (1- f))
    dynamic = dynamic * (1- beta * (1-f))
    static = max(static, 4)
    dynamic = max(dynamic, 1)

    #Ensure that dynamic angle does not exceed static
    if dynamic >= static:
        dynamic = static * 0.95

    return static, dynamic


def mobility_value_calc(g, g_ref = 9.81, mobility_value_max = 2.0, mobility_value_min = 0.05):
    """
    Computations of mobility factor, which representes the effect of gravity on the likelihood of an avalanceh continuing once started

    Args:
        g: gravity of the planet
        g_ref: Earths gravity
        mobility_value_max: maximum mobility value to avoid extremes
        mobility_value_min: minimum mobility value to avaoid extremes
    
    Returns:
        mobility_returned: mobility value for planet
    """

    val = np.sqrt(g_ref/g) 
    mobility_returned = max(min(val, mobility_value_max), mobility_value_min) 


    return mobility_returned


def stones_added(terrain, p):
    """
    Stones added to planet

    Args:
        terrain: current terrain
        p: probability of a stone being added

    Returns:
        terrain: New terrain with stones added
    """
    Ni, Nj = terrain.shape 
    new_rocks = np.random.rand(Ni, Nj) 
    new_rocks_indices = np.where(new_rocks <= p) 
    terrain[new_rocks_indices] += 1 #Adds one particle to each of some random number of cells 

    return terrain

@njit
def propagate_avalanche(terrain, i0, j0, mobility_value, static_loc, dynamic_loc, n_stones=1):
    """
    Propagation of avalanche

    Args:
        terrain: Current terrain
        i0: x location for start of avalanche
        j0: y location for start of avalanche
        mobility value: mobility value to calculate likelihood of avalanche continuing
        static_loc: static angle for planet
        dynamic_loc: dynamic angle for planet
        n_stones: number of stones that fall during avalanche
    
    Returns:
        terrain: New terrain
        runoff_dist: max runoff distance for avalanche
        n_topples: number of topples
        avalanche_area: Area of avalanche
    """
    n_topples = 0
    affected = np.zeros_like(terrain)
    affected[i0, j0] = 1

    slope_excess = 0.0 #Stat

    Ni, Nj = terrain.shape 

    #Cancels the avalanche if its too close to the edge of the terrain
    if j0 <= 1 or j0 >= Nj - 1 or i0 <= 1 or i0 >= Ni-1:
        return terrain, 0.0, 0, 0, 0.0, 0

    runoff_dist = 0

    #List and mask showing where there are currently avalanches
    active = [(i0, j0)]
    active_mask = np.zeros_like(terrain, dtype = np.uint8)
    active_mask[i0, j0] = 1

    #Looks at the angle in each direction of the active cell
    for di, dj in directions: 
        ni = i0 + di
        nj = j0 + dj
        if 0 <= ni < Ni and 0 <= nj < Nj:
            angle = terrain[i0, j0] - terrain[ni, nj]
            if angle > static_loc:
                terrain[i0, j0] -= min(n_stones, terrain[i0, j0])
                terrain[ni, nj] += min(n_stones, terrain[i0, j0])
                active.append((ni, nj))

                runoff_dist = max(runoff_dist, np.sqrt((ni - i0)**2 + (nj - j0)**2))

                excess = angle - static_loc #
                if excess > slope_excess: 
                    slope_excess = excess

    steps = 0

    #Loop that continues until the system stabilises or max steps have been reached
    while active and steps < max_steps:
        next = []
        for i, j in active:

            #Cancels the avalanche if its too close to the edge
            if 1 >= j or j >= Nj - 1 or 1 >= i or i >= Ni - 1:
                continue

            thresh = dynamic_loc if active_mask[i, j] else static_loc

            #goes through each direction around the active cell
            for di, dj in directions:
                ni = i + di
                nj = j + dj

                if 0 <= ni < Ni and 0 <= nj < Nj:
                    angle = terrain[i,j] - terrain[ni, nj]
                    if angle > thresh:
                        #calculate the probability of avalanche continuing, based on the mobility value (which is based on friction),
                        #as well as how mcuh steeper the slopw is than required
                        p_avalanche = min(1.0, mobility_value * (angle - dynamic_loc) / dynamic_loc) 
                        if np.random.rand() < p_avalanche:
                            moved = min(n_stones, terrain[i, j]) #n_stones are moved if cell has enough particles for that
                            terrain[i, j] -= moved
                            terrain[ni, nj] += moved

                            n_topples += 1
                            affected[ni, nj] = 1

                            next.append((ni, nj))
                            active_mask[ni, nj] = 1
                

                            dist = np.sqrt((ni - i0)**2 + (nj - j0)**2)
                            if dist > runoff_dist:
                                runoff_dist = dist
        
        avalanche_area = np.sum(affected)
        steps += 1    
        active = next
        
        
    return terrain, runoff_dist, n_topples, avalanche_area, slope_excess, steps

all_runouts = {planet: [] for planet, g in planet_data}
mean_runouts_per_rep = {planet: [] for planet, g in planet_data}
mean_sizes_per_rep = {planet: [] for planet, g in planet_data}
mean_areas_per_rep = {planet: [] for planet, g in planet_data}
all_sizes = {planet: [] for planet, g in planet_data}
all_areas = {planet: [] for planet, g in planet_data}
all_slope_excess = {planet: [] for planet, g in planet_data}
all_steps = {planet: [] for planet, g in planet_data}
single_topple_fraction = {}

#Main simulation loop
for planet, g in planet_data:
    mob_val = mobility_value_calc(g)
    stones = 1
    static, dynamic = slope_for_gravity(g)
    for rep in range(repititions):
        terrain = np.zeros([size_of_terrain,size_of_terrain]) #Empty terrain
        runoff_dist_list = [] 
        avalanche_sizes_list = []
        avalanche_areas_list = []

        Ni, Nj = terrain.shape

        num_avalanches = 0

        #Loop to add more particles and start avalanches
        #Repeats until target number of avalanches has been added
        while num_avalanches < target_num_avalanches:

            terrain = stones_added(terrain, p)

            i0 = np.random.randint(Ni)
            j0 = np.random.randint(Nj)

            terrain, runoff, n_topples, avalanche_area, excess_slope, steps = propagate_avalanche(terrain, i0, j0, mob_val, static, dynamic)
            if n_topples > 0:
                runoff_dist_list.append(runoff)
                num_avalanches += 1
                avalanche_sizes_list.append(n_topples)
                avalanche_areas_list.append(avalanche_area)
                all_slope_excess[planet].append(excess_slope)
                all_steps[planet].append(steps)

        if runoff_dist_list:
            mean_runouts_per_rep[planet].append(np.mean(runoff_dist_list))
            mean_sizes_per_rep[planet].append(np.mean(avalanche_sizes_list))
            mean_areas_per_rep[planet].append(np.mean(avalanche_areas_list))

            all_runouts[planet].extend(runoff_dist_list)
            all_sizes[planet].extend(avalanche_sizes_list)
            all_areas[planet].extend(avalanche_areas_list)

        else:
            mean_runouts_per_rep[planet].append(0.0)
            mean_sizes_per_rep[planet].append(0.0)
            mean_areas_per_rep[planet].append(0.0)

            all_runouts[planet].extend(0.0)
            all_sizes[planet].extend(0.0)
            all_areas[planet].extend(0.0)
        mean_runout = np.mean(runoff_dist_list)
        median_runout = np.median(runoff_dist_list)
        std_runout = np.std(runoff_dist_list)
        max_runout = np.max(runoff_dist_list)
        print(f"{planet} rep {rep}: mean={mean_runout:.2f}, median = {median_runout}, std={std_runout:.2f}, max = {max_runout:.2f}")


gravities = [g for planet, g in planet_data]

planet_colours = {
    "Mercury": "brown",
    "Venus" : "orange",
    "Earth": "green",
    "Mars": "red",
    "Jupiter": "blue",
    "Saturn": "purple",
    "Uranus": "cyan",
    "Neptune": "black"
}


mean_runouts = []
err_runouts = []
median_runouts = []
err_median_runouts = []

mean_sizes = []
err_sizes = []
median_size = []
err_median_size = []

mean_areas = []
err_areas = []
median_area = []
err_median_area = []

runout_skew = []
size_skew = []
area_skew = []

runout_kurtosis = []
size_kurtosis = []
area_kurtosis = []

runout_fraction = []
size_fraction = []
area_fraction = []

Threshold_runout = 3
Threshold_area = 4
Threshold_topple = 4

#lists different values for each planet for graphing
for planet, g in planet_data:

    #Single topple fraction
    sizes = np.array(all_sizes[planet])
    single_topple_fraction[planet] = np.mean(sizes==1)

    #Runout
    rep_means = np.array(mean_runouts_per_rep[planet])
    mean_runouts.append(np.median(rep_means))
    err_runouts.append(np.std(rep_means))

    #Median
    all_runout_loc = all_runouts[planet]
    median = np.median(all_runout_loc)
    median_runouts.append(median)
    err_lower_runout = median - np.percentile(all_runout_loc, 25)
    err_higher_runout = np.percentile(all_runout_loc, 75) - median
    err_median_runouts.append([[err_lower_runout], [err_higher_runout]])

    #Skew
    runout_skew.append(skew(all_runout_loc))
    #Kurtosis
    runout_kurtosis.append(kurtosis(all_runout_loc))
    #Fraction
    all_runout_loc = np.array(all_runout_loc)
    runout_fraction.append(np.mean(all_runout_loc > Threshold_runout)*100)


    #Sizes
    sizes = np.array(mean_sizes_per_rep[planet])
    mean_sizes.append(np.median(sizes))
    err_sizes.append(np.std(sizes))

    #Median
    all_sizes_loc = all_sizes[planet]
    median_size_calc = np.median(all_sizes_loc)
    median_size.append(median_size_calc)
    err_median_size.append([[median_size_calc-np.percentile(all_sizes_loc, 25)], [np.percentile(all_sizes_loc, 75) - median_size_calc]])

    #Skew
    size_skew.append(skew(all_sizes_loc))
    #kurtosis
    size_kurtosis.append(kurtosis(all_sizes_loc))
    #Fractin
    all_sizes_loc = np.array(all_sizes_loc)
    size_fraction.append(np.mean(all_sizes_loc>Threshold_topple)*100)


    #Area
    areas = np.array(mean_areas_per_rep[planet])
    mean_areas.append(np.median(areas))
    err_areas.append(np.std(areas))

    #Median
    all_areas_loc = all_areas[planet]
    median_area_calc = np.median(all_areas_loc)
    median_area.append(median_area_calc)
    err_median_area.append([[median_area_calc - np.percentile(all_areas_loc, 25)], [np.percentile(all_areas_loc, 75) - median_area_calc]])

    #Skew
    area_skew.append(skew(all_areas_loc))
    #Kurtosis
    area_kurtosis.append(kurtosis(all_areas_loc))
    #Fraction
    area_fraction.append(np.mean(np.array(all_areas_loc)>Threshold_area)*100)



def graph(yname, name, y, error, ymin, ymax, log):
    """
    Function to generate graph with error bars

    Args:
        yname: Name of y variable
        name: Name of graph
        y: list of what is being graphed
        error: list of error bars
        ymin, ymax: min and max y value in graph
    """
    fig, ax = plt.subplots()
    for i, (planet, g) in enumerate(planet_data):
        print(f"{planet}: g = {g}, {name} = {y[i]}")   
        plt.errorbar(
            gravities[i], 
            y[i],
            yerr = error[i],
            fmt = 'o',
            color = planet_colours[planet],
            capsize=4,
            markersize = 8,
            elinewidth=1.5,
            label=planet
        )
    ax.set_xlabel("Gravity (m/s^2)")
    ax.set_ylabel(yname)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.set_title(name)
    ax.grid(True)
    
    ax.set_ylim(ymin, ymax)
    ax.margins(y=0.05)
    ax.legend(loc="best")
    if log:
        plt.yscale('log')
    plt.tight_layout()
    plt.show()

def graph_without_error(yname, name, y, log):
    """
    Function to generate graph without errorbars

    Args: 
        yname: name of y variable
        name: name of graph
        y: List of y variable values
    """
    fig, ax = plt.subplots()
    print(y)
    for i, (planet, g) in enumerate(planet_data):
        print(f"{planet}: g = {g}, {name} = {y[i]}")   
        plt.scatter(
            gravities[i], 
            y[i],
            s = 30,
            c = planet_colours[planet],
            marker = 'o',
            label=planet
        )
    ax.set_xlabel("Gravity (m/s^2)")
    ax.set_ylabel(yname)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.set_title(name)
    ax.grid(True)
    ax.margins(y=0.05)
    ax.legend()
    if log:
        plt.yscale('log')
    plt.tight_layout()
    plt.show()


#Mean runout
graph("Mean runout distance (grid units)", "Effects of gravity on avalanche runout", mean_runouts, err_runouts, 1.05, 1.30, False)

#Median runout
graph("Median runout distance (grid units)", "Effects of gravity on avalanche runout", median_runouts, err_median_runouts, 0.95, 2.50, False) 

#skewness runout
graph_without_error("Skewness for all runouts for planet", "Skewness of runout for different planets", runout_skew, False)

#kurtosis runout
graph_without_error("Kurtosis for all runouts for planet", "Kurtosis of runout for different planets", runout_kurtosis, False)

#Fraction runout
graph_without_error("Fraction of large avalanche (runouts>3 grid units)", "Fraction of large avalanches for different planets", runout_fraction, True)




#Mean Area
graph("Mean area (grid units)", "Effects of gravity on avalanche area", mean_areas, err_areas, 2.25, 3.15, False)

#Median area
graph("Median area (grid units)", "Effects of gravity on avalanche area", median_area, err_median_area, 1.75, 3.5, False)

#skewness area
graph_without_error("Skewness for all areas for planet", "Skewness of area for different planets", area_skew, False)

#Kurtosis area
graph_without_error("Kurtosis for all areas for planet", "Kurtosis of area for different planets", area_kurtosis, False)

#Fractijon area
graph_without_error("Fraction of large avalanche (Area > 5 cells)", "Fraction of large avalanches for different planets", area_fraction, True)


from collections import Counter
import pandas as pd



def plot_size_distribution_linear(all_vars, xlabel, title, bins_loc):
    fig, ax = plt.subplots(4, 2, sharex=True, sharey=True, gridspec_kw = {'wspace':0, 'hspace':0})
    i = 0
    tables = {}

    for planet, g in planet_data:
        row = i//2
        column = i%2

        vars = np.array(all_vars[planet])
        #vars = vars[vars > 0]

        counts = Counter(vars)
        
        df = pd.DataFrame(
            sorted(counts.items()),
            columns=["Size", "Count"]
        )

        tables[planet] = df

        print(planet)
        print(df.to_string(index=False))


        ax[row, column].hist(
            vars,
            bins_loc,
            log=True
        )

        ax[row, column].text(
            0.95, 0.95, 
            f"{planet} \n $g={g}$",
            transform=ax[row, column].transAxes,
            va='top',
            ha='right',
            fontsize=11
        )

        ax[row, column].yaxis.grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.7,
            alpha=0.6
        )
        i+=1
    fig.supxlabel(xlabel)
    fig.supylabel("Probability density")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


plot_size_distribution_linear(all_sizes, "Avalanche area (num of affected cells)", "Avalanche area distribution for different planets", np.arange(1,21))
plot_size_distribution_linear(all_runouts, "Avalanche runout distance (euclidian distance in grid units)", "Avalanche runout distribution for different planets", [1, np.sqrt(2), 2, np.sqrt(5), np.sqrt(8), 3, np.sqrt(10), np.sqrt(13), 4, np.sqrt(17), np.sqrt(18), np.sqrt(20), 5])