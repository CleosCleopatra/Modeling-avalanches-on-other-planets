import numpy as np
from matplotlib import pyplot as plt
import math
from numba import njit

np.random.seed(5)

max_steps = 500
planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
planet_data = [("Mercury", 3.70), ("Venus", 8.87), ("Earth", 9.81), ("Mars", 3.71), ("Jupiter", 24.79), ("Saturn", 10.44), ("Uranus", 8.69), ("Neptune", 11.15)]
alpha = 1.0
beta = 0.5

plt.rcParams.update({
    "font.size": 13,          # base size
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13
})

plt.margins(y=0.05)

def slope_for_gravity(g, static = 6, dynamic = 3):
    f = np.sqrt(g / 9.81)
    static = static * (1 + alpha * (1- f))
    dynamic = dynamic * (1- beta * (1-f))
    static = max(static, 0.5)
    dynamic = max(dynamic, 0.1)
    if dynamic >= static:
        dynamic = static * 0.95
    return static, dynamic

def mass_move_calc(g, g_ref = 9.81, mass_move_max = 2.0, mass_move_min = 0.05):
    #F_friction = mu * m * g
    #mu and m stay the same, the only thing t$g_0hat changes is g
    #Bcs this is just a sandpile model of the system, 
    #that just aims to scale them, not find actual valeus
    val = np.sqrt(g_ref/g)


    return max(min(val, mass_move_max), mass_move_min) #https://www.nature.com/articles/s41526-023-00308-w


def stones_added(terrain, p):
    Ni, Nj = terrain.shape 
    new_rocks = np.random.rand(Ni, Nj) 
    new_rocks_indices = np.where(new_rocks <= p) 
    terrain[new_rocks_indices] += 1 

    return terrain

directions = ((0,1),(0, -1),(1, 0), (-1, 0))
min_runoff = 2.0

@njit
def propagate_avalanche(terrain, i0, j0, n_stones, mass_move, static_loc, dynamic_loc):
    n_topples = 0
    affected = np.zeros_like(terrain)
    affected[i0, j0] = 1

    Ni, Nj = terrain.shape 

    if j0 <= 1 or j0 >= Nj - 1 or i0 <= 1 or i0 >= Ni-1:
        return terrain, 0.0, 0, 0

    runoff_dist = 0

    active = [(i0, j0)]
    active_mask = np.zeros_like(terrain, dtype = np.uint8)
    active_mask[i0, j0] = 1

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

    steps = 0

    while active and steps < max_steps:
        next = []
        for i, j in active:

            if 1 >= j or j >= Nj - 1 or 1 >= i or i >= Ni - 1:
                continue

            thresh = dynamic_loc if active_mask[i, j] else static_loc

            min_h = terrain[i, j]

            for di, dj in directions:
                ni = i + di
                nj = j + dj

                if 0 <= ni < Ni and 0 <= nj < Nj:
                    angle = terrain[i,j] - terrain[ni, nj]
                    #To add some randomness, since real granular flow is kind of stiochastic, especially right at the border
                    if angle > thresh:
                        extra = angle - thresh
                        p_avalanche = min(1.0, mass_move * (angle - dynamic_loc) / dynamic_loc) #Friction
                        if np.random.rand() < p_avalanche:

                            terrain[i, j] -= min(n_stones, terrain[i,j])
                            terrain[ni, nj] += min(n_stones, terrain[i, j])

                            n_topples += 1
                            affected[ni, nj] = 1

                            next.append((ni, nj))
                            active_mask[ni, nj] = 1
                

                            dist = np.sqrt((ni - i0)**2 + (nj - j0)**2) #Since runout is generally defined as the downlope trravel dist
                            if dist > runoff_dist:
                                runoff_dist = dist
        
        avalanche_area = np.sum(affected)
        steps += 1    
        active = next
        
        
    return terrain, runoff_dist, n_topples, avalanche_area
p = 0.02 #Growth probability
target_num_avalanches = 300 
repititions = 100
size_of_terrain = 128
all_runouts = {planet: [] for planet, g in planet_data}
mean_runouts_per_rep = {planet: [] for planet, g in planet_data}
mean_sizes_per_rep = {planet: [] for planet, g in planet_data}
mean_areas_per_rep = {planet: [] for planet, g in planet_data}
all_sizes = {planet: [] for planet, g in planet_data}
all_areas = {planet: [] for planet, g in planet_data}

for planet, g in planet_data:
    mass_move = mass_move_calc(g)
    stones = 1
    static, dynamic = slope_for_gravity(g)
    for rep in range(repititions):
        print(rep)
        terrain = np.zeros([size_of_terrain,size_of_terrain]) #Empty terrain
        runoff_dist_list = [] #Empty list of avalanche sizes
        avalanche_sizes_list = []
        avalanche_areas_list = []

        Ni, Nj = terrain.shape


        num_avalanches = 0

        while num_avalanches < target_num_avalanches:

            terrain = stones_added(terrain, p)

            i0 = np.random.randint(Ni)
            j0 = np.random.randint(Nj)

            terrain, runoff, n_topples, avalanche_area = propagate_avalanche(terrain, i0, j0, stones, mass_move, static, dynamic)
            if n_topples > 0:
                runoff_dist_list.append(runoff)
                num_avalanches += 1
                avalanche_sizes_list.append(n_topples)
                avalanche_areas_list.append(avalanche_area)

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

mean_sizes = []
err_sizes = []

mean_areas = []
err_areas = []

gravities = []
mean_runouts = []
for planet, g in planet_data:
    gravities.append(g)

    #Runout
    rep_means = np.array(mean_runouts_per_rep[planet])
    mean_runouts.append(np.median(rep_means))
    err_runouts.append(rep_means.std(ddof=1) / np.sqrt(len(rep_means)))

    #Sizes
    sizes = np.array(mean_sizes_per_rep[planet])
    mean_sizes.append(np.median(sizes))
    err_sizes.append(sizes.std(ddof=1) / np.sqrt(len(sizes)))

    #Area
    areas = np.array(mean_areas_per_rep[planet])
    mean_areas.append(np.median(areas))
    err_areas.append(areas.std(ddof=1) / np.sqrt(len(areas)))

fig, ax = plt.subplots()
for i, (planet, g) in enumerate(planet_data):
    print(f"{planet}: g = {gravities}, mean_runouts = {mean_runouts}")
    plt.errorbar(
        gravities[i],
        mean_runouts[i],
        yerr = err_runouts[i],
        fmt = 'o',
        color=planet_colours[planet],
        capsize = 4,
        markersize=8,
        elinewidth=1.5,
        label=planet
    )

ax.set_xlabel("Gravity (m/s^2)")
ax.set_ylabel("Mean runout distance (grid units)")
#plt.xscale("log")
#plt.yscale("log")
ax.set_facecolor("none")
fig.patch.set_alpha(0)
ax.set_title("Effect of gravity on avalanche runout")
ax.grid(True)
ax.set_ylim(1.05, 1.45)
ax.margins(y=0.05)
ax.legend()

plt.tight_layout()
plt.show()



plt.figure()
for i, (planet, g) in enumerate(planet_data):
    print(f"{planet}: g = {gravities}, mean_size = {mean_sizes}")
    plt.errorbar(
        gravities[i],
        mean_sizes[i],
        yerr=err_sizes[i],
        fmt='o',
        color=planet_colours[planet],
        capsize=4,
        markersize=8,
        elinewidth=1.5,
        label=planet
)
plt.xlabel("Gravity (m/s^2)")
plt.ylabel("Mean avalanche size (number of topples)")
plt.title("Effect of gravity on avalanche size")
plt.grid(True)
#plt.xscale("log")
#plt.yscale("log")
plt.tight_layout()
ax.set_facecolor("none")
fig.patch.set_alpha(0)
plt.legend()
plt.show()

plt.figure()
import numpy as np
from matplotlib import pyplot as plt
import math
from numba import njit

np.random.seed(5)

max_steps = 500
planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
planet_data = [("Mercury", 3.70), ("Venus", 8.87), ("Earth", 9.81), ("Mars", 3.71), ("Jupiter", 24.79), ("Saturn", 10.44), ("Uranus", 8.69), ("Neptune", 11.15)]
alpha = 1.0
beta = 0.5

plt.rcParams.update({
    "font.size": 13,          # base size
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13
})



plt.margins(y=0.05)


def slope_for_gravity(g, static = 6, dynamic = 3):
    f = np.sqrt(g / 9.81)

    static = static * (1 + alpha * (1- f))

    dynamic = dynamic * (1- beta * (1-f))

    static = max(static, 0.5)
    dynamic = max(dynamic, 0.1)
    return static, dynamic

def mass_move_calc(g, g_ref = 9.81, mass_move_max = 2.0, mass_move_min = 0.05):
    #F_friction = mu * m * g
    #mu and m stay the same, the only thing t$g_0hat changes is g
    #Bcs this is just a sandpile model of the system, 
    #that just aims to scale them, not find actual valeus
    val = np.sqrt(g_ref/g)


    return max(min(val, mass_move_max), mass_move_min) #https://www.nature.com/articles/s41526-023-00308-w


#More stones topple if the gravity is lower I think, but I got to find a source?
@njit
def stones_per_topple(g):
    orig_n_stones = max(1, int(2 * 9.81/g))
    #friction_stones = max (1, int(orig_n_stones*fric))
    return 1

def stones_added(terrain, p):
    """
    Function to grow new stones in the terrain.
    
    Parameters
    ==========
    terrain: 2-dimensional array
    p: Probability for a tree to be generated in an empty cell
    """

    Ni, Nj = terrain.shape #Dimensions of terrain

    new_rocks = np.random.rand(Ni, Nj) #Random number in each place to calc whether stone drops

    new_rocks_indices = np.where(new_rocks <= p) #The indices at which new stones drop
    terrain[new_rocks_indices] += 1 #Add stones

    return terrain

directions = ((0,1),(0, -1),(1, 0), (-1, 0))
min_runoff = 2.0

@njit
def propagate_avalanche(terrain, i0, j0, n_stones, mass_move, static_loc, dynamic_loc):
    """
    Function to propagate the avalanche on a terrain.
    
    Parameters
    ==========
    terrain: 2-dimensional array
    i0: First index of the cell where the avalanche occurs
    j0: Second index of the cell where the avalanche occurs
    """
    #n_stones = grav[0]
    #fric = grav[1]
    n_topples = 0
    affected = np.zeros_like(terrain)
    affected[i0, j0] = 1

    Ni, Nj = terrain.shape #Dimensions of the terrain

    if j0 <= 1 or j0 >= Nj - 1 or i0 <= 1 or i0 >= Ni-1:
        return terrain, 0.0, 0, 0


    runoff_dist = 0

    active = [(i0, j0)]
    active_mask = np.zeros_like(terrain, dtype = np.uint8)
    active_mask[i0, j0] = 1

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

    steps = 0

    while active and steps < max_steps:
        next = []
        for i, j in active:

            if 1 >= j or j >= Nj - 1 or 1 >= i or i >= Ni - 1:
                continue

            thresh = dynamic_loc if active_mask[i, j] else static_loc

            for di, dj in directions:
                ni = i + di
                nj = j + dj

                if 0 <= ni < Ni and 0 <= nj < Nj:
                    angle = terrain[i,j] - terrain[ni, nj]

                    #To add some randomness, since real granular flow is kind of stiochastic, especially right at the border

                    if angle > thresh:
                        extra = angle - thresh
                        p_avalanche = min(1.0, mass_move * (angle - dynamic_loc) / dynamic_loc) #Friction
                        if np.random.rand() < p_avalanche:
                            moved = min(n_stones, terrain[i, j])
                            terrain[i, j] -= moved
                            terrain[ni, nj] += moved

                            n_topples += 1
                            affected[ni, nj] = 1

                            next.append((ni, nj))
                            active_mask[ni, nj] = 1
                

                            dist = np.sqrt((ni - i0)**2 + (nj - j0)**2) #Since runout is generally defined as the downlope trravel dist
                            if dist > runoff_dist:
                                runoff_dist = dist
        
        avalanche_area = np.sum(affected)
        steps += 1    
        active = next
        
        
    return terrain, runoff_dist, n_topples, avalanche_area


p = 0.02 #Growth probability




target_num_avalanches = 300 
repititions = 100
size_of_terrain = 128

all_runouts = {planet: [] for planet, g in planet_data}
mean_runouts_per_rep = {planet: [] for planet, g in planet_data}
mean_sizes_per_rep = {planet: [] for planet, g in planet_data}
mean_areas_per_rep = {planet: [] for planet, g in planet_data}

all_runouts = {planet: [] for planet, g in planet_data}
all_sizes = {planet: [] for planet, g, in planet_data}
all_areas = {planet: [] for planet, g in planet_data}

for planet, g in planet_data:
    mass_move = mass_move_calc(g)
    stones = stones_per_topple(g)
    static, dynamic = slope_for_gravity(g)
    
    print(planet)
    #gravity_factors = [stones, mu_fric]

    for rep in range(repititions):
        print(rep)
        terrain = np.zeros([size_of_terrain,size_of_terrain]) #Empty terrain
        runoff_dist_list = [] #Empty list of avalanche sizes
        avalanche_sizes_list = []
        avalanche_areas_list = []

        Ni, Nj = terrain.shape


        num_avalanches = 0

        while num_avalanches < target_num_avalanches:
            #print(planet, rep, num_avalanches)

            terrain = stones_added(terrain, p)

            i0 = np.random.randint(Ni)
            j0 = np.random.randint(Nj)

            terrain, runoff, n_topples, avalanche_area = propagate_avalanche(terrain, i0, j0, stones, mass_move, static, dynamic)
            if n_topples > 0:
                runoff_dist_list.append(runoff)
                num_avalanches += 1
                avalanche_sizes_list.append(n_topples)
                avalanche_areas_list.append(avalanche_area)

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

gravities = []
mean_runouts = []

from scipy.stats import skew, kurtosis

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
Threshold_area = 5
Threshold_topple = 5
for planet, g in planet_data:
    gravities.append(g)

    #Runout
    rep_means = np.array(mean_runouts_per_rep[planet])
    mean_runouts.append(np.median(rep_means))
    err_runouts.append(rep_means.std(ddof=1))

    #Median
    all_runout_loc = all_runouts[planet]
    median_runouts.append(np.median(all_runout_loc))
    err_median_runouts.append(all_runout_loc.std(ddof=1))

    #Skew
    runout_skew.append(skew(all_runout_loc))
    #Kurtosis
    runout_kurtosis.append(kurtosis(all_runout_loc))
    #Fraction
    runout_fraction.append(all_runout_loc>Threshold_runout)


    #Sizes
    sizes = np.array(mean_sizes_per_rep[planet])
    mean_sizes.append(np.median(sizes))
    err_sizes.append(sizes.std(ddof=1))

    #Median
    all_sizes_loc = all_sizes[planet]
    median_runouts.append(np.median(all_sizes_loc))
    err_median_runouts.append(all_sizes_loc.std(ddof=1))

    #Skew
    size_skew.append(skew(all_sizes_loc))
    #kurtosis
    size_kurtosis.append(kurtosis(all_sizes_loc))
    #Fractin
    size_fraction.append(all_sizes_loc>Threshold_topple)


    #Area
    areas = np.array(mean_areas_per_rep[planet])
    mean_areas.append(np.median(areas))
    err_areas.append(areas.std(ddof=1))

    #Median
    all_areas_loc = all_areas[planet]
    median_runouts.append(np.median(all_sizes_loc))
    err_median_runouts.append(all_sizes_loc.std(ddof=1))

    #Skew
    area_skew.append(skew(all_areas_loc))
    #Kurtosis
    area_kurtosis.append(kurtosis(all_areas_loc))
    #Fraction
    area_fraction.append(all_areas_loc>Threshold_area)



def graph(yname, name, y, error, ymin, ymax):
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
    ax.legend()
    plt.tight_layout()
    plt.show()

def graph_without_error(yname, name, y):
    fig, ax = plt.subplots()
    for i, (planet, g) in enumerate(planet_data):
        print(f"{planet}: g = {g}, {name} = {y[i]}")   
        plt(
            gravities[i], 
            y[i],
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
    ax.margins(y=0.05)
    ax.legend()
    plt.tight_layout()
    plt.show()

#Mean runout
graph("Mean runout distance (grid units)", "Effects of gravity on avalanche runout", mean_runouts, err_runouts, 1.05, 1.45)

#Median runout
graph("Median runout distance (grid units)", "Effects of gravity on avalanche runout", median_runouts, err_median_runouts, 1.05, 1.45)

#skewness runout
graph_without_error("Skewness for all runouts for planet", "Skewness of runout for different planets", runout_skew)

#kurtosis runout
graph_without_error("Kurtosis for all runouts for planet", "Kurtosis of runout for different planets", runout_kurtosis)

#Fraction runout
graph_without_error("Fraction of large avalanche (runouts>3 grid units)", "Fraction of large avalanches for different planets", runout_fraction)




#Mean Area
graph("Mean area (grid units)", "Effects of gravity on avalanche area", mean_areas, err_areas, 1.05, 1.45)

#Median area
graph("Median area (grid units)", "Effects of gravity on avalanche area", median_area, err_median_area, 1.05, 1.45)

#skewness area
graph_without_error("Skewness for all areas for planet", "Skewness of area for different planets", area_skew)

#Kurtosis area
graph_without_error("Kurtosis for all areas for planet", "Kurtosis of area for different planets", area_kurtosis)

#Fractijon area
graph_without_error("Fraction of large avalanche (Area > 5 cells)", "Fraction of large avalanches for different planets", area_fraction)





#Mean num of topples
graph("Mean size (number of topples)", "Effects of gravity on mean size", mean_sizes, err_sizes, 1.05, 1.45)

#Median num of topples
graph("Median size (number of topples)", "Effects of gravity on median size", median_size, err_median_size, 1.05, 1.45)

#skewness num of topples
graph_without_error("Skewness for all sizes for planet", "Skewness of sizes for different planets", size_skew)

#Kurtosis size
graph_without_error("Kurtosis for all sizes for planet", "Kurtosis of sizes for different planets", size_kurtosis)

#Fraction
graph_without_error("Fractiion of large avalanches (num topples > 5)", "Fraction of large avalanches for different planets", size_fraction)

