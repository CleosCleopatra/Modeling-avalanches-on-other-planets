import numpy as np
from matplotlib import pyplot as plt
import math

np.random.seed(5)

static_earth = 47.0
dynamic_earth = 43.0
planet_data = [("Mercury", 3.70), ("Venus", 8.87), ("Earth", 9.81), ("Mars", 3.71), ("Jupiter", 24.79), ("Saturn", 10.44), ("Uranus", 8.69), ("Neptune", 11.15)]

def slope_for_gravity(g):
    g_frac = (9.81 - g) / 9.81
    static_angle = static_earth + 5 * g_frac
    dynamic_angle = dynamic_earth - 10 *g_frac
    static_slope = math.tan(math.radians(static_angle))
    dynamic_slope = math.tan(math.radians(dynamic_angle))
    return static_slope, dynamic_slope


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




#We are going to presume that the terrain tilts downwards to the right with a constant angle, to avoid having to model for a bunch of different angles

def propagate_avalanche(terrain, i0, j0, g):
    """
    Function to propagate the avalanche on a terrain.
    
    Parameters
    ==========
    terrain: 2-dimensional array
    i0: First index of the cell where the avalanche occurs
    j0: Second index of the cell where the avalanche occurs
    """
    static, dynamic = slope_for_gravity(g)

    Ni, Nj = terrain.shape #Dimensions of the terrain

    if j0 >= Nj - 1:
        return terrain, 0.0

    start = [i0, j0]
    runoff_dist = 0

    current_height = terrain[i0, j0]

    #The avalnche moves to the right, either straight to the right, or to right up/right down
    avalanche  = False
    direction_with_less = []
    for di in [-1, 0, 1]:
        ni = i0 + di
        nj = j0 + 1
        if 0 <= ni < Ni and nj < Nj:
            angle = terrain[i0, j0] - terrain[ni, nj]
        
        if not avalanche and angle <= static:
            terrian = terrain
            runoff_dist = 0.0
        else:
            direction_with_less.append([ni, nj])
            avalanche = True
    
    if not avalanche:
        return terrain, runoff_dist
    
    for directions in direction_with_less:
        terrain[i0, j0] -= 1
        terrain[directions] += 1

        active_i.append(directions[0])
        active_j.append(directions[1])


    while active_i:
        next_i = []
        next_j = []
        for n in np.arange(len(active_i)):
            i = active_i[n]
            j = active_j[n]

            if j >= Nj -1: 
                continue #If we cant go further

            
            current_height = terrain[i, j]

            neighbour_heights_list = [terrain[i, j+1], terrain[i-1, j+1], terrain[i+1, j+1]]
            #neighbour_height = terrain[i, j+1]
 

            for ni, nj in neighbour_heights_list:
                angle = terrain[i,j] - terrain[ni, nj]
                if angle > dynamic:
                    terrain[i, j] -= 1
                    terrain[ni, nj] += 1

                    next_i.append(ni)
                    next_j.append(nj)

                    dist = np.sqrt((ni - start[0])**2 + (nj - start[1])**2)
                    if dist > runoff_dist:
                        runoff_dist = dist

            
        active_i = next_i
        active_j = next_j 
        
        
    return terrain, runoff_dist


p = 0.01 #Growth probability
f = 0.2 #New stone probability probability



target_num_avalanches = 300 
repititions = 10
size_of_terrain = 100

all_runouts = {planet: [] for planet, g in planet_data}
mean_runouts_per_rep = {planet: [] for planet, g in planet_data}

for planet, g in planet_data:

    for rep in range(repititions):
        terrain = np.zeros([size_of_terrain,size_of_terrain]) #Empty terrain
        runoff_dist_list = [] #Empty list of avalanche sizes

        Ni, Nj = terrain.shape


        num_avalanches = 0

        while num_avalanches < target_num_avalanches:
            #print(planet, rep, num_avalanches)

            terrain = stones_added(terrain, p)

            p_stone = np.random.rand()
            if p_stone < f:
                i0 = np.random.randint(Ni)
                j0 = np.random.randint(Nj)

                terrain, runoff = propagate_avalanche(terrain, i0, j0, g)
                if runoff > 0.0:
                    runoff_dist_list.append(runoff)
                    num_avalanches += 1

        if runoff_dist_list:
            mean_runouts_per_rep[planet].append(np.mean(runoff_dist_list))
            all_runouts[planet].extend(runoff_dist_list)
        else:
            mean_runouts_per_rep[planet].append(0.0)
        mean_runout = np.mean(runoff_dist_list)
        std_runout = np.std(runoff_dist_list)
        max_runout = np.max(runoff_dist_list)
        print(f"{planet} rep {rep}: mean={mean_runout:.2f}, std={std_runout:.2f}, max = {max_runout:.2f}")


gravities = [g for planet, g in planet_data]

planet_colours = {
    "Mercury": "red",
    "Venus" : "orange",
    "Earth": "green",
    "Mars": "brown",
    "Jupiter": "blue",
    "Saturn": "purple",
    "Uranus": "cyan",
    "Neptune": "black"
}

planet_means= []
planet_stds = []
for planet, g in planet_data:
    mean_val = np.mean(mean_runouts_per_rep[planet])
    planet_means.append(mean_val)
    std_val = np.std(mean_runouts_per_rep[planet])
    planet_stds.append(std_val)
    plt.errorbar(
        g, mean_val, yerr=std_val,
        fmt='o', capsize=4,
        color=planet_colours[planet],
        label = planet
    )
#plt.errorbar(gravities, means, yerr= stds, fmt = 'o', capsize = 4)
#plt.scatter(gravities, means)
plt.xlabel("Gravity (m/s^2)")
plt.ylabel("Mean runout distance")
plt.title("Avalanche runout vs planetary gravity")
plt.legend(title = "Planet")
plt.show()

corr_mean = np.corrcoef(gravities, planet_means)[0,1]

from scipy.stats import skew, kurtosis
for planet, g in planet_data: 
    plt.figure()
    plt.hist(all_runouts[planet], bins = 30)
    plt.title(f"Runout distribution for {planet} (g= {g})")
    plt.xlabel("Runout distance")
    plt.ylabel("Frequency")
    plt.show()

    data = all_runouts[planet]
    if len(data) > 0:
        median_val = np.median(data)
        iqr_val = np.percentile(data, 75) - np.percentile(data, 25)
        skew_val = skew(data)
        kurt_val = kurtosis(data)
        print(f"{planet}: median={median_val:.2f}, IQR={iqr_val:.2f}, skew={skew_val:.2f}, kurtosis={kurt_val:.2f}")
