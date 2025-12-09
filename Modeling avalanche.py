import numpy as np
from matplotlib import pyplot as plt
import math

np.random.seed(5)
#random.seed(5)

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

    #fs = 0 #Initalises avalanche size as 0
    #runoff_dist = False
    if j0 >= Nj - 1:
        return terrain, 0.0

    #neighbours_right = [i0, (j0 + 1) % Nj]
    #neighbours_right = [i0, j0+1]

    start = [i0, j0]
    runoff_dist = 0

    current_height = terrain[i0, j0]

    #avalanche = False
    #active_avalanche = []

    neighbour_height = terrain[i0, j0+1]
    angle = (current_height-neighbour_height)
    if angle <= static:
        return terrain, 0.0
        #active_avalanche.append([i0, j0])
        #avalanche = True
        #terrain[i0, j0] = 0

    #last_i, last_j = i0, j0

    terrain[i0, j0] -= 1 #Why only one?
    terrain[i0, j0+1] += 1

    active_i = [i0] 
    active_j = [j0+1]

    while active_i:
        next_i = []
        next_j = []
        for n in np.arange(len(active_i)):
            i = active_i[n]
            j = active_j[n]

            if j >= Nj -1: 
                continue #If we cant go further
            
            current_height = terrain[i, j]
            neighbour_height = terrain[i, j+1]


            angle = current_height-neighbour_height

            if angle > dynamic:
                terrain[i, j] -= 1
                terrain[i, j+1] += 1

                next_i.append(i)
                next_j.append(j+1)

                #terrain[last_i,last_j] = 0
                dist = np.sqrt((i-start[0])**2 + ((j+1)-start[1])**2)
                if dist > runoff_dist:
                    runoff_dist = dist
            
        active_i = next_i
        active_j = next_j 
        
        
    return terrain, runoff_dist

#Initalise system
#N = 100 #Side of the terrain
p = 0.01 #Growth probability
f = 0.2 #New stone probability probability

#Function for complementary cumulative distribution
#def complementary_CDF(f, f_max):
#    """
#    Function to return the complementary cumulative distribution function.
#    
#    Parameters
#    ==========
#    f : Sequence of values (as they occur, non necessarily sorted)
##    f_max: Integer, maximum possible value for hte values in f
 #   """

    #num_events = len(f)
    #if num_events == 0:
    #    return np.array([]), np.array([])
    #s = np.sort(np.array(f)) / f_max #Sort f in ascending order
    #c = np.array(np.arange(num_events, 0, -1)) / (num_events) #Descending

    #c_CDF = c
    #s_rel = s

    #return c, s


#N_list=[16, 32, 64, 128, 256, 512]
target_num_avalanches = 300 
repititions = 10
size_of_terrain = 100

#Determine exponent for eth empirical cCDF by a linear fit
#global_min_rel_size = 1e-4
#global_max_rel_size = 1e-1
#gravity_list = [9.18, 5.19]

#all_avg_alpha = []
#all_stdev_alpha = []
all_runouts = {planet: [] for planet, g in planet_data}
mean_runouts_per_rep = {planet: [] for planet, g in planet_data}

#fig, axs = plt.subplots(3, 3)
#fig.tight_layout(h_pad=3, w_pad=3)
#colours = ["red", "green", "black", "blue", "pink", "orange", "purple", "brown", "gray", "cyan"]

#min_rel_size_list = []
for planet, g in planet_data:
    #alpha_list = []
    #s_rel_list = [] 
    #c_CDF_list = []

    for rep in range(repititions):
        terrain = np.zeros([size_of_terrain,size_of_terrain]) #Empty terrain
        runoff_dist_list = [] #Empty list of avalanche sizes

        Ni, Nj = terrain.shape

        #terrain_history = []

        num_avalanches = 0

        #runoff = False


        while num_avalanches < target_num_avalanches:
            print(planet, rep, num_avalanches)

            terrain = stones_added(terrain, p)

            p_stone = np.random.rand()
            if p_stone < f:
                i0 = np.random.randint(Ni)
                j0 = np.random.randint(Nj)

                terrain, runoff = propagate_avalanche(terrain, i0, j0, g)
                if runoff > 0.0:
                    runoff_dist_list.append(runoff)
                    num_avalanches += 1
                #runoff_dist_list.append(runoff)
                #if fs > 0:
                #    avalanche_size.append(fs)
                #    num_avalanches += 1
        

            #terrain[np.where(terrain == 4)] = 0
        if runoff_dist_list:
            mean_runouts_per_rep[planet].append(np.mean(runoff_dist_list))
            all_runouts[planet].extend(runoff_dist_list)
        else:
            mean_runouts_per_rep[planet].append(0.0)
        mean_runout = np.mean(runoff_dist_list)
        std_runout = np.std(runoff_dist_list)
        max_runout = np.max(runoff_dist_list)
        print(f"{planet} rep {rep}: mean={mean_runout:.2f}, std={std_runout:.2f}, max = {max_runout:.2f}")

        #all_runouts[planet].append(runoff_dist_list)
        #print(f"[DEBUG] Planet={planet}, rep={rep}, avalanches collected={len(runoff_dist_list)}")
        #Lets compare the terrains
        #c_CDF, s_rel = complementary_CDF(runoff_dist_list, terrain.size)

        #c_CDF_list.append(c_CDF)
        #s_rel_list.append(s_rel)



        #min_fit = max(global_min_rel_size, s_rel[0] * 1.05)
        #max_fit = min(global_max_rel_size, s_rel[-1] * 0.95)


        #is_min = np.searchsorted(s_rel, min_fit)
        #is_max = np.searchsorted(s_rel, max_fit)

        #print(f"[DEBUG] Planet={planet}, rep={rep}, fit window size={is_max - is_min}, s_rel range={s_rel[0]:.4e}-{s_rel[-1]:.4e}")

        #if is_max - is_min < 3:
        #    continue

        #new_p = np.polyfit(np.log(s_rel[is_min:is_max]), np.log(c_CDF[is_min: is_max]), 1)

        #beta = new_p[0]
        #print(f'The empirical cCDF has an exponent beta = {beta:.4}')

        #alpha = 1 - beta
        #print(f'The empirical prob. distr. exponent: -alpha')
        #print(f'with alpha = {alpha:.4}')
        #alpha_list.append(alpha)
    
    #if len(alpha_list) > 0:
    #    alpha_sum = sum(alpha_list)
    #    all_avg_alpha.append(alpha_sum / len(alpha_list))


    #row = idx // 2
    #col = idx % 2
    #print(f"N is {idx} which gives {row, col}, ")
    #print(s_rel_list)
    #for i in range(len(s_rel_list)):
    #    axs[row, col].loglog(s_rel_list[i], c_CDF_list[i], '.-', markersize=2,
    #               label=f'i = {i}', color = colours[i])

    #if len(alpha_list) > 1:
    #    standard_dev=stdev(alpha_list)
    #else:
    #    standard_dev = 0.0
    #all_stdev_alpha.append(standard_dev)
    #axs[row, col].text(
    #0.95, 0.95,
    #f'avg alpha = {all_avg_alpha[-1]:.3f} +- {standard_dev:.3f}',
    #transform=axs[row, col].transAxes,
    #ha='right', va='top',
    #fontsize=6
    #)
    #axs[row, col].legend(fontsize = 4)
    #axs[row, col].set_title(f'Empirical cCDF for N={g}', fontsize=6)
    #axs[row, col].set_xlabel('relative size', fontsize=4)
    #axs[row, col].set_ylabel('c CDF', fontsize=4)
    #axs[row, col].tick_params(axis='both', which='both', labelsize=6)
    #axs[row, col].axvline(min_fit, color='gray', linestyle='--', linewidth=0.8)
    #axs[row, col].axvline(max_fit, color='gray', linestyle='--', linewidth=0.8)

    #plt.hist(runoff_dist_list, bins=30)
    #plt.title(f"Runout distribution for {planet}")
    #plt.xlabel("Runout distance")
    #plt.ylabel("Frequency")
    #plt.show()



gravities = [g for planet, g in planet_data]
means = [np.mean(mean_runouts_per_rep[planet]) for planet, g in planet_data]
stds = [np.std(mean_runouts_per_rep[planet]) for planet, g in planet_data]

plt.errorbar(gravities, means, yerr= stds, fmt = 'o', capsize = 4)
#plt.scatter(gravities, means)
plt.xlabel("Gravity (m/s^2)")
plt.ylabel("Mean runout distance")
plt.title("Avalanche runout vs planetary gravity")
plt.show()

for planet, g in planet_data: 
    plt.figure()
    plt.hist(all_runouts[planet], bins = 30)
    plt.title(f"Runout distribution for {planet} (g= {g})")
    plt.xlabel("Runout distance")
    plt.ylabel("Frequency")
    plt.show()