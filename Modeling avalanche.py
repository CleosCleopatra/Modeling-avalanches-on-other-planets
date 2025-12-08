import numpy as np
from matplotlib import pyplot as plt
import random
from statistics import stdev

np.random.seed(5)
random.seed(5)

def stones_added(terrain, p):
    """
    Function to grow new stones in the terrain.
    
    Parameters
    ==========
    terrain: 2-dimensional array
    p: Probability for a tree to be generated in an empty cell
    """

    Ni, Nj = terrain.shape #Dimensions of forrest

    new_rocks = np.random.rand(Ni, Nj) #Random number in each place to calc whether tree grows

    #print(f"p is {p} and new_trees is {new_trees}")
    new_rocks_indices = np.where(new_rocks <= p) #The indices at which new trees actually grow
    terrain[new_rocks_indices] = 1 #Add trees

    return terrain


max_num_particles = 4

def propagate_avalanche(terrain, i0, j0):
    """
    Function to propagate the fire on a populated forest.
    
    Parameters
    ==========
    forest: 2-dimensional array
    i0: First index of the cell where the fire occurs
    j0: Second index of the cell where the fire occurs
    """

    Ni, Nj = terrain.shape #Dimensions of the forest

    fs = 0 #Initalises fire size as 0
    runoff_dist = False

    if terrain[i0, j0] == max_num_particles: #Tree where lightning strikes
        start = [i0, j0]
        runoff_dist = 0
        active_i = [i0] #Initalises list of things on fire
        active_j = [j0] #Initalises list of trees on fire
        terrain[i0, j0] = 0 #Sets tree on fire
        fs += 1 #Update fire size

        while len(active_i) > 0: #While any tree is still on fire
            next_i = []
            next_j = []
            for n in np.arange(len(active_i)): #Why do we use np.arange here?
                #Coordinates of cell up
                i = (active_i[n] + 1) % Ni 
                j = active_j[n]
                #Check status
                #if terrain[i, j] ==1:
                #    next_i.append(i)
                #    next_j.append(j)
                #    terrain[i, j] = -1 #Set current tree on fire
                #    fs += 1 #Update fire size

                terrain[i, j] += max_num_particles/4
                if terrain[i, j] >= max_num_particles:
                    next_i.append(i)
                    next_j.append(j)
                    terrain[i,j] = 0
                    dist = np.sqrt((i-start[0])**2 + (j-start[1])**2)
                    if dist > runoff_dist:
                        runoff_dist = dist
                    fs += 1

                #Coordinates of cell down
                i = (active_i[n] - 1) % Ni
                j = active_j[n]
                #Check status
                #if terrain[i, j] == 1:
                #    next_i.append(i)
                #    next_j.append(j)
                #    terrain[i, j] = -1 
                #    fs += 1
                terrain[i,j] += max_num_particles/4
                if terrain[i,j] >= max_num_particles:
                    next_i.append(i)
                    next_j.append(j)
                    terrain[i,j] = 0
                    dist = np.sqrt((i-start[0])**2 + (j-start[1])**2)
                    if dist > runoff_dist:
                        runoff_dist = dist
                    fs += 1
                
                #Coordinates of cell left
                i = active_i[n]
                j = (active_j[n] - 1) % Nj
                #Check status
                #if terrain[i, j] == 1:
                #    next_i.append(i)
                #    next_j.append(j)
                #    terrain[i, j] = -1
                #    fs += 1
                terrain[i,j] += max_num_particles/4
                if terrain[i,j] >= max_num_particles:
                    next_i.append(i)
                    next_j.append(j)
                    terrain[i,j] = 0
                    dist = np.sqrt((i-start[0])**2 + (j-start[1])**2)
                    if dist > runoff_dist:
                        runoff_dist = dist
                    fs += 1
                
                #Coordinates of cell right
                i = active_i[n]
                j = (active_j[n] + 1) % Nj

                #if terrain[i, j] == 1:
                #    next_i.append(i) #add to list
                #    next_j.append(j)
                #    terrain[i, j] = -1
                #    fs += 1
                terrain[i,j] += max_num_particles/4
                if terrain[i,j] >= max_num_particles:
                    next_i.append(i)
                    next_j.append(j)
                    terrain[i,j] = 0
                    dist = np.sqrt((i-start[0])**2 + (j-start[1])**2)
                    if dist > runoff_dist:
                        runoff_dist = dist
                    fs += 1

                #Coordinates of cell right up
                #i = (active_i[n] + 1) % Ni 
                #j = (active_j[n] + 1) % Nj

                #if terrain[i, j] == 1:
                #    next_i.append(i) #add to list
                #    next_j.append(j)
                #    terrain[i, j] = -1
                #    fs += 1
                
                #Coordinates of cell right down
                #i = (active_i[n] - 1) % Ni 
                #j = (active_j[n] + 1) % Nj

                #if terrain[i, j] == 1:
                #    next_i.append(i) #add to list
                #    next_j.append(j)
                #    terrain[i, j] = -1
                #    fs += 1
                
                #Coordinates of cell left down
                #i = (active_i[n] - 1) % Ni 
                #j = (active_j[n] - 1) % Nj

                #if terrain[i, j] == 1:
                #    next_i.append(i) #add to list
                #    next_j.append(j)
                #    terrain[i, j] = -1
                #    fs += 1
                
                #Coordinates of cell left up
                #i = (active_i[n] + 1) % Ni 
                #j = (active_j[n] - 1) % Nj

                #if terrain[i, j] == 1:
                #    next_i.append(i) #add to list
                #    next_j.append(j)
                #    terrain[i, j] = -1
                #    fs += 1
                
            
            active_i = next_i
            active_j = next_j 
        
        
    return fs, terrain, runoff_dist

#Initalise system
#N = 100 #Side of the forest
p = 0.01 #Growth probability
f = 0.2 #Lightning strike probability

#Function for complementary cumulative distribution
def complementary_CDF(f, f_max):
    """
    Function to return the complementary cumulative distribution function.
    
    Parameters
    ==========
    f : Sequence of values (as they occur, non necessarily sorted)
    f_max: Integer, maximum possible value for hte values in f
    """

    num_events = len(f)
    s = np.sort(np.array(f)) / f_max #Sort f in ascending order
    c = np.array(np.arange(num_events, 0, -1)) / (num_events) #Descending

    c_CDF = c
    s_rel = s

    return c_CDF, s_rel


#N_list=[16, 32, 64, 128, 256, 512]
target_num_avalanches = 300 #?
repititions = 5
size_of_terrain = 100

#Determine exponent for eth empirical cCDF by a linear fit
global_min_rel_size = 1e-3
global_max_rel_size = 5e-2
gravity_list = [9.18, 5.19]

all_avg_alpha = []
all_stdev_alpha = []

fig, axs = plt.subplots(3, 2)
fig.tight_layout(h_pad=3, w_pad=3)
colours = ["red", "green", "black", "blue", "pink"]
min_rel_size_list = []
for idx, g in enumerate(gravity_list):
    alpha_list = []
    s_rel_list = [] 
    c_CDF_list = []

    for rep in range(repititions):
        terrain = np.zeros([size_of_terrain,size_of_terrain]) #Empty forest
        runoff_dist_list = [] #Empty list of fire sizes

        Ni, Nj = terrain.shape

        terrain_history = []

        num_avalanches = 0

        runoff = False

        while not runoff:
            print(idx, rep, num_avalanches)

            terrain = stones_added(terrain, p)

            p_stone = np.random.rand()
            if p_stone < f:
                i0 = np.random.randint(Ni)
                j0 = np.random.randint(Nj)

                #T = int(np.sum(forest)) #Current number of trees

                fs, terrain, runoff = propagate_avalanche(terrain, i0, j0)
                runoff_dist_list.append(runoff)
                #if fs > 0:
                #    avalanche_size.append(fs)
                #    num_avalanches += 1

            #terrain[np.where(terrain == 4)] = 0

        #Lets compare the forests
        c_CDF, s_rel = complementary_CDF(runoff_dist_list, terrain.size)

        c_CDF_list.append(c_CDF)
        s_rel_list.append(s_rel)

        min_fit = max(global_min_rel_size, s_rel[0] * 1.05)
        max_fit = min(global_max_rel_size, s_rel[-1] * 0.95)


        is_min = np.searchsorted(s_rel, min_fit)
        is_max = np.searchsorted(s_rel, max_fit)

        new_p = np.polyfit(np.log(s_rel[is_min:is_max]), np.log(c_CDF[is_min: is_max]), 1)

        beta = new_p[0]
        print(f'The empirical cCDF has an exponent beta = {beta:.4}')

        alpha = 1 - beta
        print(f'The empirical prob. distr. exponent: -alpha')
        print(f'with alpha = {alpha:.4}')
        alpha_list.append(alpha)
    
    alpha_sum = sum(alpha_list)
    all_avg_alpha.append(alpha_sum / len(alpha_list))

    row = idx // 2
    col = idx % 2
    print(f"N is {idx} which gives {row, col}, ")
    print(s_rel_list)
    for i in range(len(s_rel_list)):
        axs[row, col].loglog(s_rel_list[i], c_CDF_list[i], '.-', markersize=2,
                   label=f'i = {i}', color = colours[i])

    standard_dev=stdev(alpha_list)
    all_stdev_alpha.append(standard_dev)
    axs[row, col].text(
    0.95, 0.95,
    f'avg alpha = {all_avg_alpha[-1]:.3f} +- {standard_dev:.3f}',
    transform=axs[row, col].transAxes,
    ha='right', va='top',
    fontsize=6
    )
    axs[row, col].legend(fontsize = 4)
    axs[row, col].set_title(f'Empirical cCDF for N={N}', fontsize=6)
    axs[row, col].set_xlabel('relative size', fontsize=4)
    axs[row, col].set_ylabel('c CDF', fontsize=4)
    axs[row, col].tick_params(axis='both', which='both', labelsize=6)
    axs[row, col].axvline(min_fit, color='gray', linestyle='--', linewidth=0.8)
    axs[row, col].axvline(max_fit, color='gray', linestyle='--', linewidth=0.8)


plt.show()


#avg_alpha is list
#stdev_alpha is list

#N_rev = [x**-1 for x in N_list]
#y_error_min = [all_stdev_alpha[i] for i in range(repititions+1)]
#y_error_max = [all_stdev_alpha[i] for i in range(repititions+1)]

#print(len(y_error_max))
#print(len(all_avg_alpha))
#y_error = [y_error_min, y_error_max]

#plt.errorbar(N_rev, all_avg_alpha, yerr=y_error, fmt = 'o')
plt.show()