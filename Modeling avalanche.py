import numpy as np
from matplotlib import pyplot as plt
import math
from numba import njit

np.random.seed(5)

max_steps = 5 #50
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


#More stones topple if the gravity is lower I think, but I got to find a source?
@njit
def stones_per_topple(g):
    orig_n_stones = max(1, int(2 * 9.81/g))
    #friction_stones = max (1, int(orig_n_stones*fric))
    return 1

def stones_added(terrain):
    """
    Function to grow new stones in the terrain.
    
    Parameters
    ==========
    terrain: 2-dimensional array
    p: Probability for a tree to be generated in an empty cell
    """

    Ni, Nj = terrain.shape #Dimensions of terrain


    #new_rocks = np.random.rand(Ni, Nj) #Random number in each place to calc whether stone drops

    new_rocks_indices = np.random.randint(0, Ni), np.random.randint(0, Nj)
    #new_rocks_indices = np.where(new_rocks <= p) #The indices at which new stones drop
    terrain[new_rocks_indices] += 1 #Add stones

    return terrain, new_rocks_indices[0], new_rocks_indices[1]

directions = ((0,1),(0, -1),(1, 0), (-1, 0))
min_runoff = 2.0

@njit
def propagate_avalanche(terrain, i0, j0, n_stones, mass_move, static_loc, dynamic_loc, affected, active_mask):
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
    affected[:] = 0
    affected[i0, j0] = 1

    Ni, Nj = terrain.shape #Dimensions of the terrain

    if j0 <= 1 or j0 >= Nj - 1 or i0 <= 1 or i0 >= Ni-1:
        return terrain, 0.0, 0, 0


    runoff_dist = 0

    active_i = np.empty(max_steps * 4, dtype = np.int32)
    active_j = np.empty(max_steps * 4, dtype = np.int32)
    n_active = 1
    active_i[0] = i0
    active_j[0] = j0
    active_mask[:] = 0
    active_mask[i0, j0] = 1

    for ind, (di, dj) in enumerate(directions): 
        ni = i0 + di
        nj = j0 + dj
        if 0 <= ni < Ni and 0 <= nj < Nj:
            angle = terrain[i0, j0] - terrain[ni, nj]
            if angle > static_loc:
                terrain[i0, j0] -= min(n_stones, terrain[i0, j0])
                terrain[ni, nj] += min(n_stones, terrain[i0, j0])
                active_i[ind] = ni
                active_j[ind] = nj

                runoff_dist = max(runoff_dist, np.sqrt((ni - i0)**2 + (nj - j0)**2))

    steps = 0

    while n_active > 0 and steps < max_steps:

        new_n_active = 0
        for k in range(n_active):
            i = active_i[k]
            j = active_j[k]

            if 1 >= j or j >= Nj - 1 or 1 >= i or i >= Ni - 1:
                continue

            thresh = dynamic_loc if active_mask[i, j] else static_loc

            if terrain[i, j] < thresh:
                continue

            min_h = terrain[i, j]

            for di, dj in directions:
                ni = i + di
                nj = j + dj

                if 0 <= ni < Ni and 0 <= nj < Nj:
                    angle = terrain[i,j] - terrain[ni, nj]

                    #To add some randomness, since real granular flow is kind of stiochastic, especially right at the border

                    if angle > thresh:
                        extra = angle - thresh
                        p_avalanche = min(1.0, mass_move * (angle - dynamic_loc) / dynamic_loc) 
                        if np.random.rand() < p_avalanche:
                            moved = min(n_stones, terrain[i, j])
                            terrain[i, j] -= moved
                            terrain[ni, nj] += moved

                            n_topples += 1
                            affected[ni, nj] = 1

                            if moved > 0:
                                active_i[new_n_active] = ni
                                active_j[new_n_active] = nj
                                active_mask[ni, nj] = 1
                                new_n_active +=1
                

                            dist = np.sqrt((ni - i0)**2 + (nj - j0)**2) #Since runout is generally defined as the downlope trravel dist
                            if dist > runoff_dist:
                                runoff_dist = dist
        
        n_active = new_n_active
        avalanche_area = np.sum(affected)
        steps += 1    
        
        
    return terrain, runoff_dist, n_topples, avalanche_area


p = 0.02 #Growth probability

import seaborn as sns


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

max_runouts_per_planet = {planet: [] for planet, g in planet_data}

terrains_different_planets = {planet : [] for planet, g in planet_data}

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

        affected = np.zeros_like(terrain)
        active_mask = np.zeros_like(affected)

        while num_avalanches < target_num_avalanches:
            if rep == 0:
                sns.heatmap(terrain, cmap='coolwarm')
                plt.title(f"Number of rocks per cell, {planet}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.show()

            #print(planet, rep, num_avalanches)

            terrain, i0, j0 = stones_added(terrain)

            #i0 = np.random.randint(Ni)
            #j0 = np.random.randint(Nj)

            terrain, runoff, n_topples, avalanche_area = propagate_avalanche(terrain, i0, j0, stones, mass_move, static, dynamic, affected, active_mask)
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
        max_runouts_per_planet[planet].append(np.max(runoff_dist_list))
        if rep == 0:
            terrains_different_planets[planet] = terrain.copy()

        print(f"{planet} rep {rep}: mean={mean_runout:.2f}, median = {median_runout}, std={std_runout:.2f}")


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
err_runouts_median = []

mean_sizes = []
err_sizes = []
median_sizes = []
err_sizes_median = []

mean_areas = []
err_areas = []
median_areas = []
err_areas_median = []

max_runout_distance = []
max_runout_dist_err = []

gravities = []

for planet, g in planet_data:
    gravities.append(g)

    #Runout
    rep_means = np.array(mean_runouts_per_rep[planet])
    mean_runouts.append(np.mean(rep_means))
    err_runouts.append(rep_means.std(ddof=1) / np.sqrt(len(rep_means)))

    rep_median = np.array(all_runouts[planet])
    median_runouts.append(np.median(rep_median))
    err_runouts_median.append(rep_median.std(ddof=1) / np.sqrt(len(rep_median)))

    #Sizes
    sizes = np.array(mean_sizes_per_rep[planet])
    mean_sizes.append(np.mean(sizes))
    err_sizes.append(sizes.std(ddof=1) / np.sqrt(len(sizes)))
    sized_median = np.array(all_sizes[planet])
    median_sizes.append(np.median(sized_median))
    err_sizes_median.append(sized_median.std(ddof=1) / np.sqrt(len(sized_median)))

    #Area
    areas = np.array(mean_areas_per_rep[planet])
    mean_areas.append(np.mean(areas))
    err_areas.append(areas.std(ddof=1) / np.sqrt(len(areas)))
    area_median = np.array(all_areas[planet])
    median_areas.append(np.median(area_median))
    err_areas_median.append(area_median.std(ddof=1) / np.sqrt(len(area_median)))

    #Max
    max_runout = np.array(max_runouts_per_planet[planet])
    max_runout_distance.append(np.mean(max_runout))
    max_runout_dist_err.append(max_runout.std(ddof=1) / np.sqrt(len(max_runout)))
  #Mean runout distances
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
plt.yscale("log")
ax.set_facecolor("none")
fig.patch.set_alpha(0)
ax.set_title("Effect of gravity on mean avalanche runout distance")
ax.grid(True)
ax.margins(y=0.05)
ax.legend()
plt.tight_layout()
plt.show()


#median runout distances
fig, ax = plt.subplots()
for i, (planet, g) in enumerate(planet_data):
    plt.errorbar(
        gravities[i],
        median_runouts[i],
        yerr = err_runouts_median[i],
        fmt = 'o',
        color=planet_colours[planet],
        capsize = 4,
        markersize=8,
        elinewidth=1.5,
        label=planet
    )
ax.set_xlabel("Gravity (m/s^2)")
ax.set_ylabel("Median runout distance (grid units)")
#plt.xscale("log")
#plt.yscale("log")
ax.set_facecolor("none")
fig.patch.set_alpha(0)
ax.set_title("Effect of gravity on median avalanche runout distance")
ax.grid(True)
ax.margins(y=0.05)
ax.legend()
plt.tight_layout()
plt.show()


#Mean avalanche size (number of topples)
fig, ax = plt.subplots()
for i, (planet, g) in enumerate(planet_data):
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
ax.set_xlabel("Gravity (m/s^2)")
ax.set_ylabel("Mean avalanche size (number of topples)")
ax.set_title("Effect of gravity on mean avalanche size")
ax.grid(True)
#plt.xscale("log")
#plt.yscale("log")
plt.tight_layout()
ax.set_facecolor("none")
fig.patch.set_alpha(0)
ax.legend()
plt.show()
fig, ax = plt.subplots()

#Median avalanche size (number of topples)
fig, ax = plt.subplots()
for i, (planet, g) in enumerate(planet_data):
    plt.errorbar(
        gravities[i],
        median_sizes[i],
        yerr=err_sizes_median[i],
        fmt='o',
        color=planet_colours[planet],
        capsize=4,
        markersize=8,
        elinewidth=1.5,
        label=planet
)
ax.set_xlabel("Gravity (m/s^2)")
ax.set_ylabel("Median avalanche size (number of topples)")
ax.set_title("Effect of gravity on median avalanche size")
ax.grid(True)
#plt.xscale("log")
#plt.yscale("log")
plt.tight_layout()
ax.set_facecolor("none")
fig.patch.set_alpha(0)
ax.legend()
plt.show()
fig, ax = plt.subplots()

#Mean avalanche area (cells affected)
fig, ax = plt.subplots()
for i, (planet, g) in enumerate(planet_data):
    plt.errorbar(
        gravities[i],
        mean_areas[i],
        yerr=err_areas[i],
        fmt='o',
        color=planet_colours[planet],
        capsize=4,
        markersize=8,
        elinewidth=1.5,
        label=planet
)
ax.set_xlabel("Gravity (m/s^2)")
ax.set_ylabel("Mean avalanche area (number of cells)")
ax.set_title("Effect of gravity on mean avalanche area")
ax.grid(True)
#plt.xscale("log")
#plt.yscale("log")
plt.tight_layout()
ax.set_facecolor("none")
fig.patch.set_alpha(0)
ax.legend()
plt.show()
fig, ax = plt.subplots()

#Median avalanche area (cells affected)
fig, ax = plt.subplots()
for i, (planet, g) in enumerate(planet_data):
    plt.errorbar(
        gravities[i],
        median_areas[i],
        yerr=err_areas_median[i],
        fmt='o',
        color=planet_colours[planet],
        capsize=4,
        markersize=8,
        elinewidth=1.5,
        label=planet
)
ax.set_xlabel("Gravity (m/s^2)")
ax.set_ylabel("Median avalanche area (number of cells)")
ax.set_title("Effect of gravity on median avalanche area")
ax.grid(True)
#plt.xscale("log")
#plt.yscale("log")
plt.tight_layout()
ax.set_facecolor("none")
fig.patch.set_alpha(0)
ax.legend()
plt.show()
fig, ax = plt.subplots()

#Max values










fig, axes = plt.subplots(2,4, figsize=(14,16), sharex=True, sharey=True)

all_data = np.concatenate(list(all_runouts.values()))
bins = np.histogram_bin_edges(all_data, bins = 30)

for ax, (planet, g) in zip(axes.flat, planet_data): 
    data = all_runouts[planet]
    data2 = all_sizes[planet]
    large_mask_sizes = np.array(data2) >= 3
    large_sizes = np.array(data2)[large_mask_sizes]
    ax.hist(data, bins = bins, color=planet_colours[planet], alpha=0.8)
    ax.set_title(f"{planet} g={g}")
    ax.set_yscale("log")

    if len(data) > 0:
        median_val = np.median(data)
        mean_val = np.mean(data)
        std_val = np.std(data)
        max_val = np.max(data)

        skew_val = np.skew(data)
        kurt_val = np.kurtosis(data)

        system_size = size_of_terrain
        large_events = [r for r in data if r > 2]
        frac_large = len(large_events) / len(data)

        print(
            f"{planet}: mean runout = {mean_val}, median = {median_val}, std = {std_val}, max = {max_val}, skew = {skew_val}, kurtosis = {kurt_val}, Fraction large (>0.3L) = {100*frac_large:.2f}%"
        )

fig.supxlabel("Runout distance (grid units)")
fig.supylabel("Frequency")
fig.suptitle("Runout distance distribution across planets")
#plt.tight_layout()
plt.show()


corr_mean = np.corrcoef(gravities, mean_runouts)[0,1]
print(f"Correlation(gravity vs mean runout): {corr_mean: .3f}")


fig, axes = plt.subplots(2,4, sharex= True, sharey= True)
for ax, (planet, g) in zip(axes.flat, planet_data):
    terrain = terrains_different_planets[planet]
    im = ax.imshow(terrain, cmap = 'terrain')
    ax.set_title(f'{planet}, with g = {g} terrain after one iteration')
    ax.set_xlabel("X (grid units)")
    ax.set_ylabel("Y (grid units)")
fig.colorbar(im, ax = axes.ravel().tolist(), label = "Height (particles)")
plt.tight_layout()
plt.show()











#Graph of two steps for each of the extremes?
#Mean run-out vs. gravity
#Mean size vs. gravity
#Median runout vs. gravity
#Median size vs. gravity
#mean/median tupple size vs. gravity
#same for all except for size 1
#

"""plt.show()
planet_means= []
planet_stds = []
for planet, g in planet_data:
    mean_val = np.mean(all_runouts[planet])
    planet_means.append(mean_val)
    std_val = np.std(all_runouts[planet])
    planet_stds.append(std_val)
    plt.errorbar(
        g, mean_val, yerr=std_val,
        fmt='o', capsize=4,
        color=planet_colours[planet],
        label = f"{planet}, g = {g}"
    )
#plt.errorbar(gravities, means, yerr= stds, fmt = 'o', capsize = 4)
#plt.scatter(gravities, means)
plt.xlabel("Gravity (m/s^2)")
plt.ylabel("Mean runout distance")
plt.title("Avalanche runout vs planetary gravity")
plt.legend(title = "Planet")
plt.show()



large_runout_means = []
large_runout_stds = []
for planet, g in planet_data:
    sizes = np.array(all_sizes[planet])
    runouts = np.array(all_runouts[planet])
    mask = runouts >= 2
    large_runouts = runouts[mask]

    median = np.median(large_runouts)
    large_runout_means.append(mean_val)
    q25, q75 = np.percentile(large_runouts, [25, 75])
    large_runout_stds.append(std_val)
    plt.errorbar(
        g, median, yerr=[[median - q25], [q75-median]],
        fmt='o', capsize=4,
        color=planet_colours[planet],
        label = f"{planet}, g = {g}"
    )
#plt.errorbar(gravities, means, yerr= stds, fmt = 'o', capsize = 4)
#plt.scatter(gravities, means)
plt.xlabel("Gravity (m/s^2)")
plt.ylabel("Mean avalanche size (large avalanches only)")
plt.title("Effect of gravity on avalanches size for large-avalanches (size >= 3)")
plt.legend(title = "Planet")
plt.show()




data = []
labels = []
for planet, g in planet_data:
    sizes = np.array(all_sizes[planet])
    runouts = np.array(all_runouts[planet])

    data.append(runouts[runouts >= 2])
    labels.append(planet)

plt.boxplot(data, labels = labels, showfliers = False)

corr_mean = np.corrcoef(gravities, planet_means)[0,1]
plt.ylabel("Runout distance")
plt.title("Runout distribution for large avalanches with runout distance >= 2")
plt.xticks(rotation = 45)
plt.show()



from scipy.stats import skew, kurtosis
fig, axes = plt.subplots(2,4, figsize=(14,16), sharex=True, sharey=True)

all_data = np.concatenate(list(all_runouts.values()))
bins = np.histogram_bin_edges(all_data, bins = 30)

for ax, (planet, g) in zip(axes.flat, planet_data): 
    #counts, wins = np.histogram(all_runouts[planet], bins = 30)
    #for i in range(len(counts)):
    #    print(f"Bin {i+1}: [{wins[i], wins[i+1]}] has {counts[i]} events")
    #plt.figure()
    data = all_runouts[planet]
    data2 = all_sizes[planet]
    large_mask_sizes = np.array(data2) >= 3
    large_sizes = np.array(data2)[large_mask_sizes]
    ax.hist(data, bins = bins, color=planet_colours[planet], alpha=0.8)
    ax.set_title(f"{planet} g={g}")
    ax.set_yscale("log")
    #data = all_runouts[planet]
    if len(data) > 0:

        median_val = np.median(data)
        mean_val = np.mean(data)
        iqr_val = np.percentile(data, 75) - np.percentile(data, 25)
        skew_val = skew(data)
        kurt_val = kurtosis(data)
        print(f"{planet}, distance: median={median_val:.2f}, mean={mean_val:.2f} IQR={iqr_val:.2f}, skew={skew_val:.2f}, kurtosis={kurt_val:.2f}")

        median_val_size = np.median(data2)
        median_large_size = np.median(large_sizes)
        mean_val_size = np.mean(data2)
        iqr_val_size = np.percentile(data2, 75) - np.percentile(data2, 25)
        skew_val_size = skew(data2)
        kurt_val_size = kurtosis(data2)
        print(f"{planet}, size: median={median_val_size:.2f}, mean={mean_val_size:.2f} IQR={iqr_val_size:.2f}, skew={skew_val_size:.2f}, kurtosis={kurt_val_size:.2f}, median for larger={median_large_size}")


fig.supxlabel("Runout distance")
fig.supylabel("Frequency")
fig.suptitle("Runout distance distribution across planets")
#plt.tight_layout()
plt.show()







from scipy.stats import skew, kurtosis
fig, axes = plt.subplots(2,4, figsize=(14,16), sharex=True, sharey=True)

all_data = np.concatenate(list(all_runouts.values()))
bins = np.histogram_bin_edges(all_data, bins = 30)

for ax, (planet, g) in zip(axes.flat, planet_data): 
    data = all_runouts[planet]
    data2 = all_sizes[planet]
    large_mask_sizes = np.array(data2) >= 3
    large_sizes = np.array(data2)[large_mask_sizes]
    ax.hist(data, bins = bins, color=planet_colours[planet], alpha=0.8)
    ax.set_title(f"{planet} g={g}")
    ax.set_yscale("log")

fig.supxlabel("Runout distance")
fig.supylabel("Frequency")
fig.suptitle("Runout distance distribution across planets")
#plt.tight_layout()
plt.show()


Low-Gravity planets have rarer but more extreme avalanches (higher skeness and kurtosis)
High gravity planets proudce longer and more symmetric runout, resulting in higher median runout and reduced distribution skew
Mean area and size decreases with gravity up to Earth like conditions, but increases again at very high gravity, usggesting competining stabilising effects
Across all planets, large events occured in less than 3% of cases. 

Effect on area:
Mean runout distance increases overall with gravity
Earth, venus, saturn, uranus and neptune closer, indicating weak sensitivity under earth like conditions

Median is one for all planets except jupiter, indicating gravity mainly affects tail of distribution
    Low gravity have higher skewness and kurtosis, implying higher likelihood of rare extreme large_events


"""


