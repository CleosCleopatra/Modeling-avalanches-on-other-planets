#Trying to figure out model data
#https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011JE003865?
#States that The increase of the static angle of repose is about 5° with decreasing gravitational acceleration (from 1 to 0.1 g), whereas the dynamic angle decreases with about 10°
#
g_planet = 9.81
g_earth = []
g_relative = g_planet / g_earth

#WE are going to look at the static angle as the angle right before avalanche and dynamic angle as the angle right after avalanche

#Although we can see from the graph that the relationship between angles and g is linear, to simplify the model and find the general relationship, we are going to assume that they are. Based on model data from study
#Values on earth, I used gravel (ncrushed stone) from this study https://www.sciencedirect.com/science/article/pii/S0032591018301153?
#It gives a average angle value 0f  45 degrees for gravel and says that dynamic angle is usually around 3-10 degrees less than static
#Therefore we pick static = 47 and dynamic = 43
static_earth = 47
dynamic_angle = 43
static_angle = static_earth + 5 * (1 - g)
dynamic_angle = dynamic_earth - 10 * (1 - g)

