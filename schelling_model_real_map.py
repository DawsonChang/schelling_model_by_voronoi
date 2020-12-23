import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import pandas as pd
import matplotlib.pyplot as plt
import mapclassify as mc
from shapely.geometry import Point

import numpy as np
from numpy.random import seed
from numpy.random import rand
from numpy.random import shuffle
import time

from scipy.spatial import Delaunay
from collections import defaultdict
import itertools

import random

from PIL import Image, ImageDraw


# graph ploting
def print_graph():
    # global graph_num
    # global images

    groupArr = np.ravel(group)
    
    d = pd.DataFrame({'group': groupArr})
    d['geometry'] = points
    house = gpd.GeoDataFrame(d, geometry='geometry')
    
    scheme = mc.NaturalBreaks(house['group'], k=3)
    
    ax = gplt.voronoi(
        house, clip=curr_map, linewidth=0.5, edgecolor='white',
        projection=gcrs.Mercator(),hue='group', scheme=scheme, cmap='Blues',
        legend=True
    )
    
    gplt.polyplot(curr_map, edgecolor='None', facecolor='lightgray', ax=ax)
    gplt.pointplot(house, ax=ax, extent=curr_map.total_bounds)
    plt.title('Schelling model of Melbourne' + ', t: ' + str(t) )
    plt.show()

def random_points_in_polygon(number, polygon):
    points = []
    X = []
    Y = []
    min_x, min_y, max_x, max_y = polygon.bounds
    i= 0
    while i < number:
        cur_x = random.uniform(min_x, max_x)
        cur_y = random.uniform(min_y, max_y)
        point = Point(cur_x, cur_y)
        if polygon.contains(point):
            points.append(point)
            X.append(cur_x)
            Y.append(cur_y)
            i += 1
    return points, X, Y


N = 1000
empty_rate = 0.1

# read the map
curr_map = gpd.read_file(gplt.datasets.get_path('melbourne'))
points, X, Y = random_points_in_polygon(N, curr_map.iloc[0].geometry)


# generate N random points (X,Y) 
seed(102)
t = 0.7

# generate the three groups (including empty house)
empty = int(N * empty_rate)
first_group = int((N - empty) / 2)
second_group = N - empty - first_group
group = [0.0] * empty + [1.0] * first_group + [2.0] * second_group

# generate the four groups (including empty house)

# initialize them randomly
t_time = 1000 * time.time()
seed(int(t_time) % 2**32)
shuffle(group)

# plot the init graph
print_graph()

# generate the hashtable for adjacent cell of voronoi diagram
points_voronoi = list(zip(X, Y))
tri = Delaunay(points_voronoi)
neiList = defaultdict(set)
for p in tri.vertices:
    for i,j in itertools.combinations(p,2):
        neiList[i].add(j)
        neiList[j].add(i)


# process of Schelling model
steps = 0
while(True):
    steps += 1
    needToMove = []
    groupValue = []
    for i in range(N):
        totalNeighbor = 0
        neighborInSameGroup = 0
        if (group[i] == 0):
            needToMove.append(i)
            groupValue.append(group[i])
            continue
        for j in neiList[i]:
            if (group[j] != 0):
                totalNeighbor += 1
            if (group[i] == group[j]):
                neighborInSameGroup += 1
        if(totalNeighbor == 0):
            needToMove.append(i)
            groupValue.append(group[i])
            continue
        if(neighborInSameGroup/totalNeighbor < t):
            needToMove.append(i)
            groupValue.append(group[i])
    
    # length of needToMove is equal to empty, meaning that there is no people 
    # should move (they are satisfied with their neighbor)
    if(len(needToMove) <= empty):
        break
    
    t_time = 1000 * time.time()
    seed(int(t_time) % 2**32)
    shuffle(groupValue)
    
    k = 0
    for i in needToMove:
        group[i] = groupValue[k]
        k += 1


print("steps: {}".format(steps))
# plot the result graph
print_graph()
