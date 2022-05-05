#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas as gpd
from osgeo import ogr
import pandas as pd
import tifffile as tiff #needed for the tif data for perry county
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from math import sin, cos, sqrt, atan2, radians
import sys
from shapely.geometry import Polygon, box, Point, LineString, MultiLineString
import pickle
import gurobipy as gp
from gurobipy import GRB
import tifffile as tiff #needed for the tif data for perry county
import xlwt
from xlwt import Workbook
from shapely.ops import snap, split, nearest_points
#from shapely.geometry import MultiPoint, LineString
#from dbfread import DBF
import osmnx as ox
import networkx as nx
import math
import gurobipy as gp
from gurobipy import GRB
import os


# In[2]:


os.chdir("/Users/yuelanzhu/Downloads/Research/my code file")


# In[3]:


# calculate distance
def haversinedist(lat1, lon1, lat2, lon2):
    R = 6373.0
    
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance * 1000


# In[4]:


# read data
def readClusterFile(fileID):
    file = np.genfromtxt(fileID, delimiter=",", skip_header = 1)
    file = file[:,1:]
    return file
clusterfile = 'Centralized_elevcluster' + str(1) + '.csv'


# In[5]:


building_coords = readClusterFile(clusterfile)


# In[6]:


#convert data to dataframe
df = pd.DataFrame(building_coords,
                  columns = ['longitude','latitude','elevation'])


# In[7]:


#check # of null
df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(building_coords))


# In[8]:


# cluster, n_clusters is from table 2 in paper draft, using ward-tree
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=100, affinity='euclidean', linkage='ward')
cluster.fit_predict(building_coords[:,1:2])


# In[9]:


# add cluster to dataframe
df['cluster.labels_'] = cluster.labels_.tolist()


# In[10]:


# select highest and lowest nodes in cluster
tank = df.loc[df.groupby(['cluster.labels_'])['elevation'].idxmax()].reset_index(drop=True)
trem = df.loc[df.groupby(['cluster.labels_'])['elevation'].idxmin()].reset_index(drop=True)


# In[11]:


# creat MST

graph = []
mstree = []
def createMSTnx(dataframe):
    #only takes longitude and latitude from the cluster data, note we also have elevation and stuff there too
    #coordinates = twoDcluster
    for k in range(10):
        cluster = dataframe[dataframe['cluster.labels_'] == k]
        cluster = cluster[['longitude','latitude']]
        latlon = cluster.to_numpy()
        nrows, ncols = latlon.shape
        #creates graph
        graph.append(nx.Graph())
        weights = []
        #distance in km as weight between each point in the graph
        for i in range(nrows):
            graph[k].add_node(i,pos=(latlon[i,0],latlon[i,1]))
            for j in range(i+1,nrows):
                dist = haversinedist(latlon[i,1], latlon[i,0], latlon[j,1], latlon[j,0])
                weights.append(dist)
                graph[k].add_edge(i,j, weight = dist)
        #creates MST
        mstree.append(nx.minimum_spanning_tree(graph[k]))
    return mstree, graph


# In[12]:


createMSTnx(df)


# In[13]:


# draw a simulation of graph

nx.draw(graph[9])


# In[14]:


# draw a simulation of MST

nx.draw(mstree[9],with_labels = True)


# In[15]:


m = gp.Model('pipe and pump distribution')
m.Params.timeLimit = 12000


# In[16]:


# pipe parameters
pipesize = [0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
pipesize_str, pipecost = gp.multidict({'0.05': 8.7, '0.06': 9.5, '0.08': 11,                                                    '0.1': 12.6, '0.15': 43.5,'0.2': 141, '0.25': 151, '0.3': 161, '0.35':230, '0.4': 246, '0.45':262, 
                                                   '0.5':292, '0.6':315})


# In[17]:


# pump parameters
# power in kW, default value from Matlab document
pumptype = [2, 5, 800]
# efficiency corresponding to power
pump_eff = [0.8, 0.9, 0.9]


# In[18]:


# find label of tank node in cluster
tank_lon = str(tank[tank['cluster.labels_'] == 9][['longitude']].iat[0,0])
tank_lat = str(tank[tank['cluster.labels_'] == 9][['latitude']].iat[0,0])
for i in mstree[9].nodes():
    if str(mstree[9].nodes[i]['pos'][0]) == tank_lon and str(mstree[9].nodes[i]['pos'][1]) == tank_lat:
        tank_node = i


# In[19]:


tank_node


# In[20]:


# set variables

# binary variable indicating if at edge n pipe of type t is implemented
arc_sizes = m.addVars(mstree[9].edges(), pipesize, vtype = GRB.BINARY, name = "DIAMETER")

# continuous variable indicating flow through edge n.
q = m.addVars(mstree[9].edges(),lb = 0, vtype = GRB.CONTINUOUS, name = "FLOW")

# continuous variable indicating inflow and outflow for each node
qin = m.addVars(mstree[9].nodes(),lb = 0, vtype = GRB.CONTINUOUS, name = "INFLOW")
qout = m.addVars(mstree[9].nodes(),lb = 0, vtype = GRB.CONTINUOUS, name = "OUTFLOW")

# binary variable indicating if at edge n the pump of type j is implemented.
arc_pumps = m.addVars(mstree[9].edges(),pumptype, vtype = GRB.BINARY, name = "PUMP")

#continuous variable representing the pressure injected by the pump at edge n.
pr = m.addVars(mstree[9].edges(),lb = 0, vtype = GRB.CONTINUOUS, name = "PRESSURE BY PUMP")

#continuous variable representing the head pressure at node i
H = m.addVars(mstree[9].nodes(),lb = 0, vtype = GRB.CONTINUOUS, name = "HEAD PRESSURE")

#pipe elevations at node i 
e = m.addVars(mstree[9].nodes(), vtype = GRB.CONTINUOUS, name = 'In Node Elevation')

# area of section of pipe of diameter dt
A = m.addVars(mstree[9].edges(),lb = 0, vtype = GRB.CONTINUOUS, name = "pipe area")

# head pressure loss for edge n
loss = m.addVars(mstree[9].edges(),lb = 0, vtype = GRB.CONTINUOUS, name = "loss")

# just a calculated item used to avoid bug
item = m.addVars(mstree[9].edges(),lb = 0, vtype = GRB.CONTINUOUS, name = "item")


# In[21]:


#node elevation excavation in meters
#upper bound is arbritrary maximum depth assuming 1 foot or 0.3048 meters of cover beneath the surface is needed for the pipes
#a lower bound variable is created but not used. In future models might need to implement that depending on the site (digging too deep for excavation is not feasible for many projects)
cluster9 = df[df['cluster.labels_'] == 9]
cluster9_elv = cluster9[['elevation']]
elevation_ub = dict()
elevation_lb = dict()
for i in range(mstree[9].number_of_nodes()):
    elevation_ub[i] = cluster9_elv.iloc[[i]] - 0.3048
    elevation_lb[i] = cluster9_elv.iloc[[i]] - 30


# In[49]:


innode = {}
outnode = {}
for i,j in mstree[9].edges():
    if len(nx.shortest_path(mstree[9],source=tank_node, target=i)) > len(nx.shortest_path(mstree[9],source=tank_node, target=j)):
        # if node j is nearer to tank
        innode.setdefault(i, [])
        innode[i].append(j)
        outnode.setdefault(j, [])
        outnode[j].append(i)
    else:
        # if node i is nearer to tank
        innode.setdefault(j, [])
        innode[j].append(i)
        outnode.setdefault(i, [])
        outnode[i].append(j)


# In[51]:


for i in qin:
    innode.setdefault(i,[])
    outnode.setdefault(i, [])


# In[46]:


innode


# In[47]:


outnode


# In[53]:


q


# In[67]:


qin


# In[66]:


for i in qin:
    m.addConstr(sum(q[i,j] for j in innode[i]) == sum(q[i,j] for j in outnode[i]) + 0.9464)


# In[68]:


m.addConstr(sum(q[i,j] for i in innode
                for j in innode[i]) == 
            sum(q[i,j] for i in outnode
                for j in outnode[i]) + 0.9464)


# for i in qin:
#     n = []
#     n.append(
#         len(nx.shortest_path(mstree[9],source=tank_node, target=i)) > 
#         len(nx.shortest_path(mstree[9],source=tank_node, target=j)))
#     m.addConstr(sum(q[i,j] for j in n
#     
#     m.addConstr(sum(q[i,j] for i,j in mstree[9].edges() and 
#                     len(nx.shortest_path(mstree[9],source=tank_node, target=i)) > 
#                     len(nx.shortest_path(mstree[9],source=tank_node, target=j)))
#                 == sum(q[i,j] for i,j in mstree[9].edges() and 
#                     len(nx.shortest_path(mstree[9],source=tank_node, target=i)) <
#                     len(nx.shortest_path(mstree[9],source=tank_node, target=j)))
#                 + 0.9464, "water demand")

# In[29]:


# add constraints

for i in qin:
    #qin− qout = daily water demand for each node
    #m.addConstr(qin[i]-qout[i] >= 0.9464, "water demand")# assume 250 gpd, covert it to 0.9464 m3/h
    m.addConstr(H[i] >= 68.9476, "Min head pressure") #10 psi, covert it to kpa
    m.addConstr(H[i] <= 551.581, "Max head pressure") #80 psi, covert it to kpa  
    
    # pipe elevation must be betwenn lb and ub
    m.addConstr(e[i] >= float(str(elevation_lb[i])[20:]))
    m.addConstr(e[i] <= float(str(elevation_ub[i])[20:]))


# for i,j in mstree[9].edges():
#     
#     if len(nx.shortest_path(mstree[9],source=tank_node, target=i)) < len(nx.shortest_path(mstree[9],source=tank_node, target=j)):
#         # if node i is nearer to tank
#         # define q[i,j] based on qin and qout
#         linkqin = []
#         keys = [*range(0, len(qin), 1)]
#         values = [0]*len(qin)
#         for a, b in mstree[9].edges():
#             if a==i:
#                 linkqin.append(qin[b]) 
#                 values[a]=sum(linkqin)
#         m.addConstr(qout[i] == values[i])
#         m.addConstr(q[i,j] == qout[i])
#         qoutsum = dict(zip(keys, values))

# In[28]:


for i,j in mstree[9].edges():
    
    if len(nx.shortest_path(mstree[9],source=tank_node, target=i)) < len(nx.shortest_path(mstree[9],source=tank_node, target=j)):
        # if node i is nearer to tank        
        # Pipe slope must be greater than 1%, or a pump station is used
        m.addConstr(
            ((1-arc_pumps.sum(i, j, '*'))*0.01 <= (e[i] - e[j]) / mstree[9][i][j]["weight"]), 
            ("slope max" + '[' + str(i) + ',' + str(j)+ ']'))

    if len(nx.shortest_path(mstree[9],source=tank_node, target=i)) > len(nx.shortest_path(mstree[9],source=tank_node, target=j)):
        # if node j is nearer to tank       
        # Pipe slope must be greater than 1%, or a pump station is used
        m.addConstr(
            ((1-arc_pumps.sum(i, j, '*'))*0.01 <= (e[j] - e[i]) / mstree[9][i][j]["weight"]), 
            ("slope max" + '[' + str(i) + ',' + str(j)+ ']'))
    
    #only one pipe per edge
    m.addConstr((arc_sizes.sum(i, j, '*') == 1.0), "single size chosen")
    #only one pump per edge
    m.addConstr((arc_pumps.sum(i, j, '*') <= 1.0), "single type chosen")
    
    # Velocity must be between 0.6 and 3 m/s to 30
    m.addConstr((
    q[i,j] <= ((3.14/8)*gp.quicksum(arc_sizes[i,j,k]*k**2 for k in pipesize)) * 10), "Velocity Max Constr")
    m.addConstr((
    q[i,j] >= ((3.14/8)*gp.quicksum(arc_pumps[i,j,k]*k**2 for k in pumptype)) * 0.6), "Velocity Min Constr" + str([i,j]))  
    
    # injected pressure pr[i,j] must be less than the capacity of pump.
    m.addConstr((
        pr[i,j]*q[i,j]*(9.80665*1000) <= 
        gp.quicksum(arc_pumps[i,j,k]*k*pump_eff[pumptype.index(k)] for k in pumptype)), 
        "pressure by pump less than the capacity")
    
    # conservation of Head at edge n, use Hazen Williams equation
    m.addConstr(A[i,j] == 3.14/4*gp.quicksum(arc_sizes[i,j,k]*k for k in pipesize))
    f = 0.3
    m.addConstr(item[i,j]*2*9.80665*A[i,j]==q[i,j]*q[i,j])
    m.addConstr(loss[i,j] == (gp.quicksum(arc_sizes[i,j,k]/k for k in pipesize))*f*mstree[9][i][j]["weight"]*item[i,j])#(q[i,j])**2/(2*9.80665*A[i,j]))
    m.addConstr((H[i]-H[j]==loss[i,j]-pr[i,j]),"conservation of Head")


# # add constraints
# 
# for i,j in mstree[9].edges():
#     
#     if len(nx.shortest_path(mstree[9],source=tank_node, target=i)) > len(nx.shortest_path(mstree[9],source=tank_node, target=j)):
#         # if node j is nearer to tank
#         # define q[i,j] based on qin and qout
#         linkqin = []
#         for a, b in mstree[9].edges():
#             if b==j:
#                 linkqin.append(qin[a])
#         m.addConstr(qout[j] == qoutsum[j]+sum(linkqin))
#         m.addConstr(q[i,j] == qout[j])

# In[ ]:


excavation = 25
bedding_cost_sq_ft = 6
capital_cost_pump_station = 0
ps_flow_cost = 0
ps_OM_cost = 2795
hometreatment = 5500
#water_spec_weight = 9.8
fixed_treatment_cost = 10000
#kwh_hour = 0.114
collection_om = 209


# In[ ]:


# set obective function

# pipe cost
obj1 = gp.quicksum(mstree[9][i][j]["weight"] * gp.quicksum(pipecost[str(k)] * arc_sizes[i, j, k] 
                                                           for k in pipesize) for i,j in mstree[9].edges())

# Capital cost of pump stations
obj2 = gp.quicksum(arc_pumps.sum(i, j, '*') * capital_cost_pump_station  for i,j in mstree[9].edges())

#Operation and maintenance costs
obj3 = gp.quicksum(arc_pumps.sum(i, j, '*')*ps_OM_cost for i, j in mstree[9].edges())

# cost of exaction and bedding for all installed collection systems pipes
obj4 = ((1 + gp.quicksum(arc_sizes[i,j,k]*k for k in pipesize)*0.01) * mstree[9][i][j]["weight"] 
        * bedding_cost_sq_ft + excavation * \
        (1 + gp.quicksum(arc_sizes[i,j,k]*k for k in pipesize)*0.01) * mstree[9][i][j]["weight"] * 0.5
        * ((elevation_ub[i] - e[i]) + (elevation_ub[j] - e[j]))
        for i, j in mstree[9].edges())

# cost of elevated base (in $/meter)=1
obj5 = gp.quicksum(H[i] for i in mstree[9].nodes())

obj = obj1 + obj2 + obj3 +obj5#+ obj4 


# In[ ]:


m.setObjective(obj, GRB.MINIMIZE)


# In[ ]:


#m.params.NonConvex = 2


# In[ ]:


m.optimize()


# In[ ]:


# water demand
# max velocity
# R11
# these 3 constrs are output of IIS


# In[ ]:


status = m.status
if status == GRB.Status.OPTIMAL:
    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))
    print('Obj: %g' % m.objVal)
       
elif status == GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)
    # do IIS
    m.computeIIS()
    m.write("m.ilp")
    gp.read("m.ilp")
    for c in m.getConstrs():
        if c.IISConstr:
            print('%s' % c.constrName)


# In[ ]:


m.update()
copy = m.copy()


# In[ ]:


if m.status == GRB.INFEASIBLE:
   vars = copy.getVars()
   ubpen = [1.0]*copy.numVars
   copy.feasRelax(1, False, vars, None, ubpen, None, None)
   copy.optimize()


# In[ ]:




