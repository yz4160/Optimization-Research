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
import csv
from sklearn.cluster import AgglomerativeClustering


# In[2]:


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
    return distance #unit in km


# In[3]:


# read data
def readClusterFile(fileID):
    file = np.genfromtxt(fileID, delimiter=",", skip_header = 1)
    file = file[:,1:]
    return file
clusterfile = 'Centralized_elevcluster' + str(1) + '.csv'


# In[4]:


# read data
def readClusterFile(fileID):
    file = np.genfromtxt(fileID, delimiter=",", skip_header = 1)
    file = file[:,1:]
    return file
clusterfile = 'Centralized_elevcluster' + str(1) + '.csv'
building_coords = readClusterFile(clusterfile)


# In[5]:


# pipe parameters
pipesize = [0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6] #m
pipesize_str, pipecost = gp.multidict({'0.05': 8.7, '0.06': 9.5, '0.08': 11,                                                    '0.1': 12.6, '0.15': 43.5,'0.2': 141, '0.25': 151, '0.3': 161, '0.35':230, '0.4': 246, '0.45':262, 
                                                   '0.5':292, '0.6':315}) #$/m
excavation = 25
bedding_cost_sq_ft = 6
capital_cost_pump_station = 10#166000
ps_flow_cost = 0.19
ps_OM_cost = 10#175950


# In[6]:


# creat MST

graph = []
mstree = []
def createMSTnx(dataframe,n):
    #only takes longitude and latitude from the cluster data, note we also have elevation and stuff there too
    #coordinates = twoDcluster
    cluster = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward')
    cluster.fit_predict(building_coords[:,1:2])
    # add cluster to dataframe
    dataframe['cluster.labels_'] = cluster.labels_.tolist()
    for k in range(n):
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


# In[7]:


#convert data to dataframe
df = pd.DataFrame(building_coords,
                  columns = ['longitude','latitude','elevation'])
createMSTnx(df,58)


# In[8]:


def minlp(cluster_labels, pipesize, pipecost, water_demand, Hmax, excavation, bedding_cost_sq_ft, capital_cost_pump_station, ps_OM_cost):
    
    # select highest and lowest nodes in cluster
    tank = df.loc[df.groupby(['cluster.labels_'])['elevation'].idxmax()].reset_index(drop=True)
    trem = df.loc[df.groupby(['cluster.labels_'])['elevation'].idxmin()].reset_index(drop=True)
                      
     # find label of tank node in cluster
    tank_lon = str(tank[tank['cluster.labels_'] == cluster_labels][['longitude']].iat[0,0])
    tank_lat = str(tank[tank['cluster.labels_'] == cluster_labels][['latitude']].iat[0,0])
    for i in mstree[cluster_labels].nodes():
        if str(mstree[cluster_labels].nodes[i]['pos'][0]) == tank_lon and str(mstree[cluster_labels].nodes[i]['pos'][1]) == tank_lat:
            tank_node = i                  
    #create flow direction dict
    links = list(mstree[cluster_labels].edges)
    flow_dir = {}
    for i,j in links:
        if len(nx.shortest_path(mstree[cluster_labels],source=tank_node, target=i)) < len(nx.shortest_path(mstree[cluster_labels],source=tank_node, target=j)):
        # if node i is nearer to tank
           flow_dir[i,j]=(i,j)
        if len(nx.shortest_path(mstree[cluster_labels],source=tank_node, target=i)) > len(nx.shortest_path(mstree[cluster_labels],source=tank_node, target=j)):
         # if node j is nearer to tank
           flow_dir[i,j]=(j,i) 
    innode = {}
    outnode = {}
    for i,j in links:
        if len(nx.shortest_path(mstree[cluster_labels],source=tank_node, target=i)) > len(nx.shortest_path(mstree[cluster_labels],source=tank_node, target=j)):
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
    for i in mstree[cluster_labels].nodes():
        innode.setdefault(i,[])
        outnode.setdefault(i, [])
   # pipe flow
    pipe_flow = {}
    for i,j in links:
        start = flow_dir[i,j][0]
        end = flow_dir[i,j][1]
        pipe_flow[i,j] = water_demand*(len(outnode[end])+1) #250gpd to m3/s  *3  
   # Edge list
    edge_list = {}
    for i,j in links:
        edge_list[i,j] =[pipe_flow[i,j],mstree[cluster_labels][i][j]["weight"]]
   #node elevation excavation in meters
   #upper bound is arbritrary maximum depth assuming 1 foot or 0.3048 meters of cover beneath the surface is needed for the pipes
   #a lower bound variable is created but not used. In future models might need to implement that depending on the site (digging too deep for excavation is not feasible for many projects)
    cluster = df[df['cluster.labels_'] == cluster_labels]
    cluster_elv = cluster[['elevation']]
    elevation_ub = dict()
    elevation_lb = dict()
    for i in range(mstree[cluster_labels].number_of_nodes()):
        elevation_ub[i] = cluster_elv.iloc[[i]] - 0.3048
        elevation_lb[i] = cluster_elv.iloc[[i]] - 30                 
    m = gp.Model('pipe and pump distribution')
    m.Params.timeLimit = 12000
   # binary variable indicating if at link ij pipe of diameter k is implemented
    d = m.addVars(mstree[cluster_labels].edges(), pipesize, vtype = GRB.BINARY, name = "diameter")
   #continuous variable representing the hydro loss in link ij.
    HL = m.addVars(mstree[cluster_labels].edges(),lb = 0, vtype = GRB.CONTINUOUS, name = "hydro loss")
    # hydro gradient
    J = m.addVars(mstree[cluster_labels].edges(),lb = 0, vtype = GRB.CONTINUOUS, name = "hydro gradient")
   # binary variable indicating if at link ij the pump of type m is implemented.
    p = m.addVars(mstree[cluster_labels].edges(), vtype = GRB.BINARY, name = "PUMP")
   #continuous variable representing the pressure injected by the pump at link ij.
    pr = m.addVars(mstree[cluster_labels].edges(),lb = 0, vtype = GRB.CONTINUOUS, name = "PRESSURE BY PUMP")
   #pipe elevations at node i 
    e = m.addVars(mstree[cluster_labels].nodes(), vtype = GRB.CONTINUOUS, name = 'In Node Elevation')
   #continuous variable representing the head pressure at node i
    P = m.addVars(mstree[cluster_labels].nodes(),lb = 0, vtype = GRB.CONTINUOUS, name = "HEAD PRESSURE")
    for i,j in links:
#Hazen Williams equation, A = 10700, C=140, P472
        m.addConstr(J[i,j] == 
                10700*(edge_list[i,j][0]/140)**1.852*gp.quicksum(d[i,j,k]/(k**4.8704) for k in pipesize)
               , "Hydro loss")
        m.addConstr(HL[i,j] == J[i,j]*mstree[cluster_labels][i][j]["weight"])
        m.addConstr(HL[i,j] <= Hmax)
#only one pipe per edge
        m.addConstr((d.sum(i, j, '*') == 1.0), "single size chosen")
#injected pressure pr[i,j] must be less than a huge number.
        m.addConstr(pr[i,j]<=p[i,j]*10**6)
        if len(nx.shortest_path(mstree[cluster_labels],source=tank_node, target=i)) < len(nx.shortest_path(mstree[cluster_labels],source=tank_node, target=j)):
        # if node i is nearer to tank        
            m.addConstr(P[j] == P[i] -HL[i,j] + e[i] - e[j] + pr[i,j], "Energy balance")
        if len(nx.shortest_path(mstree[cluster_labels],source=tank_node, target=i)) > len(nx.shortest_path(mstree[cluster_labels],source=tank_node, target=j)):
        # if node j is nearer to tank        
            m.addConstr(P[i] == P[j] -HL[i,j] + e[j] - e[i] + pr[i,j], "Energy balance")
# Velocity must be between 0.6 and 3 m/s to 30
        m.addConstr((
           edge_list[i,j][0] <= ((3.14/8)*gp.quicksum(d[i,j,k]*k**2 for k in pipesize)) * 10), "Velocity Max Constr"+ str([i,j]))
    #m.addConstr((
    #edge_list[i,j][0] >= ((3.14/8)*gp.quicksum(d[i,j,k]*k**2 for k in pipesize)) * 0.6), "Velocity Min Constr" + str([i,j]))  
    for i in mstree[cluster_labels].nodes():
#pipe elevation must be betwenn lb and ub
        m.addConstr(e[i] >= float(str(elevation_lb[i])[20:]), "PIPE ELV LB")
        m.addConstr(e[i] <= float(str(elevation_ub[i])[20:]), "PIPE ELV UB")
    
        m.addConstr(P[i] >= 7.03, "Min head pressure") #10 psi, covert it to m
        m.addConstr(P[i] <= 56.26, "Max head pressure") #80 psi, covert it to m                    
                      
                      
    # pipe cost 
    obj1 = gp.quicksum(1000 * mstree[cluster_labels][i][j]["weight"] * gp.quicksum(pipecost[str(k)] * d[i, j, k] 
                                                           for k in pipesize) for i,j in mstree[cluster_labels].edges())
    # Capital cost of pump stations
    obj2 = gp.quicksum(p.sum(i, j) * capital_cost_pump_station for i,j in mstree[cluster_labels].edges())

    #Operation and maintenance costs
    obj3 = gp.quicksum(p.sum(i, j)*ps_OM_cost for i, j in mstree[cluster_labels].edges())

    obj = obj1+obj2+obj3                  
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    
    
    #save the result
    modelname = "pipe&pump distri" + str(cluster_labels) + ".csv" 
        #m.write(modelname)
    modelfile = open(modelname, "w")
    #modelfile.write("pipecost" + str(pipecost))
    modelfile.write('number of nodes: %g \n' % len(mstree[cluster_labels].nodes()))
    modelfile.write('capital_cost_pump_station: %g \n' % capital_cost_pump_station)
    modelfile.write('ps_OM_cost: %g \n' % ps_OM_cost)
    modelfile.write('water_demand: %g \n' % water_demand)
    modelfile.write('Solution Value: %g \n' % m.objVal)
    modelfile.write('Solution Value 1: %g \n' % obj1.getValue())
    modelfile.write('Solution Value 2: %g \n'% obj2.getValue())
    modelfile.write('Solution Value 3: %g \n'% obj3.getValue())                  
    for v in m.getVars():
        modelfile.write('%s %g \n' % (str(v.varName), v.x))
    modelfile.close()                  
    
    return obj.getValue(), obj1.getValue(), obj2.getValue(), obj3.getValue()
                      


# In[9]:


minlp(6, pipesize, pipecost, 30*1.095E-5, 0.2, excavation, bedding_cost_sq_ft, capital_cost_pump_station, ps_OM_cost)


# In[ ]:




