# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:36:02 2021

@author: ainag

PRACTIQUES EN EMPRESA UBICS: RECERCA COMPUTACIONAL EN SISTEMES COMPLEXOS
APLICACIO D'UN MODEL DE TRES ESTATS EN NODES MOBILS (RANDOM WALKS)

"""
from __future__ import division #IMPORTANT per les divisions entre enters
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import time
import numba
import json
from numba import jit

G_main = nx.Graph()

N0 = 155 #number of agents
N = N0
small_number = int(N*0.20)  # remove preipheria
# For this particular N0, R and the positions.txt used in this simulation, cleaning peripheria gives single connected component with N = 104, ave_k = 4.86, modularity = 0.79
d_c = np.sqrt(4.51/(np.pi*(N-1))) #critical distance. we will work below it
R =  1.00001*d_c #distance criteria: what each node "sees"
#
nsteps = 100
vmin = 0.400 #initial velocity module
vmax = 0.410
temp = 0.05 #TEMPERATURA
alpha = 0.75 #RANG ALPHA INTERESSANT: 0-1
siz_nod = 40 #for some plots
L = 1 #table size (unchangeble btm)
#
#
N_of_trials = 1 #number of velocities we take
#
N_of_repetitions = 100 #times that we repeat each velocity
#
v_interval = (vmax-vmin)/N_of_trials
#
velocity_list = []
consent_times = []
#
list_of_nodes = []
#
for a in range(N):
    list_of_nodes.append(a)
#
#file=open('reachab_normal_0'+'_N'+str(N)+'_a'+'_v'+str(vmin)+'.csv','w')
file=open('reachab_normal_1'+'_N'+str(N)+'_a'+'_v'+"%5.3f"%vmin+'.csv','w')
#file=open('reachab_normal__1'+'_N'+str(N)+'_a'+'_v'+str(vmin)+'.csv','w')
#
#
#%% Random walk function

def one_random_walk_step(v,positions,N):
    dict = {}
    for a in range(N):
        x = positions[a][0]
        y = positions[a][1]
        alpha = random.uniform(0,2*np.pi)
        deltax = v*np.cos(alpha)
        deltay = v*np.sin(alpha)
        new_x = x + deltax
        new_y = y + deltay
        while new_x > 1: #if the agents leave the field, they apper on the other side
            new_x -= 1 #like if we had periodical countorn conditions
        while new_x < 0:
            new_x += 1
        while new_y > 1:
            new_y -= 1
        while new_y < 0:
            new_y += 1
        pos = (new_x, new_y)
        dict.update({a:pos})
         #graph is actuallised with the new position of every node

    return dict

#%% Function to check the distance

def euclidean_distance(pos1,pos2): #both pos1 and pos2 are tuples (x,y)
    x = pos2[0] - pos1[0]
    y = pos2[1] - pos1[1]
    d = np.sqrt(x**2+y**2)
    return d

#function that returns which nodes are inside the radius
def neighbors(node, nodes, R, G):
    neighs = []
    for a in nodes:
        if a != node:
            d = euclidean_distance(G.nodes[a]['posis'], G.nodes[node]['posis'])
            if d < R:
                neighs.append(a)
                G.add_edge(node, a) #so we can see which nodes are interacting

    return neighs

def build_graph_edges(G,R):
  E=0
  N = G.number_of_nodes()
  for a in range(N): #takes every node
    for b in range(N):
        if a != b:
            d = euclidean_distance(G.nodes[a]['posis'], G.nodes[b]['posis'])
            if (d < R) and (b < a): #so there are no repeated relations
                G.add_edge(a,b)
  return G

# -----------------------------------------------------------------------------
#   MAIN
# -----------------------------------------------------------------------------
#  Positions
# -----------------------------------------------------------------------------
#
with open("RW_positions.txt", "r") as read_content:   # To read initial positions from a external file so they are always the same
    dictio = json.load(read_content)

keys_values = dictio.items()

#%%
start=time.time()

recull = [0]*nsteps
for rep in range(N_of_repetitions):
#
    positions_main = {int(key): value for key, value in keys_values}
    N = N0 #number of agents  at the beggining
    G_main = nx.Graph()
    orient = []
    for a in range(N): #creates the nodes and a dictionary that includes the position of every node
        G_main.add_node(a)
#    
##################################################################    
    nx.set_node_attributes(G_main,positions_main,'posis') #now every node has an associated position
#    for i in range(N):
#      print(i, G_main.nodes[i]['posis'])
#
# Orientation/ Opinions
    spins=np.random.randint(-1,2,N)  # Generates a random sequence of n opinion states (between -1 and 1)
    agents = dict(enumerate(spins)) # creates the dictionary to add the opinions to the nodes: {node number: spin}


    for trial in range(N_of_trials): #this loops over the velocities. same initial configurations are used for every v (stored in G_main)
        G = nx.Graph()
        G = G_main #copies the main graph to an auxiliary graph to another graph. G_main will keep the initial conditions
#        for i in range(N):
#          print(i, G.nodes[i]['posis'])
        nx.set_node_attributes(G,agents,'orientation') # CREATED G_MAIN. WILL REMAIN THE SAME FOR THE WHOLE REPETITION
        spins = np.array(list((nx.get_node_attributes(G,'orientation')).values()))
#
        G = build_graph_edges(G,R)
#        for i in range(N):
#          print(i, G.nodes[i]['posis'])
# ---------------------------------------------------------------------------------------------------------------------------
# Clean peripheria --------------------------------------
####################################################################################################################
        H = G
########################
        if (nx.is_connected(H) == False):
#          G.remove_edges_from(G.selfloop_edges())  # remove any self-loop
          for component in list(nx.connected_components(H)):
            if (len (component)<small_number):
              for node in component:
#                print(node, '...rmvd')
                H.remove_node(node)
                del positions_main[node]
        G = nx.convert_node_labels_to_integers(H) # Relabel consecutively
        N = G.number_of_nodes()
        list_of_nodes = list(np.arange(N))
        new_dict = {}
#
        for key,value in enumerate(positions_main.values(),0):
          new_dict[key] = value
#   Update positions dict  
        positions_main = new_dict
#
# Links, average degree and number of nodes
#        num_links = G.number_of_edges()
#        ave_k = 2.0*num_links/N
#        print('spins len  = ', len(spins), N)
#   Positives, neutrals and negatives
        num0=(spins[:]==0).sum()    # number of nodes with neutral opinion
        num1=(spins[:]==1).sum()    # number of nodes with rightist opinion
        num_1=(spins[:]==-1).sum()  # number of nodes with leftist opinion
#
        print('INITIAL PERIPHERIA REMOVED', N,  num1, num0, num_1)
#        print('L = ', num_links, 'N = ', N, '<k> = ', ave_k)
        print(nx.is_connected(G))
        if (nx.is_connected(G) == False):   # If we still don't have a single connected component
          print("NON_CONNECTED", rep)
          break
# ----------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
#
        not_positive = []

        for node in list_of_nodes:
            if spins[node] != 1: #nodes que no son positives
            #if spins[node] != 0: #nodes que no son neutrals
#            if spins[node] != -1: #nodes que no son negatives 
                not_positive.append(node)

        v = vmin + trial*v_interval
        #
        # MARKOV CHAIN.
        #--------------------------------------------------------------------------
# ---- All agents back to main positions        
        positions = positions_main
#  Steps loop for every rep (only one vel)
        for step in range (nsteps):                # Steps LOOP
            G = nx.create_empty_copy(G)
            positions = one_random_walk_step(v,positions,N)
            nx.set_node_attributes(G,positions,'posis') #now every node has an associated position
            # G_main will save the initial configuration so it can be used for all the velocities
            new_not_positive = not_positive
            for a in not_positive:           # Every non positive agent 
              #a=np.random.randint(N)            # no longer random
              nodes = G.nodes()
              # Neighbors of the non-positive agent "a"
              neigh_list = neighbors(a, nodes, R, G)

              for b in neigh_list: #if any neigh is positive
#                  print(new_not_positive)
                  if G.nodes[b]['orientation'] == 1:   # Positive neighbors --- > so it counts links between |+1> - |0> and |+1> - |-1>
#                  if G.nodes[b]['orientation'] == 0:
#                  if G.nodes[b]['orientation'] == -1:
                      try:
                          new_not_positive.remove(a) #we take into account the contact  -> non-positive node "a" has visited a positive node
                      except ValueError: #already removed
#                          print('error')
                          pass
            not_positive = new_not_positive
            recull[step] += (1-(len(not_positive)/N))   # Store the number of non-positive nodes that has never been in touch with a positive node
#            print(str(step) +'\t' + str(1-(len(not_positive)/N)) + '\n')

#
# ******** ELASTIC *****************************
            positions = positions_main  # they return to main positions
#%%
for step in range(nsteps):
    recull[step] = recull[step]/N_of_repetitions
    file.write(str(step) +'\t' + str(recull[step]) + '\n')


end=time.time()                       # End measuring time
time_elapsed=end-start

print(time_elapsed)

file.close()
