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
from numba import jit
import imageio

G_main = nx.Graph()
filenames = []

N = 100 #number of agents
d_c = np.sqrt(4.51/(np.pi*(N-1))) #critical distance. we will work below it
R =  d_c - 0.1*d_c #distance criteria: what each node "sees"
nsteps = 1000
vmin = 0.10 #initial velocity module
vmax = 0.3
temp = 0.05 #TEMPERATURA
alpha = 1.2 #RANG ALPHA INTERESSANT: 0-1
siz_nod = 40 #for some plots
L = 1 #table size (unchangeble btm)
factor_frenada = 0

# Define the three possible states
spin1=np.array([1,0]) #amunt
spin0=np.array([0,alpha]) #el "neutre"
spin_1=np.array([-1,0]) #avall

N_of_trials = 1 #number of velocities we take

N_of_repetitions = 100 #times that we repeat each velocity

v_interval = (vmax-vmin)/N_of_trials

velocity_list = []
consent_times = []

list_of_nodes = []

for a in range(N):
    list_of_nodes.append(a)


file=open('RW'+'_N'+str(N)+'_f'+str("{:.2f}".format(factor_frenada))+'_a'+str("{:.2f}".format(alpha))+'_T'+str(temp)+'_vmin'+str(vmin)+'_vmax'+str(vmax)+'.csv','w')

results=open('res'+str(vmin)+'_'+str(vmax)+'_'+str("{:.2f}".format(alpha))+'_f'+str("{:.2f}".format(factor_frenada))+'.csv', 'w')

time_data_polar = open('time_polar_v'+str(vmin)+'_a'+str(alpha)+'.csv', 'w')
time_data_neutral = open('time_neutral_v'+str(vmin)+'_a'+str(alpha)+'.csv', 'w')
time_data_partial = open('time_partial_v'+str(vmin)+'_a'+str(alpha)+'.csv', 'w')



size_polar = open('size_polar'+str(vmin)+'_'+str(vmax)+'_'+str(alpha)+'.csv', 'w')

size_neutral = open('size_neutral'+str(vmin)+'_'+str(vmax)+'_'+str(alpha)+'.csv', 'w')

#%% Random walk function


def one_random_walk_step(v,positions,N,G,consent):
    dict = {}
    velocitats = []
    for node in range(N):
        neighs = neighbors(node, R, G)
        veins1 = 0
        veins_1 = 0
        veins0 = 0
        for nei in neighs:
            if G.nodes[nei]['orientation'] == 1:
                veins1 += 1
            elif G.nodes[nei]['orientation'] == -1:
                veins_1 += 1
            else:
                veins0 +=1
        if (veins1 > veins_1) and (veins1 > veins0):
            majoria = 1
        elif (veins_1 > veins1) and (veins_1 > veins0):
            majoria = -1
        else:
            majoria = 5 #numero random per evitar errors
        if majoria == G.nodes[node]['orientation']: #la majoria son com ell
            velocitats.append(v*factor_frenada)
        elif majoria == -G.nodes[node]['orientation']: #la majoria son oposats
            velocitats.append(v)
        else:
            velocitats.append(v)

    #print(velocitats)
    if velocitats == N*[0.0]: #all nodes are steady
        consent = True

    for a in range(N):
        v = velocitats[a]
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

    return dict, consent


#%% Function to check the distance
def euclidean_distance(pos1,pos2): #both pos1 and pos2 are tuples (x,y)
    x = pos2[0] - pos1[0]
    y = pos2[1] - pos1[1]
    d = np.sqrt(x**2+y**2)
    return d

#function that returns which nodes are inside the radius
def neighbors(node, R, G):
    neighs = []
    nodes = G.nodes()
    for a in nodes:
        if a != node:
            d = euclidean_distance(G.nodes[a]['posis'], G.nodes[node]['posis'])
            if d < R:
                neighs.append(a)
                #G.add_edge(node, a) #so we can see which nodes are interacting

    return neighs




# ------------------------------------------------------------------------------
#   ENERGY FUNCTION
# ------------------------------------------------------------------------------
#  It calculates the energy of the graph embedded system of spins in array (negative sum over neighbors pairwise products, Heisenberg hamiltonian)
#  INPUTS:
#    - G: graph we want to calculate the energy of
#    - array:  "vectors". (N,2) array with the spin state of every node
#    - R: radius

def energy_function(G,array,R,N):
  E=0
  adj_list = []      # Adjacence list, empty. it will be a list of lists
  for a in range(N): #takes every node
    node_a = [a] #in this list we will store the nodes that are interacting with a
    # isn't it what the function neighbors does?
    for b in range(N): #takes every node again to check if its a neigh of a
        if a != b:
            d = euclidean_distance(G.nodes[a]['posis'], G.nodes[b]['posis'])
            if (d < R) and (b < a): #so there are no repeated relations
                node_a.append(b)
                G.add_edge(a,b)
    adj_list.append(node_a)

  for i in range(N):    # Sum over nodes
    for j in adj_list[i]:    # Sum over nearest neighbors. j is the index of i list inside adj_list
      E = E - array[i,0]*array[int(j),0] # First component (it can be either -1, 0, or +1)
      E = E - array[i,1]*array[int(j),1] # Second component ( it can be 0 or alpha**2)
    E = E + array[i,0]*array[i,0] + array[i,1]*array[i,1]  # Discount self-contribution
#
  return E
#
# ------------------------------------------------------------------------------
#   ENERGY CHANGE FUNCTION
# ------------------------------------------------------------------------------
#  It calculates the difference of energy due to the change of orientation of one single spin from initial_spin to final_spin)
# l'he deixat tal qual
def energy_change(neigh,initial_spin,final_spin,graph):
# neigh = array containing the neighbors of the selected spin
# initial_spin = initial orientation of the selected spin
# final_spin = proposed orientation for the selected spin
#
  neighbors_0 = 0  # Initialize the sum over the first component
  neighbors_1 = 0  # Initialize the sum over the second component
  for j in neigh:
    neighbors_0 = neighbors_0 + graph.nodes[j]['orientation']
    if graph.nodes[j]['orientation'] == 0:
      neighbors_1 = neighbors_1 + alpha
#
  E_initial=-(initial_spin[0]*neighbors_0+initial_spin[1]*neighbors_1)
  E_final=-(final_spin[0]*neighbors_0+final_spin[1]*neighbors_1)
#
  diff=E_final-E_initial
#
  return diff

def initial_positions(N,G): #unused now
    positions = {}

    for a in range(N): #creates the nodes and a dictionary that includes the position of every node
        G.add_node(a)
        x = random.random() #we will have a 1x1 field
        y = random.random()
        pos = (x,y)
        positions.update({a:pos})

    nx.set_node_attributes(G,positions,'posis') #now every node has an associated position


    return G

def average_degree(G):
    suma = 0
    for node in G.nodes():
        suma += G.degree(node)
    return suma/G.number_of_nodes()


def visualitzacio(G,step,N,vel,alpha,filenames):
    color_map = []
    for node in G:
        if G.nodes[node]['orientation'] == 1:
            color_map.append('blue')
        elif G.nodes[node]['orientation'] == -1:
            color_map.append('green')
        else:
            color_map.append('red')

    pos_ini=nx.get_node_attributes(G,'posis')
    nx.draw(G, pos_ini,  with_labels = False, node_size = 30, node_color=color_map)

    # create file name and append it to a list
    filename = 'N' + str(N) + 'rep' + str(rep) + '_v' + str(vel) + '_a' + str(alpha) + 'step_' + str(step) + '.png'
    filenames.append(filename)
                    # save frame
    plt.title('a =' + str(alpha) +'vel ' + str(vel))
    plt.savefig(filename)
    plt.close()# build gif
    return
# -----------------------------------------------------------------------------
#   MAIN
# -----------------------------------------------------------------------------
#  States
# -----------------------------------------------------------------------------
#

#%%
start=time.time()


steps_array_neutral = []
steps_array_partial = []
steps_array_polar = []

for rep in range(N_of_repetitions):
    print(rep)
    orient = []
    magne = []
    timer = []
    positions = {}

    for a in range(N): #creates the nodes and a dictionary that includes the position of every node
        G_main.add_node(a)
        x = random.random() #we will have a 1x1 field
        y = random.random()
        pos = (x,y)
        positions.update({a:pos})

    nx.set_node_attributes(G_main,positions,'posis') #now every node has an associated position

    spins=np.random.randint(-1,2,N)  # Generates a random sequence of n opinion states (between -1 and 1)
    agents = dict(enumerate(spins)) # creates the dictionary to add the opinions to the nodes: {node number: spin}

    nx.set_node_attributes(G_main,agents,'orientation') # CREATED G_MAIN. WILL REMAIN THE SAME FOR THE WHOLE REPETITION

    for trial in range(N_of_trials): #this loops over the velocities. same initial configurations are used for every v (stored in G_main)
        consent = False
        G = nx.Graph()
        G = G_main #copies the main graph to an auxiliary graph to another graph. G_main will keep the initial conditions
        spins = np.array(list((nx.get_node_attributes(G,'orientation')).values()))

        num0=(spins[:]==0).sum()    # number of nodes with neutral opinion
        num1=(spins[:]==1).sum()    # number of nodes with rightist opinion
        num_1=(spins[:]==-1).sum()  # number of nodes with leftist opinion

        # Construction of initial vectors array for the energy calculation

        zeros=np.zeros(N)
        vectors=np.stack((spins,zeros), axis=-1)  # Stack zeros to the array spins, building a (N, 2) matrix
        for i in range (N):  # Loop over rows
          if vectors[i,0]==0:  # If the first component of a row is 0
            vectors[i,1]=alpha # Then the second component is alpha

        v = vmin + trial*v_interval
        print(v)

        #
        # MARKOV CHAIN.
        #--------------------------------------------------------------------------
        #step = -1
        #visualitzacio(G,step,N,v,alpha,filenames)
        step = 0
        while consent == False:                # Steps LOOP
            G = nx.create_empty_copy(G)
            nx.set_node_attributes(G,positions,'posis') #now every node has an associated position

            random.shuffle(list_of_nodes)

            for a in list_of_nodes:           # Every MC step has N flip attempts
              #a=np.random.randint(N)            # no longer random
              neigh_list = neighbors(a, R, G)

              neighborhood = int(len(neigh_list))
              neigh_array = np.asarray(neigh_list)
            #
            #  Spin Changes and their associated energy change
            #---------------------------------------------------------------------------
              if (G.nodes[a]['orientation']==-1) and (neighborhood != 0):           # Selected spin = -1
            #---------------------------------------------------------------------------
                diffenergy_10=energy_change(neigh_array,spin_1,spin0,G)  # -1 to 0
                p_10=np.exp(-((diffenergy_10)/temp)) #probability to change
            #
                diffenergy_11=energy_change(neigh_array,spin_1,spin1,G)  # -1 to +1
                p_11=np.exp(-((diffenergy_11)/temp))
            #
                K_1=1/(p_10+p_11+1)     # Normalization constant
                p_10=K_1*p_10
                p_11=K_1*p_11
        #------------------------  Acceptance Criterium  ----------------------------
        #
                randunif=np.random.uniform(0.0,1.0)
                if randunif<p_10: #the spin will change to 0
                  G.nodes[a]['orientation']=0
                  num0=num0+1
                  num_1=num_1-1

                elif randunif<(p_10+p_11): #the spin will change to 1
                  G.nodes[a]['orientation']=1
                  num1=num1+1
                  num_1=num_1-1

        #
        #---------------------------------------------------------------------------
              elif (G.nodes[a]['orientation']==0) and (neighborhood != 0):           # Selected spin = 0
        #---------------------------------------------------------------------------
                diffenergy0_1=energy_change(neigh_array,spin0,spin_1,G)  # 0 to -1
                p0_1=np.exp(-((diffenergy0_1)/temp))
        #
                diffenergy01=energy_change(neigh_array,spin0,spin1,G)     # 0 to +1
                p01=np.exp(-((diffenergy01)/temp))
        #
                K0=1/(p0_1+p01+1)     # Normalization constant
                p0_1=K0*p0_1
                p01=K0*p01
        #
        #------------------------  Acceptance Criterium  ----------------------------
        #
                randunif=np.random.uniform(0.0,1.0)
                if randunif<p0_1:
                  G.nodes[a]['orientation']=-1
                  num0=num0-1
                  num_1=num_1+1

                elif randunif<(p0_1+p01):
                  G.nodes[a]['orientation']=1
                  num0=num0-1
                  num1=num1+1

        #
        #---------------------------------------------------------------------------
              elif (G.nodes[a]['orientation']==1) and (neighborhood != 0):                           # Selected spin = 1
        #---------------------------------------------------------------------------
                diffenergy10=energy_change(neigh_array,spin1,spin0,G)    # 1 to 0
                p10=np.exp(-((diffenergy10)/temp))
        #
                diffenergy1_1=energy_change(neigh_array,spin1,spin_1,G)  # 1 to -1
                p1_1=np.exp(-((diffenergy1_1)/temp))
        #
                K1=1/(p10+p1_1+1)     # Normalization constant
                p10=K1*p10
                p1_1=K1*p1_1
        #
        #------------------------  Acceptance Criterium  ----------------------------
        #
                randunif=np.random.uniform(0.0,1.0)
                if randunif<p10:
                  G.nodes[a]['orientation']=0
                  num0=num0+1
                  num1=num1-1

                elif randunif<(p10+p1_1):
                  G.nodes[a]['orientation']=-1
                  num1=num1-1
                  num_1=num_1+1

            #per calcular l'energia
            spins = list((nx.get_node_attributes(G,'orientation')).values())
            vectors=np.stack((spins,zeros), axis=-1)  # Stack zeros to the array spins, building a (N, 2) matrix
            for i in range (N):  # Loop over rows
                if vectors[i,0]==0:  # If the first component of a row is 0
                    vectors[i,1]=alpha # Then the second component is alpha
            energy=energy_function(G,vectors,R,N)

            #visualitzacio(G,step,N,v,alpha,filenames)

            positions, consent = one_random_walk_step(v,positions,N,G,consent)
            # G_main will save the initial configuration so it can be used for all the velocities

            #print(velocitats)

            #checking orientation

            #if (num0 == N) or(num1 == N) or (num_1 == N): #consent
                #orient.append(str(G.nodes[0]['orientation']))
                #global_magnetization = abs(num1 - num_1)/N
                #magne.append(global_magnetization)
                #velocity_list.append(v)
                #consent_times.append(step)
                #consent = True
                #time_data.write(str(alpha) + '\t' + str("{:.3f}".format(v)) + '\t' + str(step) + '\n')
                #break

            if (num0 == N):
                consent = True
                steps_array_neutral.append(step)
                break
                #time_data.write(str(alpha) + '\t' + str("{:.3f}".format(v)) + '\t' + str(step) + '\n')
                #visualitzacio(G,step,N,v,alpha,filenames)

            if (num1 == N) or (num_1 == N): #consent
                consent = True
                steps_array_polar.append(step)
                break
                #time_data.write(str(alpha) + '\t' + str("{:.3f}".format(v)) + '\t' + str(step) + '\n')
                #visualitzacio(G,step,N,v,alpha,filenames)


            #if num0 == N:
            #    consent = True

            #if consent == True: #if we have reached tha max number of steps, every node is neutral or all nodes are steady
                #    orient.append(str(num0/N)) #percentatge de neutres
                #    global_magnetization = abs(num1 - num_1)/N
                #    magne.append(global_magnetization)
                #    break
            #aqui lo d'escriure el % de neutres

            if step == nsteps-1:
                consent = True
                print('we need more steps')
                break

            if consent == True: #velocitat nula
                orient.append(str(num0/N)) #percentatge de neutres
                global_magnetization = abs(num1 - num_1)/N
                magne.append(global_magnetization)
                steps_array_partial.append(step)
                #visualitzacio(G,step,N,vel,alpha,filenames)

            step += 1
            #aqui lo d'escriure el % de neutres

        #
#    for r in orient:
#        results.write(str(r) + '\t')
#    for a in magne:
#        file.write(str(a) + '\t')
#    for a in timer:
#        time_data.write(str(a) + '\t')

#    results.write('\n')
#    file.write('\n')
#    time_data.write('\n')
#%%


#for a in steps_array_polar:
#    time_data.write(str(a) + '\t')
#time_data.write('\n')
#for a in steps_array_neutral:
#    time_data.write(str(a) + '\t')
#time_data.write('\n')
#for a in steps_array_partial:
#    time_data.write(str(a) + '\t')

steps_array_polar = np.array(steps_array_polar)
steps_array_neutral = np.array(steps_array_neutral)
steps_array_partial = np.array(steps_array_partial)

for i in range(steps_array_polar.size):
  time_data_polar.write(str(steps_array_polar[i]) + '\n')

for i in range(steps_array_neutral.size):
  time_data_neutral.write(str(steps_array_neutral[i]) + '\n')

for i in range(steps_array_partial.size):
  time_data_partial.write(str(steps_array_partial[i]) + '\n')



end=time.time()                       # End measuring time
time_elapsed=end-start
print('time_elapsed = ', time_elapsed)


file.close()
results.close()
time_data_polar.close()
time_data_neutral.close()
time_data_partial.close()


