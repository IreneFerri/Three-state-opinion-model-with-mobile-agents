# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:36:02 2021
Updated on Tue Dec 27 04:26:00 2022

@author: ainag, irene.ferri

PRACTIQUES EN EMPRESA UBICS: RECERCA COMPUTACIONAL EN SISTEMES COMPLEXOS
APLICACIO D'UN MODEL DE TRES ESTATS EN NODES MOBILS (RANDOM WALKS)

PhD. Condensed Matter Physics

Extension of the 'Equilibrium and dynamics of a three-state opinion model' (Ferri et al), arXiv:2210.03054 [cond-mat.stat-mech]
This program calculates magnetization and fraction of neutral agents for a system of agents who moves in a square plane of dimension LxL
above the percolation, for a specific value of alpha and a range of small velocities, The aim is to connect the results with those presented in the 
cited article for Erdos-Renyi networks with average degree = 4 (appendix E)


"""
from __future__ import division #IMPORTANT per les divisions entre enters
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import time
#import numba
#from numba import jit
import json

G_main = nx.Graph()

N0 = 155
N = N0 #number of agents
small_number = int(N*0.20)  # remove preipheria
d_c = np.sqrt(4.51/(np.pi*(N-1))) #critical distance. we will work above it
R =  1.0001*d_c #distance criteria: what each node "sees"
#nsteps = int(300*np.log(N))  # Consider reached steady state
nsteps = 1  # Maximum number of steps even if the stady state has not reached
vmin = 0 #initial velocity module
vmax = 0.1
temp = 0.05 #TEMPERATURA   Lower temps can give undeflows in exponentials when calculating change probabilities
alpha = 0.0 #RANG ALPHA INTERESSANT: 0-1
siz_nod = 40 #for some plots
L = 1 #table size (unchangeble btm)
stable_condition = int(100*np.log(N))  # To Consider reached steady state
stable_condition = 30
tol = 0.001 # To Consider reached steady state

# Define the three possible states
spin1=np.array([1,0]) # rightist
spin0=np.array([0,alpha]) # neutral
spin_1=np.array([-1,0]) # leftists

N_of_trials =  1 #number of velocities we take

N_of_repetitions = 1 #times that we repeat each velocity

v_interval = (vmax-vmin)/N_of_trials

velocity_list = []
consent_times = []

list_of_nodes = []

for a in range(N):
    list_of_nodes.append(a)

# RESULTS FILES ---------------------------------------------------------

# The following file will contain 3 columns:
# First column is Fail:0.000 in case the system does not achieve consensus in the max MCS, and 'Velocity = ' str(v) otherwise
# Second column is empty in case the system does not achieve consensus in the max MCS, and 'CONSENT ACHIVED AT STEP' str(step) otherwise
# Third column is empty in case the system does not achieve consensus in the max MCS, and 'ALL THE SPINS ARE' str(state of consensus) otherwise
# All velocities are written in different lines and when a new repetition starts it just adds up more lines (no separation between reps in different blocks)  ---->  better improve this output
#file=open('RW'+'_N'+str(N)+'_a'+str(alpha)+'_T'+str(temp)+'_vmin'+str(vmin)+'_vmax'+str(vmax)+'.csv','w')
#
#
# The following file will contain a number of columns equal to the number of velocities swapped, and each line is one repetition. NaN indicates non-consensus in the max MCS, otherwise the numbers indicates the state of consensus
#results=open('res'+str(vmin)+'_'+str(vmax)+'_'+str(alpha)+'.csv', 'w')
#
#
# The following file will contain 3 columns:
# First column is alpha
# Second column is velocity
# Third column is the number of MCS that has taken to achieve consensus in one simulation. Again all velocities are written in different lines and when a new repetition starts it just adds up more lines (no separation between reps in different blocks)  ---->  better improve this output
#time_data = open('time'+str(vmin)+'_'+str(vmax)+'_'+str(alpha)+'.csv', 'w')
#
#
#  The following file will contain 8 columns: N, num_links, temp, energy, num1, num0, num_1, imc_stability
res_energy_filename = 'RW_res_T'+str(temp)+'_N'+str(N)+'_a'+str(alpha)+'_glauber_imc.csv'
res_energy = open(res_energy_filename, 'w')
#
# ---------------------------------------------------------
# FUNCTIONS ---------------------------------------------------------
# ---------------------------------------------------------
def change_dict_key(d, old_key, new_key, default_value=None):
    d[new_key] = d.pop(old_key, default_value)

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
#@jit
def euclidean_distance(pos1,pos2): #both pos1 and pos2 are tuples (x,y)
    x = pos2[0] - pos1[0]
    y = pos2[1] - pos1[1]
    d = np.sqrt(x**2+y**2)
    return d

#function that returns which nodes are inside the radius, and the graph G
def neighbors(node, nodes, R, G):
    neighs = []
    for a in nodes:
        if a != node:
            d = euclidean_distance(G.nodes[a]['posis'], G.nodes[node]['posis'])
            if d < R:
                neighs.append(a)
                G.add_edge(node, a) #so we can see which nodes are interacting

    return [neighs, G]


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
  return [E, G]
#
# ------------------------------------------------------------------------------
#   ENERGY CHANGE FUNCTION
# ------------------------------------------------------------------------------
#  It calculates the difference of energy due to the change of orientation of one single spin from initial_spin to final_spin)
# l'he deixat tal qual

def energy_change(neigh,initial_spin,final_spin,G):
# neigh = array containing the neighbors of the selected spin
# initial_spin = initial orientation of the selected spin
# final_spin = proposed orientation for the selected spin
#
  neighbors_0 = 0  # Initialize the sum over the first component
  neighbors_1 = 0  # Initialize the sum over the second component
  for j in neigh:
#    print(j,  G.nodes[j]['orientation'])
    neighbors_0 = neighbors_0 + G.nodes[j]['orientation']
    if G.nodes[j]['orientation'] == 0:
      neighbors_1 = neighbors_1 + alpha
#
  E_initial=-(initial_spin[0]*neighbors_0+initial_spin[1]*neighbors_1)
  E_final=-(final_spin[0]*neighbors_0+final_spin[1]*neighbors_1)
#
  diff=E_final-E_initial
#
  return diff

# -----------------------------------------------------------------------------
#   MAIN
# -----------------------------------------------------------------------------
#  States
# -----------------------------------------------------------------------------
#

#%%
start=time.time()



#print(positions_main)
#np.random.seed(seed=4)
for rep in range(N_of_repetitions):
    with open("RW_positions.txt", "r") as read_content:   # To read initial positions from a external file so they are always the same
        dictio = json.load(read_content)

    keys_values = dictio.items()

    positions_main = {int(key): value for key, value in keys_values}

    N = N0
    G_main = nx.Graph()
    np.random.seed(seed=4)
    orient = []
    for a in range(N): #creates the nodes and a dictionary that includes the position of every node
        G_main.add_node(a)

    nx.set_node_attributes(G_main,positions_main,'posis') #now every node has an associated position


#    np.random.seed(seed=4)
    energy_array = np.zeros(nsteps)  # Creates an array of zeros to store the energy at every step
    spins=np.random.randint(-1,2,N)  # Generates a random sequence of n opinion states (between -1 and 1)
    agents = dict(enumerate(spins)) # creates the dictionary to add the opinions to the nodes: {node number: spin}

    nx.set_node_attributes(G_main,agents,'orientation') # CREATED G_MAIN. WILL REMAIN THE SAME FOR THE WHOLE REPETITION




 # Construction of initial vectors array for the energy calculation
    spins = np.array(list((nx.get_node_attributes(G_main,'orientation')).values()))

    num0=(spins[:]==0).sum()    # number of nodes with neutral opinion
    num1=(spins[:]==1).sum()    # number of nodes with rightist opinion
    num_1=(spins[:]==-1).sum()  # number of nodes with leftist opinion

    zeros=np.zeros(N)
    vectors=np.stack((spins,zeros), axis=-1)  # Stack zeros to the array spins, building a (N, 2) matrix
    for i in range (N):  # Loop over rows
      if vectors[i,0]==0:  # If the first component of a row is 0
        vectors[i,1]=alpha # Then the second component is alpha
    energy=energy_function(G_main,vectors,R,N)[0]  # Calculate the total energy using a Heisenberg hamiltonian extended over the neighbors of each node, the second element also returns the graph
#     G = energy_function(G,vectors,R,N)[1]
    num_links = G_main.number_of_edges()
    N = G_main.number_of_nodes()
    ave_k = 2*num_links/N
    print('L = ', num_links, 'N = ', N, '<k> = ', ave_k) 
    print(nx.is_connected(G_main))
    print('INITIAL', energy, num1, num0, num_1)

    print(G_main.nodes[0]['orientation'])
    print ('........')
 ####################################################################################################################
    H = nx.Graph()
    H = G_main
# Plot the graph 
#pos=nx.spring_layout(G)
    step = -1
#imagename = 'RW_video_N'+str(N).zfill(4)+'_T'+str(temp)+'_a'+str(alpha)+'_k'+str(ave_k)+'_v'+str(v)+'_t'+str(ste    p)+'_R'+str(rep)+'.png'
    fig, ax = plt.subplots(1)
    fig.set_size_inches(9, 9)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('N = '+str(N)+' - k = '+str(ave_k)+r' - $\alpha = $'+str(alpha) + ' - T = '+str(temp), fontsize = 10    )
    ax.text(0.9, 0.9, 't = '+str(step))
    for i in range (N):
      if H.nodes[i]['orientation'] == -1:   # RED
        nx.draw_networkx_nodes(H,positions_main,nodelist=[i],node_color='#fc0202',node_size=siz_nod)
      elif H.nodes[i]['orientation'] == 0:    # BLACK

       nx.draw_networkx_nodes(H,positions_main,nodelist=[i],node_color='#000000',node_size=siz_nod)
      else:                           # BLUE
        nx.draw_networkx_nodes(H,positions_main,nodelist=[i],node_color='#0100ff',node_size=siz_nod)
    nx.draw_networkx_edges(H,positions_main,width=1.0)
    plt.tight_layout
#     fig.savefig(imagename)
    plt.show()
    plt.close()

# Clean peripheria --------------------------------------
    if (nx.is_connected(H) == False):
#      G.remove_edges_from(G.selfloop_edges())  # remove any self-loop
      for component in list(nx.connected_components(H)):
       if (len (component)<small_number):
         for node in component:
           print(node, '...rmvd')
           H.remove_node(node)
           del positions_main[node]
    G = nx.convert_node_labels_to_integers(H) # Relabel consecutively
    N = G.number_of_nodes()
    list_of_nodes = list(np.arange(N))
    new_dict = {}
#
    for key,value in enumerate(positions_main.values(),0):
      new_dict[key] = value

    positions_main = new_dict

    spins = np.array(list((nx.get_node_attributes(G,'orientation')).values()))
    zeros=np.zeros(N)
    vectors=np.stack((spins,zeros), axis=-1)  # Stack zeros to the array spins, building a (N, 2) matrix
    for i in range (N):  # Loop over rows
      if vectors[i,0]==0:  # If the first component of a row is 0
        vectors[i,1]=alpha # Then the second component is alpha
    energy=energy_function(G,vectors,R,N)[0]  

    num_links = G.number_of_edges()

    ave_k = 2*num_links/N
    energy=energy_function(G,vectors,R,N)[0]  
    print('L = ', num_links, 'N = ', N, '<k> = ', ave_k)
    print(nx.is_connected(G))

#    spins = np.array(list((nx.get_node_attributes(G,'orientation')).values()))
    print('spins len  = ', len(spins), N)
   

    num0=(spins[:]==0).sum()    # number of nodes with neutral opinion
    num1=(spins[:]==1).sum()    # number of nodes with rightist opinion
    num_1=(spins[:]==-1).sum()  # number of nodes with leftist opinion

#######################################################################################
    print('INITIAL PERIPHERIA REMOVED', energy, num1, num0, num_1)
## Plot the graph 
##################################################################################################################################################################################

    step = 0
#imagename = 'RW_video_N'+str(N).zfill(4)+'_T'+str(temp)+'_a'+str(alpha)+'_k'+str(ave_k)+'_v'+str(v)+'_t'+str(ste    p)+'_R'+str(rep)+'.png'
    fig, ax = plt.subplots(1)
    fig.set_size_inches(9, 9)
    ax.set_aspect('equal', adjustable='box')
#    ax.set_title('N = '+str(N)+' - k = '+str(ave_k)+r' - $\alpha = $'+str(alpha) + ' - T = '+str(temp), fontsize = 10    )
    ax.set_title('N = '+str(N)+' - k = '+str(ave_k), fontsize = 17)
#    ax.text(0.9, 0.9, 't = '+str(step))
    ax.text(0.9, 0.9, '(a)', fontsize = 17)
    for i in range (N):
      if G.nodes[i]['orientation'] == -1:   # RED
#        nx.draw_networkx_nodes(G,positions_main,nodelist=[i],node_color='#fc0202',node_size=siz_nod)
        nx.draw_networkx_nodes(G,positions_main,nodelist=[i],node_color='grey',node_size=siz_nod)
      elif G.nodes[i]['orientation'] == 0:    # BLACK

#       nx.draw_networkx_nodes(G,positions_main,nodelist=[i],node_color='#000000',node_size=siz_nod)
       nx.draw_networkx_nodes(G,positions_main,nodelist=[i],node_color='grey',node_size=siz_nod)
      else:                           # BLUE
#        nx.draw_networkx_nodes(G,positions_main,nodelist=[i],node_color='#0100ff',node_size=siz_nod)
        nx.draw_networkx_nodes(G,positions_main,nodelist=[i],node_color='grey',node_size=siz_nod)
    nx.draw_networkx_edges(G,positions_main,width=1.0)
    plt.tight_layout
#     fig.savefig(imagename)
    plt.show()
    plt.close()

# ************************************************************************
## write adjacency list
# ************************************************************************

#    net_name = 'RW_N'+str(N)+'_new.net'
#    K = nx.relabel_nodes(G, lambda x: x + 1)  # Fortran indexing
#    nx.write_edgelist(K, net_name, data=False)
# ********************************************************************************************************

#
#    orient = []

#    for a in range(N): #creates the nodes and a dictionary that includes the position of every node
#        G_main.add_node(a)

#    nx.set_node_attributes(G_main,positions_main,'posis') #now every node has an associated position

    #print(G_main.nodes[0]['posis'])

#    np.random.seed(seed=4)
#    energy_array = np.zeros(nsteps)  # Creates an array of zeros to store the energy at every step
#    spins=np.random.randint(-1,2,N)  # Generates a random sequence of n opinion states (between -1 and 1)
#    agents = dict(enumerate(spins)) # creates the dictionary to add the opinions to the nodes: {node number: spin}

#    nx.set_node_attributes(G_main,agents,'orientation') # CREATED G_MAIN. WILL REMAIN THE SAME FOR THE WHOLE REPETITION


    for trial in range(N_of_trials): #this LOOPS OVER THE VELOCITIES. same initial configurations are used for every v (stored in G_main)
        consent = False
 #       G = nx.Graph()
#        G = G_main #copies the main graph to an auxiliary graph to another graph. G_main will keep the initial conditions
#        spins = np.array(list((nx.get_node_attributes(G,'orientation')).values()))
#        print('spins len  = ', len(spins), N)
#        
#
#        num0=(spins[:]==0).sum()    # number of nodes with neutral opinion
#        num1=(spins[:]==1).sum()    # number of nodes with rightist opinion
#        num_1=(spins[:]==-1).sum()  # number of nodes with leftist opinion
#
#        zeros=np.zeros(N)
#        vectors=np.stack((spins,zeros), axis=-1)  # Stack zeros to the array spins, building a (N, 2) matrix
#        for i in range (N):  # Loop over rows
#          if vectors[i,0]==0:  # If the first component of a row is 0
#            vectors[i,1]=alpha # Then the second component is alpha
        energy=energy_function(G,vectors,R,N)[0]
        
        
        v = vmin + trial*v_interval
        print('-----------------')
#        print(G.nodes[0]['orientation'])

#########################################################################################################################
####################################################################################################################
        # MARKOV CHAIN.  GLAUBER DYNAMICS
        #--------------------------------------------------------------------------
        step = 0
        positions = positions_main
        stability = 0   # to check in the energy has changed or we are in a isoenergetic ensamble of configurations

        print(positions)



####################################################################################################################
####################################################################################################################
# Plot the graph 
#pos=nx.spring_layout(G)
        step = 0
#        imagename = 'RW_video_N'+str(N).zfill(4)+'_T'+str(temp)+'_a'+str(alpha)+'_k'+str(ave_k)+'_v'+str(v)+'_t'+str(step)+'_R'+str(rep)+'.png'
        fig, ax = plt.subplots(1)
        fig.set_size_inches(9, 9)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('N = '+str(N)+' - k = '+str(ave_k)+r' - $\alpha = $'+str(alpha) + ' - T = '+str(temp), fontsize = 10)
        ax.text(0.9, 0.9, 't = '+str(step))
        for i in range (N):
          if G.nodes[i]['orientation'] == -1:   # RED
            nx.draw_networkx_nodes(G,positions_main,nodelist=[i],node_color='#fc0202',node_size=siz_nod)
          elif G.nodes[i]['orientation'] == 0:    # BLACK

           nx.draw_networkx_nodes(G,positions_main,nodelist=[i],node_color='#000000',node_size=siz_nod)
          else:                           # BLUE
            nx.draw_networkx_nodes(G,positions_main,nodelist=[i],node_color='#0100ff',node_size=siz_nod)
        nx.draw_networkx_edges(G,positions_main,width=1.0)
        plt.tight_layout
#        fig.savefig(imagename)
        plt.show()
        plt.close()
####################################################################################################################



        for step in range (1, nsteps):                # Steps LOOP



          if (stability < stable_condition):
            energy_prev = energy                



            G = nx.create_empty_copy(G)
            positions = one_random_walk_step(v,positions,N)
            nx.set_node_attributes(G,positions,'posis') #now every node has an associated position
            # G_main will save the initial configuration so it can be used for all the velocities

            random.shuffle(list_of_nodes)  # randomized here
#
            for a in list_of_nodes:           # Every MC step has N flip attempts
              #a=np.random.randint(N)            # no longer random --> randomized with shuffle
              nodes = G.nodes()
              neigh_list = neighbors(a, nodes, R, G)[0]

              neighborhood = int(len(neigh_list))
              neigh_array = np.asarray(neigh_list)
#              print('neigh_array = ', neigh_array)
            #
            #  Spin Changes and their associated energy change
            #---------------------------------------------------------------------------
              if (G.nodes[a]['orientation']==-1) and (neighborhood != 0):           # Selected spin = -1
            #---------------------------------------------------------------------------
                diffenergy_10=energy_change(neigh_array,spin_1,spin0,G)  # -1 to 0
                p_10=np.exp(-(np.longdouble(diffenergy_10)/temp)) #probability to change
            #
                diffenergy_11=energy_change(neigh_array,spin_1,spin1,G)  # -1 to +1
                p_11=np.exp(-(np.longdouble(diffenergy_11)/temp))
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
                  energy=energy+diffenergy_10
                elif randunif<(p_10+p_11): #the spin will change to 1
                  G.nodes[a]['orientation']=1
                  num1=num1+1
                  num_1=num_1-1
                  energy=energy+diffenergy_11
        #
        #---------------------------------------------------------------------------
              elif (G.nodes[a]['orientation']==0) and (neighborhood != 0):           # Selected spin = 0
        #---------------------------------------------------------------------------
                diffenergy0_1=energy_change(neigh_array,spin0,spin_1,G)  # 0 to -1
                p0_1=np.exp(-(np.longdouble(diffenergy0_1)/temp))
        #
                diffenergy01=energy_change(neigh_array,spin0,spin1,G)     # 0 to +1
                p01=np.exp(-(np.longdouble(diffenergy01)/temp))
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
                  energy=energy+diffenergy0_1
                elif randunif<(p0_1+p01):
                  G.nodes[a]['orientation']=1
                  num0=num0-1
                  num1=num1+1
                  energy=energy+diffenergy01
        #
        #---------------------------------------------------------------------------
              elif (G.nodes[a]['orientation']==1) and (neighborhood != 0):                           # Selected spin = 1
        #---------------------------------------------------------------------------
                diffenergy10=energy_change(neigh_array,spin1,spin0,G)    # 1 to 0
                p10=np.exp(-(np.longdouble(diffenergy10)/temp))
        #
                diffenergy1_1=energy_change(neigh_array,spin1,spin_1,G)  # 1 to -1
                p1_1=np.exp(-(np.longdouble(diffenergy1_1)/temp))
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
                  energy=energy+diffenergy10
                elif randunif<(p10+p1_1):
                  G.nodes[a]['orientation']=-1
                  num1=num1-1
                  num_1=num_1+1
                  energy=energy+diffenergy1_1
                  if (a ==49) or (a==8):
                    print('ACCEPT')
                else:
                  if (a ==49) or (a==8):
                    print('REJECT')

              if (a ==49) or (a==8):
                print(' ************************************', a, G.nodes[a]['orientation'], neigh_list, energy_prev, energy, diffenergy1_1, p1_1)



##            #per calcular l'energia
#            spins = list((nx.get_node_attributes(G,'orientation')).values())
#            vectors=np.stack((spins,zeros), axis=-1)  # Stack zeros to the array spins, building a (N, 2) matrix
#            for i in range (N):  # Loop over rows
#                if vectors[i,0]==0:  # If the first component of a row is 0
#                    vectors[i,1]=alpha # Then the second component is alpha
#            energy=energy_function(G,vectors,R,N)[0]  # element 0 is energy, element 1 is the graph
#            G = energy_function(G,vectors,R,N)[1]  # element 0 is energy, element 1 is the graph
#            print(step, energy, num1, num0, num_1)
            #print(2*G.number_of_edges()/N)

            energy_array[step] = energy_array[step] + energy  # Store the energy in the corresponding step for every repetition
            global_magnetization = num1 - num_1
#            print(a, energy_array)
####################################################################################################################
####################################################################################################################
# Plot the graph 
#pos=nx.spring_layout(G)
            imagename = 'RW_video_N'+str(N).zfill(4)+'_T'+str(temp)+'_a'+str(alpha)+'_k'+str(ave_k)+'_v'+str(v)+'_t'+str(step)+'_R'+str(rep)+'.png'
            fig, ax = plt.subplots(1)
            fig.set_size_inches(9, 9)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title('N = '+str(N)+' - k = '+str(ave_k)+r' - $\alpha = $'+str(alpha) + ' - T = '+str(temp), fontsize = 10)
            ax.text(0.9, 0.9, 't = '+str(step))
#
            for i in range (N):
              if G.nodes[i]['orientation'] == -1:   # RED
                nx.draw_networkx_nodes(G,positions,nodelist=[i],node_color='#fc0202',node_size=siz_nod)
              elif G.nodes[i]['orientation'] == 0:    # BLACK
                nx.draw_networkx_nodes(G,positions,nodelist=[i],node_color='#000000',node_size=siz_nod)
              else:                           # BLUE
                nx.draw_networkx_nodes(G,positions,nodelist=[i],node_color='#0100ff',node_size=siz_nod)
             
            nx.draw_networkx_labels(G, pos=positions)
            nx.draw_networkx_edges(G,positions,width=1.0)
            plt.show()
            plt.tight_layout
#            fig.savefig(imagename)
            plt.close()
#########################################################################################################################
####################################################################################################################



####################################################################################################################
####################################################################################################################
####################################################################################################################
             
#            if step%10 == 0:
#                print("Repetition " + str(rep))
#                print("Velocity " + str(v))
#                print("Step " + str (step))
#                print("Active links " + str(G.number_of_edges()))
#                print("Global magnetization " + str(global_magnetization))
#                print("Neutral spins " + str(num0))
#                print("-------------------")
#            #checking orientation
#


            if (abs(energy_prev - energy) < tol):
                stability = stability + 1
            else:
                stability = 0
          else:
            break  




#            if (num0 == N) or(num1 == N) or (num_1 == N): #consent
#                file.write("Velocity " + str(v) + '\t')
#                file.write("CONSENT ACHIVED AT STEP " + str(step)+ '\t')
#                file.write("ALL THE SPINS ARE " + str(G.nodes[0]['orientation']) + '\n')
#                orient.append(str(G.nodes[0]['orientation']))
#                velocity_list.append(v)
#                consent_times.append(step)
#                consent = True
#                time_data.write(str(alpha) + '\t' + str("{:.3f}".format(v)) + '\t' + str(step) + '\n')
#                break

#        if consent == False: #if we have reached the max number of trials
#            file.write('Fail:' + str("{:.3f}".format(v)) + '\n')
#            orient.append(str('NaN'))

#    for r in orient:
#        results.write(str(r) + '\t')

    num_links = G.number_of_edges()
#    results.write('\n')
    print(N, num_links, temp, energy, num1, num0, num_1)
    res_energy.write(str(N)+' '+str(num_links)+' '+str(temp)+' '+str(energy)+' '+str(num1)+' '+str(num0)+' '+str(num_1)+' '+str(step)+'\n')
#%%
end=time.time()                       # End measuring time
time_elapsed=end-start

print(time_elapsed)

#file.close()
#results.close()
#time_data.close()
