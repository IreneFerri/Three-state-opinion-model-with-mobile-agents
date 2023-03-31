# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:36:02 2021
Updated on Wed Jan 11 11:15:00 2023


@author: ainag, irene.ferri
"""

import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import time
import json
import community as community_louvain

G_main = nx.Graph()

#DIAGRAMA DE FASE AMB VELOCITAT I RADI. (no encara), o velocitat vs alpha

#en funcio d alpha, punt on passen de tenir consens a +-1 a 0. 

# # temps de consens vs velocitat x alpha 0 i alpha 1. mirar l'efecte de la velocitat. 

#TREBALLAR X SOTA DE LA DENSITAT CRITICA (PERCOLACIÓ). Fixar radi i N FET

# magnetitzacio global: espins amunt - espins avall, tambe els que estan neutres. FET

# mesurar en quant temps arriben al consens. mirar primer si realment arriben 
#sempre al consens. FET


# FER DOS ARRAYS UN PER L'ESTAT ACTUAL I UN X EL FUTUR FET

#observacions pel futur: 
    

N0 = 155 #number of agents
N = N0
small_number = int(N*0.20)  # remove preipheria
d_c = np.sqrt(4.51/(np.pi*(N-1))) #critical distance. we will work below it
R =  1.00001*d_c #distance criteria: what each node "sees"
nsteps = 50*N
vmin = 0.07 #initial velocity module
vmax = 0.08
temp=0.05 #TEMPERATURA NO TOQUIS
alpha = 1.25 #PROVAR AMB ALPHA 0 I ALPHA 1
siz_nod = 40
L = 1 #IMPLEMENTAR QUE ES PUGUI CANVIAR LA L?¿
#es com que ho normalitza amb el grau però no se perq

N_of_trials = 1

N_of_repetitions = 100

v_interval = (vmax-vmin)/N_of_trials

velocity_list = []
consent_times = []

fails = open('fails_RW'+'_N'+str(N)+'_a'+str(alpha)+'_T'+str(temp)+'_v'+str(vmin)+'.csv','w')
consent_file=open('RW'+'_N'+str(N)+'_a'+str(alpha)+'_T'+str(temp)+'_v'+str(vmin)+'.csv','w')



"""

fig, ax = plt.subplots()
plt.title("Initial configuration")
nx.draw(G,positions, with_labels = True,ax=ax) 
limits=plt.axis('on') # turns on axis
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
"""

#%% Random walk function

def one_random_walk_step(G):
    N =  G.number_of_nodes()
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
        G.nodes[a]['posis'] = pos #graph is actuallised with the new position of every node
   
    return G


#%% Function to check the distance

def euclidean_distance(pos1,pos2): #both pos1 and pos2 are tuples (x,y)
    x = pos2[0] - pos1[0]
    y = pos2[1] - pos1[1]
    d = np.sqrt(x**2+y**2)
    return d 

def neighbors(node, R, G):
    neighs = []
    nodes = G.nodes()
    for a in nodes:
        if a != node: 
            d = euclidean_distance(G.nodes[a]['posis'], G.nodes[node]['posis'])
            if d < R:
                neighs.append(a)
                G.add_edge(node, a) #so we can see which nodes are interacting
    
    return neighs


# ------------------------------------------------------------------------------
#   ENERGY FUNCTION
# ------------------------------------------------------------------------------
#  It alculates the energy of the graph embedded system of spins in array (negative sum over neighbors pairwise products, Heisenberg hamiltonian)
#
def energy_function(G,array,R):
  E=0
  N = G.number_of_nodes()
  adj_list = []      # Adjacence list, empty
  for a in range(N): #takes every node
    node_a = [] #another empty list
    node_a.append(a)
    for b in range(N):
        if a != b: 
            d = euclidean_distance(G.nodes[a]['posis'], G.nodes[b]['posis'])
            if (d < R) and (b < a): #so there are no repeated relations
                node_a.append(b)
                G.add_edge(a,b)
    adj_list.append(node_a)



  for i in range(N):    # Sum over nodes
    for j in adj_list[i]:    # Sum over nearest neighbors
      E = E - array[i,0]*array[int(j),0] # First component (it can be either -1, 0, or +1) 
      E = E - array[i,1]*array[int(j),1] # Second component ( it can be 0 or alpha)
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

# ----------------------------------------------------------------------------- 
#   MAIN
# ----------------------------------------------------------------------------
with open("RW_positions.txt", "r") as read_content:   # To read initial positions from a external file so they are always the same
    dictio = json.load(read_content)

keys_values = dictio.items()
#
# -----------------------------------------------------------------------------
#  States
# -----------------------------------------------------------------------------
#

#%%
start=time.time()
#
for rep in range(N_of_repetitions):


    positions_main = {int(key): value for key, value in keys_values}
    N = N0 #number of agents  at the beggining
    G_main = nx.Graph()
    orient = []
# For random position uncomment next lines #######################
#    positions = {}
#    
    for a in range(N): #creates the nodes and a dictionary that includes the position of every node
        G_main.add_node(a)
#        x = random.random() #we will have a 1x1 field
#        y = random.random()
#        pos = (x,y)
#        positions.update({a:pos})
##################################################################    
    nx.set_node_attributes(G_main,positions_main,'posis') #now every node has an associated position
    
    # Define the three possible states
    spin1=np.array([1,0]) #amunt
    spin0=np.array([0,alpha]) #el "neutre" 
    spin_1=np.array([-1,0]) #avall
    #
    # -----------------------------------------------------------------------------
    #
    steps_array = np.zeros(nsteps)  # Creates an array of zeros to store the energy at every step
    
    spins=np.random.randint(-1,2,N)  # Generates a random sequence of n opinion states
    #
    #   
    agents = dict(enumerate(spins)) # creates the dictionary to add the opinions to the nodes
    #
    nx.set_node_attributes(G_main,agents,'orientation')
    
    for trial in range(N_of_trials):  #this LOOPS OVER THE VELOCITIES. same initial configurations are used for every v (stored in G_main)
        v = vmin
        print('REP = ', rep, 'trial = ', trial, 'v = ',v)
        consent = False
        
        G = nx.Graph()
        G = G_main

        positions = positions_main
        
        num0=(spins[:]==0).sum()    # number of nodes with neutral opinion
        num1=(spins[:]==1).sum()    # number of nodes with rightist opinion
        num_1=(spins[:]==-1).sum()  # number of nodes with leftist opinion
        
        # Construction of initial vectors array for the energy calculation
        # This array is not modified during MC, so it is only useful for the
        # initial configuration
        #
        zeros=np.zeros(N) 
        vectors=np.stack((spins,zeros), axis=-1)  # Stack zeros to the array spins, building a (N, 2) matrix
        for i in range (N):  # Loop over rows
          if vectors[i,0]==0:  # If the first component of a row is 0
            vectors[i,1]=alpha # Then the second component is alpha
        
        energy=energy_function(G,vectors,R)  # Calculate the total energy using a Heisenberg hamiltonian extended over the neighbors of each node
    
        num_links = G_main.number_of_edges()
        N = G_main.number_of_nodes()
        ave_k = 2*num_links/N
        print('L = ', num_links, 'N = ', N, '<k> = ', ave_k)
        print('CONNECTED before remove ',nx.is_connected(G_main))
        print('INITIAL', energy, num1, num0, num_1)
        print ('........')
 ####################################################################################################################
        H = nx.Graph()
        H = G_main
#     Plot the graph 
#         fig.savefig(imagename)
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# Clean peripheria --------------------------------------
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
        spins = np.array(list((nx.get_node_attributes(G,'orientation')).values()))
        zeros=np.zeros(N)
        vectors=np.stack((spins,zeros), axis=-1)  # Stack zeros to the array spins, building a (N, 2) matrix
        for i in range (N):  # Loop over rows
          if vectors[i,0]==0:  # If the first component of a row is 0
            vectors[i,1]=alpha # Then the second component is alpha
        energy=energy_function(G,vectors,R)
#
        num_links = G.number_of_edges()
#
        ave_k = 2*num_links/N
        energy=energy_function(G,vectors,R)
#
        print('spins len  = ', len(spins), N)
#   
        num0=(spins[:]==0).sum()    # number of nodes with neutral opinion
        num1=(spins[:]==1).sum()    # number of nodes with rightist opinion
        num_1=(spins[:]==-1).sum()  # number of nodes with leftist opinion
#
        partition = community_louvain.best_partition(G)   # Modulariry
        mod = community_louvain.modularity(partition, G)
#
        print('INITIAL PERIPHERIA REMOVED', energy, num1, num0, num_1)
        print('L = ', num_links, 'N = ', N, '<k> = ', ave_k, 'Modularity best Louvain = ', mod)
        print(nx.is_connected(G))
        if (nx.is_connected(G) == False):
          print("NON_CONNECTED", rep)
          break
# ----------------------------------------------------------------------------------------
        #
        # MARKOV CHAIN. 
        #---------------------------------------------------------------------------
        #
        # 
        positions = positions_main
        for step in range (nsteps):                # Steps LOOP 
        
        #-------------------------------------------------------
        #           FIGURES (comment this for +20 steps)
        #-------------------------------------------------------
            # fig, ax = plt.subplots()
            # plt.title("configuration" + str(step))
            # plt.xlim(0, 1)     # set the xlim to left, right
            # plt.ylim(0, 1)     # set the xlim to left, right    
            
            # pos = G.nodes.data('posis')
            
            # for i in range (N):
            #   if G.nodes[i]['orientation'] == -1:   # RED
            #     nx.draw_networkx_nodes(G,pos,nodelist=[i],node_color='red',node_size=siz_nod)
            #   elif G.nodes[i]['orientation'] == 0:    #GREEN 
            #     nx.draw_networkx_nodes(G,pos,nodelist=[i],node_color='green',node_size=siz_nod)
            #   else:                           # BLUE
            #     nx.draw_networkx_nodes(G,pos,nodelist=[i],node_color='blue',node_size=siz_nod)
            
            # limits=plt.axis('on') # turns on axis
            # ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            # plt.savefig("configuration" + str(step))   
        #-------------------------------------------------------
            
            G.remove_edges_from(G.edges())
            G = one_random_walk_step(G)
            G_aux = nx.Graph.copy(G) #auxiliar graph. it will contain the NEW states
        
            for element in range (N):           # Every MC step has N flip attempts
              a=np.random.randint(N)            # no longer random
              a = element
              neigh_list = neighbors(a, R, G_aux)
         
              neighborhood = int(len(neigh_list))
              neigh_array = np.asarray(neigh_list)
            #
            #  Spin Changes and their associated energy change 
            #---------------------------------------------------------------------------
              if G.nodes[a]['orientation']==-1:           # Selected spin = -1
            #---------------------------------------------------------------------------
                diffenergy_10=energy_change(neigh_array,spin_1,spin0,G)  # -1 to 0
                p_10=np.exp(-(diffenergy_10)/temp) #probability to change
            #
                diffenergy_11=energy_change(neigh_array,spin_1,spin1,G)  # -1 to +1
                p_11=np.exp(-(diffenergy_11)/temp)
            #
                K_1=1/(p_10+p_11+1)     # Normalization constant
                p_10=K_1*p_10
                p_11=K_1*p_11
                
        #------------------------  Acceptance Criterium  ----------------------------
        #
                randunif=np.random.uniform(0.0,1.0) 
                if randunif<p_10: #the spin will change to 0
                  G_aux.nodes[a]['orientation']=0 
                  num0=num0+1
                  num_1=num_1-1
                  energy=energy+diffenergy_10 
                elif randunif<(p_10+p_11): #the spin will change to 1
                  G_aux.nodes[a]['orientation']=1
                  num1=num1+1
                  num_1=num_1-1
                  energy=energy+diffenergy_11
        #
        #---------------------------------------------------------------------------
              elif G.nodes[a]['orientation']==0:           # Selected spin = 0
        #---------------------------------------------------------------------------
                diffenergy0_1=energy_change(neigh_array,spin0,spin_1,G)  # 0 to -1
                p0_1=np.exp(-(diffenergy0_1)/temp)
        #
                diffenergy01=energy_change(neigh_array,spin0,spin1,G)     # 0 to +1
                p01=np.exp(-(diffenergy01)/temp)
        #
                K0=1/(p0_1+p01+1)     # Normalization constant
                p0_1=K0*p0_1
                p01=K0*p01
        #
        #------------------------  Acceptance Criterium  ----------------------------
        #
                randunif=np.random.uniform(0.0,1.0)
                if randunif<p0_1:
                  G_aux.nodes[a]['orientation']=-1
                  num0=num0-1
                  num_1=num_1+1
                  energy=energy+diffenergy0_1
                elif randunif<(p0_1+p01):
                  G_aux.nodes[a]['orientation']=1
                  num0=num0-1
                  num1=num1+1
                  energy=energy+diffenergy01
        #   
        #---------------------------------------------------------------------------
              else:                           # Selected spin = 1
        #---------------------------------------------------------------------------
                diffenergy10=energy_change(neigh_array,spin1,spin0,G)    # 1 to 0
                p10=np.exp(-(diffenergy10)/temp)
        #
                diffenergy1_1=energy_change(neigh_array,spin1,spin_1,G)  # 1 to -1
                p1_1=np.exp(-(diffenergy1_1)/temp)
        #
                K1=1/(p10+p1_1+1)     # Normalization constant
                p10=K1*p10
                p1_1=K1*p1_1
        #
        #------------------------  Acceptance Criterium  ----------------------------
        #
                randunif=np.random.uniform(0.0,1.0)
                if randunif<p10:
                  G_aux.nodes[a]['orientation']=0
                  num0=num0+1
                  num1=num1-1
                  energy=energy+diffenergy10
                elif randunif<(p10+p1_1):
                  G_aux.nodes[a]['orientation']=-1
                  num1=num1-1
                  num_1=num_1+1
                  energy=energy+diffenergy1_1
                  
                  
        #
            G = G_aux
            energy=energy_function(G,vectors,R)
            steps_array[step] = steps_array[step] + energy  # Store the energy in the corresponding step for every repetition 
            global_magnetization = num1 - num_1
            
#            if step%1000 == 0:


#
# -----------------------------------------------------------------------
# Plot the orientation of the last config
# -----------------------------------------------------------------------

#                fig, ax = plt.subplots()
#                plt.title(r"$step$ " + str(step))
#
#                pos = G.nodes.data('posis')
#
#                for i in range (N):
#                  if G.nodes[i]['orientation'] == -1:   # RED
#                     nx.draw_networkx_nodes(G,pos,nodelist=[i],node_color='red',node_size=siz_nod)
#                  elif G.nodes[i]['orientation'] == 0:    #GREEN 
#                    nx.draw_networkx_nodes(G,pos,nodelist=[i],node_color='green',node_size=siz_nod)
#                  else:                           # BLUE
#                    nx.draw_networkx_nodes(G,pos,nodelist=[i],node_color='blue',node_size=siz_nod)
#
#
#                nx.draw_networkx_edges(G,pos)
#                limits=plt.axis('on') # turns on axis
#                ax.set_aspect('equal', adjustable='box')
#                ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#
#
#
#
##                pos=nx.spring_layout(G)
##                for i in range (N):
##                    if G.nodes[i]['orientation'] == -1:   # RED
##                        nx.draw_networkx_nodes(G,G.nodes[i]['posis'],nodelist=[i],node_color='red',node_size=siz_nod)
##                    elif G.nodes[i]['orientation'] == 0:    #GREEN 
##                        nx.draw_networkx_nodes(G,G.nodes[i]['posis'],nodelist=[i],node_color='green',node_size=siz_nod)
##                    else:                           # BLUE
##                        nx.draw_networkx_nodes(G,G.nodes[i]['posis'],nodelist=[i],node_color='blue',node_size=siz_nod)
#                plt.show()
#
       
#                print("Repetition " + str(rep))
#                print("Velocity " + str(v))
#                print("Step " + str (step))
#                print("Active links " + str(G.number_of_edges()))
#                print("energy " + str (energy))
#                print("Global magnetization " + str(global_magnetization))
#                print("Neutral spins " + str(num0))
#                print("-------------------")
            #checking orientation
            
            if (num0 == N) or(num1 == N) or (num_1 == N): #consent
                print ("CONSENT ACHIVED AT STEP " + str(step))
                print ("ALL THE SPINS ARE " + str(G.nodes[0]['orientation']))
                velocity_list.append(v)
                consent_times.append(step)
                consent = True
                consent_file.write(str(rep)+' '+str(step)+' '+str(global_magnetization)+ '\n')
                break
      
        if consent == False:
            fails.write(str(v) + '\n')
            consent_file.write(str(rep)+' '+'NaN'+ '\n')
#
#
        print('v = ', v)
        v = vmin + trial*v_interval  # Increase velocity
#
end=time.time()                       # End measuring time
time_elapsed=end-start
print ('time elapsed',time_elapsed,'seconds')

consent_file.close()
fails.close()


#
# -----------------------------------------------------------------------
# Plot the orientation of the last config
# -----------------------------------------------------------------------

fig, ax = plt.subplots()
plt.title("Last configuration"+ "- step " + str(step))


pos = G.nodes.data('posis')

for i in range (N):
  if G.nodes[i]['orientation'] == -1:   # RED
    nx.draw_networkx_nodes(G,pos,nodelist=[i],node_color='red',node_size=siz_nod)
  elif G.nodes[i]['orientation'] == 0:    #GREEN 
    nx.draw_networkx_nodes(G,pos,nodelist=[i],node_color='green',node_size=siz_nod)
  else:                           # BLUE
    nx.draw_networkx_nodes(G,pos,nodelist=[i],node_color='blue',node_size=siz_nod)


nx.draw_networkx_edges(G,pos)
limits=plt.axis('on') # turns on axis
ax.set_aspect('equal', adjustable='box')
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)



"""
pos=nx.spring_layout(G)
for i in range (N):
  if G.nodes[i]['orientation'] == -1:   # RED
    nx.draw_networkx_nodes(G,G.nodes[i]['posis'],nodelist=[i],node_color='red',node_size=siz_nod)
  elif G.nodes[i]['orientation'] == 0:    #GREEN 
    nx.draw_networkx_nodes(G,G.nodes[i]['posis'],nodelist=[i],node_color='green',node_size=siz_nod)
  else:                           # BLUE
    nx.draw_networkx_nodes(G,G.nodes[i]['posis'],nodelist=[i],node_color='blue',node_size=siz_nod)
    
#
"""
#%%

llista_promitjos = []
llista_velocitats = []
llista_desvest = []

for a in range(N_of_trials):
    velocity = vmin + a*v_interval
    #print(velocity)
    llista_velocitats.append(velocity)
    indexes = [p for p, v in enumerate(velocity_list) if v== velocity]
    #print(indexes)
    suma = 0
    suma2 = 0
    for i in indexes:
        #print(consent_times[i])
        suma += consent_times[i]
    try:
      promig = suma/len(indexes)
    except ZeroDivisionError:
      print("No hi ha resultats per la velocitat " + str(velocity))
      promig = 0
    for i in indexes:
        suma2 += (consent_times[i] - promig)**2
    desvest = np.sqrt(suma2/N_of_repetitions)
    llista_desvest.append(desvest)
    llista_promitjos.append(promig)
    
file=open('RW'+'_N'+str(N)+'_a'+str(alpha)+'_T'+str(temp)+'_vmin'+str(vmin)+'_vmax'+str(vmax)+'.csv','w')

#  WRITING RESULTS ----------------------------------------------------------
for a in range (N_of_trials):
  file.write(str(llista_velocitats[a])+' '+str(llista_promitjos[a])+' '+str(llista_desvest[a])+ ' ' + str(G.nodes[0]['orientation'])+ '\n')  
             
file.close()

# PLOTTING RESULTS

plt.scatter(llista_velocitats, llista_promitjos, s=10)
plt.errorbar(llista_velocitats,llista_promitjos,yerr=llista_desvest, linestyle="none")
plt.xlabel('Velocity')
plt.ylabel('Time to consent')
plt.title ('Velocity vs time to consent. Alpha = ' + str(alpha))
plt.savefig('a' +str(alpha) + '_vmin'+str(vmin)+'_vmax'+str(vmax) + ".png")
#ax.set_aspect('equal', adjustable='box')

print(G)
