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
nsteps = N
vmin = 0.006 #initial velocity module
vmax = 0.007
siz_nod = 40
L = 1 #IMPLEMENTAR QUE ES PUGUI CANVIAR LA L?¿
N_of_trials = 1
N_of_repetitions = 100
#
v_interval = (vmax-vmin)/N_of_trials
#
prob_file = open('second_neigh_RW'+'_N'+str(N)+'_v'+"%5.3f"%vmin+'_non_elastic.csv','w')
#
#%% Random walk function
def one_random_walk_step(G):
# elastic
    N =  G.number_of_nodes()
    for a in range(N):
#        
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
###############################################################
def one_random_walk_step(v,positions,N):
# non-elastic
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


###############################################################



#%% Function to check the distance

def euclidean_distance(pos1,pos2): #both pos1 and pos2 are tuples (x,y)
    x = pos2[0] - pos1[0]
    y = pos2[1] - pos1[1]
    d = np.sqrt(x**2+y**2)
    return d 
#
##############################################################
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
#
##############################################################
def second_neighbors(N, R, G):
    second_neigh = []
    for i in range(N):
        neigh_list = neighbors(i, R, G)
        second_neigh_i = []
        for element in neigh_list:
          second_neigh_i_element = neighbors(element, R, G)
          second_neigh_i.append(second_neigh_i_element)
        flat_second_neigh_i = [item for sublist in second_neigh_i for item in sublist]
        flat_second_neigh_i_unique = [*set(flat_second_neigh_i)]
        flat_second_neigh_i_unique = [ele for ele in flat_second_neigh_i_unique if ele not in neigh_list] # Remove first neigh
        if (i in flat_second_neigh_i_unique):
          self = flat_second_neigh_i_unique.index(i)
#          print(i, flat_second_neigh_i_unique[self])
          flat_second_neigh_i_unique.pop(self)
        second_neigh.append(flat_second_neigh_i_unique)
    return second_neigh
#
##############################################################
def build_graph(G,R):
  N = G.number_of_nodes()
  for a in range(N): #takes every node
    for b in range(N):
        if a != b:
            d = euclidean_distance(G.nodes[a]['posis'], G.nodes[b]['posis'])
            if (d < R) and (b < a): #so there are no repeated relations
                G.add_edge(a,b)
  return G
#
##############################################################
#
def plot_G(G):
# Plot the graph 
    fig, ax = plt.subplots(1)
    fig.set_size_inches(9, 9)
#    ax.set_title('N = '+str(N)+' - k = '+str(ave_k)+r' - $\alpha = $'+str(alpha) + ' - T = '+str(temp), fontsize = 10    )
#    ax.text(0.9, 0.9, 't = '+str(step))
    new_dict = {}
    for a in range(N):
      new_dict[a] = G.nodes[a]['posis']
    positions = new_dict
#    print(positions)
    nx.draw_networkx_nodes(G,positions, node_size = 1)  
    nx.draw_networkx_edges(G,positions)
    nx.draw_networkx_labels(G,positions) 
#    plt.tight_layout
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, grid_linestyle=':', labelsize = 10)
    ax.set_xlim(0.00,1.00)
    ax.set_ylim(0.00,1.00)
    ax.grid(b=True, which='major', color='black', linestyle='dotted')
    fig.savefig(str(step)+'.png')
    plt.show()
    return
## ----------------------------------------------------------------------------- 
#   MAIN
# ----------------------------------------------------------------------------
with open("RW_positions.txt", "r") as read_content:   # To read initial positions from a external file so they are always the same
    dictio = json.load(read_content)

keys_values = dictio.items()
#
start=time.time()
#
for rep in range(N_of_repetitions):
    positions_main = {int(key): value for key, value in keys_values}
    N = N0 #number of agents  at the beggining
    G_main = nx.Graph()
#    
    for a in range(N): #creates the nodes and a dictionary that includes the position of every node
        G_main.add_node(a)
    nx.set_node_attributes(G_main,positions_main,'posis') #now every node has an associated position
    build_graph(G_main,R)  # place the edges
    # -----------------------------------------------------------------------------

    step = 0
#    print('plot G_main before removing peripheria')
#    print(G_main.nodes[0]['posis'])
#    plot_G(G_main)
#
    for trial in range(N_of_trials):  #this LOOPS OVER THE VELOCITIES. same initial configurations are used for every v (stored in G_main)
        v = vmin
        print('REP = ', rep, 'trial = ', trial, 'v = ',v)
        consent = False
        
#        G = nx.Graph()
#        G = G_main
        positions = positions_main
#        
        num_links = G_main.number_of_edges()
        N = G_main.number_of_nodes()
        ave_k = 2*num_links/N
        print('L = ', num_links, 'N = ', N, '<k> = ', ave_k)
        print('CONNECTED before remove ',nx.is_connected(G_main))
        print ('........')
 ####################################################################################################################
        H = nx.Graph()
        H = G_main
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
        num_links = G.number_of_edges()
        ave_k = 2.0*num_links/N
#
#
        partition = community_louvain.best_partition(G)   # Modulariry
        mod = community_louvain.modularity(partition, G)
#
        print('INITIAL PERIPHERIA REMOVED')
        print('L = ', num_links, 'N = ', N, '<k> = ', ave_k, 'Modularity best Louvain = ', mod)
        print(nx.is_connected(G))
        if (nx.is_connected(G) == False):
          print("NON_CONNECTED", rep)
          break
        print('plot G after removing peripheria')
#        plot_G(G)
# ----------------------------------------------------------------------------------------
        #
        # MOVEMENT 
        #---------------------------------------------------------------------------
        #
        Prob_matrix = np.zeros((N, nsteps))
        # 
        positions = positions_main
        for step in range (nsteps):                # Steps LOOP 
#           
            second_neighbors_list = second_neighbors(N, R, G)  # previous Second Neighbors (list of N lists)
##            G.remove_edges_from(G.edges())  # Remove previous edges  ELASTIC CASE
#            for a in range (N):
#              print(a, G.nodes[a]['posis'])
# ************************************************************************
# ************************ MOVE NOVES ************************************************ 
##            G = one_random_walk_step(G)     # Move nodes ******************************* ELASTIC CASE
#            print(' make a step -------------')
#            for a in range (N):



            positions = one_random_walk_step(v,positions,N)

            nx.set_node_attributes(G,positions,'posis') #now every node has the NEW associated position

            build_graph(G,R)               # build new edges
#            print(second_neighbors_list) 
#            print('plot G, rep = ', rep, ' step = ', step)
#            plot_G(G)
            for a in range (N):           # Every MC step has N flip attempts
              neigh_list = neighbors(a, R, G)   # current First Neighbors
#              print(a, neigh_list)
              counter_tot = len(second_neighbors_list[a])  # counter for the number of nodes which were sencond neighs of "a" in the previous step
#              print(a, neigh_list, counter_tot)
              counter_P = 0   # Counter for the number of nodes which were sencond neighs of "a" in the previous step AND first in this one
              for second_neigh in second_neighbors_list[a]:
                if second_neigh in neigh_list:
#                  print('YES ******** ', a, second_neigh)
                  counter_P = counter_P+1
              if (counter_tot != 0):
                Prob_matrix[a, step] = counter_P/counter_tot
              else:
                Prob_matrix[a, step] = 0
                 
#        print(Prob_matrix) 
        ave_prob = Prob_matrix.mean()
        prob_file.write(str(ave_prob)+'\n')
#        print('ave_prob = ', ave_prob)
#
# 
        print('v = ', v)
#        print('######## END OF REP ', rep, ' ###################')
        v = vmin + trial*v_interval  # Increase velocity
#
end=time.time()                       # End measuring time
time_elapsed=end-start
print ('time elapsed',time_elapsed,'seconds')

prob_file.close()


