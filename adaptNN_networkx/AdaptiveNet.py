import networkx as nx
import numpy as np
import copy
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

'''
DiGraph:
    node index: ( , )
    edge index: ((,),(,))
'''
def AdaptiveNet_gen(L = 6, Periodic_BC = True):
    net = nx.DiGraph()
    N = L*L
    # add nodes
    nodes_list = [(i,j) for i in range(L) for j in range(L)]
    net.add_nodes_from(nodes_list)
    assert(net.order() == N)
    
    if Periodic_BC:
        Left_edgelists = [[(i,j),(i+1,(j-1)%L)] for i in range(L-1) for j in range(L)]
        Right_edgelists = [[(i,j),(i+1,(j+1)%L)] for i in range(L-1) for j in range(L)]
    else:
        Left_edgelists = [[(i,j),(i+1,(j-1)%L)] for i in range(L-1) for j in range(1,L)]
        Right_edgelists = [[(i,j),(i+1,(j+1)%L)] for i in range(L-1) for j in range(L-1)]

    Down_edgelists = [[(i,j),(i+1,j)] for i in range(L-1) for j in range(L)]
    net.add_edges_from(Down_edgelists)
    net.add_edges_from(Left_edgelists)
    net.add_edges_from(Right_edgelists) 

    return net


'''
Draw:
    Generate pos with Graphviz:pos = graphviz_layout(lattice)
    
    networkx draw_[layout]: [spectral, shell, spring, circular, kamada_kawai, random]
    e.g. nx.draw_shell(lattice)
'''

def draw_adaptive_net(net,t,draw_Ws=False,save_fig = True):
    fig=plt.figure(figsize = [18,18])
    
    N = net.order()
    L = int(np.sqrt(N))
    net_pos = {(i,j):((j-L//2)*100,(L//2-i)*100) for i in range(L) for j in range(L)}
    # net_pos = graphviz_layout(net,prog = 'neato',args="-Goverlap=false")     
    # nx.draw_networkx(net,pos=net_pos, with_labels = False)
    
    if draw_Ws:
        Ws = nx.get_edge_attributes(net,'weight')
        for k,v in Ws.items():
            Ws[k] = int(100*Ws[k])
        nx.draw_networkx_edges(net,pos=net_pos)
        nx.draw_networkx_edge_labels(G=net, pos=net_pos, edge_labels=Ws, \
                                     font_size = 8, font_color = 'blue', label_pos = 0.6)
    # draw states
    states = nx.get_node_attributes(net,'is_fire')
    color_list = ['r' if v == 1 else 'grey' for v in states.values()]
    nx.draw_networkx_nodes(G=net, pos=net_pos, node_size=80*64//L, node_color=color_list)

    if save_fig:
        figname = 'net'+str(t)
        fig.savefig(figname)