import networkx as nx
import numpy as np
import copy
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import AdaptiveNet

import logging
import CreateLog

np.random.seed(2017)
class ConditioningTask(object):
	def __init__(self,network):
		self.A0 = 1
		self.noise0 = 0
		self.T = 0.35
		self.r = 0.01

		self.net = network
		self.L = int(np.sqrt(network.order()))
		self.input_nodes = [(0,self.L//2+2)]
		self.desired_output_node = (self.L-1,8)
		self.to_fire_outputs = []


	def normalize_Ws(self,net):
		Ws = nx.get_edge_attributes(net,'weight')
		L = self.L

		for j in net:
			if j in [(L-1, ii) for ii in range(L)]:
				continue
			sum_w = sum([Ws[(j,i)] for i in net.successors(j)])
			[Ws[(j,i)] for i in net.successors(j)] /= sum_w

		nx.set_edge_attributes(net,'weight',Ws)

	def initialize(self, net):
	    # random weights
	    for (u,v,w) in self.net.edges(data=True):
		    w['weight'] = np.random.random()

	    # all nodes unfire except input nodes
	    for n in net:
	        net.node[n]['is_fire'] = 0
	    for i in self.input_nodes:
	        net.node[i]['is_fire'] = 1
	    self.normalize_Ws(net)

	def get_new_states(self, net):
		states = nx.get_node_attributes(net,'is_fire')
		Ws = nx.get_edge_attributes(net,'weight')

		new_states = {}
		step = lambda v: int(v>self.T)
		for i in net:
			new_states[i] = step(sum([states[j]*Ws[(j,i)] for j in net.predecessors(i)]))

		for i in self.input_nodes:
			new_states[i] = 1
		return new_states

	def iterate_once(self, net):
		Ws = nx.get_edge_attributes(net,'weight')
		new_states = self.get_new_states(net)
		states = nx.get_node_attributes(net,'is_fire')

		output_nodes=[(self.L-1,j) for j in range(self.L)]
		self.to_fire_outputs = [i for i in output_nodes if new_states[i] == 1]

		# Fire too much -> upper threshold
		A = len(self.to_fire_outputs)
		self.T += np.sign(A-self.A0) * 0.01/self.L

		if A == 0 or self.desired_output_node in self.to_fire_outputs:
			self.r = 0.01 
		else:
			self.r = -0.1

		for i in net:
			if i in [(0, jj) for jj in range(self.L)]:
				continue
			for j in net.predecessors(i):
				noise = self.noise0*np.random.random()
				Ws[(j,i)] +=  (self.r * Ws[(j,i)]*(1-Ws[(j,i)])+ noise)*new_states[i]*states[j]

		nx.set_edge_attributes(net,'weight',Ws)
		self.normalize_Ws(net)

		nx.set_node_attributes(net,'is_fire',new_states)


	def iterate(self,tmax = 30000):
		self.initialize(self.net)
		my_logger,my_fh = CreateLog.create_log('firing')

		for t in range(tmax):
			self.iterate_once(self.net)

			my_logger.info('------------------t='+str(t)+'------------------')
			my_logger.info("num of firing_outputs: "+str(len(self.to_fire_outputs)))
			my_logger.info(str(self.r))
			my_logger.info("Threshold: "+str(self.T))

			if (t-1)%1000 == 0:
			    AdaptiveNet.draw_adaptive_net(self.net,t,draw_Ws=False)
		my_fh.close()  

if __name__ == "__main__":
	adaptive_net = AdaptiveNet.AdaptiveNet_gen(L = 64,Periodic_BC = True)
	task1 = ConditioningTask(adaptive_net)
	task1.iterate()
