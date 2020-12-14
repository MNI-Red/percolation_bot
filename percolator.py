import random
import time
import sys
import signal
from util import Vertex
from util import Edge
from util import Graph
from collections import defaultdict

class PercolationPlayer:

	class Timeout:
	    def __init__(self, seconds=0.5, error_message="Timeout of {0} seconds hit"):
	        self.seconds = seconds
	        self.error_message = error_message.format(seconds)
	    def handle_timeout(self, signum, frame):
	        raise TimeoutError(self.error_message)
	    def __enter__(self):
	        signal.signal(signal.SIGALRM, self.handle_timeout)
	        signal.setitimer(signal.ITIMER_REAL, self.seconds)
	    def __exit__(self, type, value, traceback):
	        signal.alarm(0)

	def GetVertex(graph, i):
	    for v in graph.V:
	        if v.index == i:
	            return v
	    return None

	def IncidentEdges(graph, v):
	    return [e for e in graph.E if (e.a == v or e.b == v)]

	def color_heuristic(graph, v, player):
		vertices = {vert for vert in graph.V}
		vertices.remove(v)
		score = 0
		for v in vertices:
			incident_edges = PercolationPlayer.IncidentEdges(graph, v)
			neighbors = [e.a if e.a != v else e.b for e in incident_edges]
			if not neighbors:
				score += 1
				break
			for n in neighbors:
				if n.color == -1:
					score += 1
				elif n.color == player:
					score += 2
				else:
					score -= 4

		return score

	#68.225
	def heuristic_one(graph, v, player):
		vertices = {vert for vert in graph.V}
		vertices.remove(v)
		score = 0
		for v in vertices:
			incident_edges = PercolationPlayer.IncidentEdges(graph, v)
			neighbors = [e.a if e.a != v else e.b for e in incident_edges]
			if not neighbors:
				score += 1
				break
			for n in neighbors:
				if n.color == player:
					score += 2
				else:
					score -= 4

		return score

	#60.325
	def heuristic_two(graph, v, player):
		incident_edges = PercolationPlayer.IncidentEdges(graph, v)
		neighbors = [e.a if e.a != v else e.b for e in incident_edges]
		if not neighbors:
			return -1
		score = 0
		for v in neighbors:
			if v.color == player:
				score -= 2
		return score

	#62.825
	def heuristic_three(graph, v, player):
		vertices = {vert for vert in graph.V}
		vertices.remove(v)
		score = 0
		for v in vertices:
			score += len(PercolationPlayer.IncidentEdges(graph, v))

		return score
	#52.65
	def heuristic_four(graph, v_in, player):
		mine = [v for v in graph.V if v.color == player and v is not v_in]
		opponents = [v for v in graph.V if v.color != player]
		return sum(PercolationPlayer.heuristic_one(graph, v, player) for v in mine) - sum(PercolationPlayer.heuristic_one(graph, v, abs(1-player)) for v in opponents)

	#51.175
	def heuristic_five(graph, v_in, player):
		mine = [v for v in graph.V if v.color == player and v is not v_in]
		opponents = [v for v in graph.V if v.color != player]
		return len(mine)-len(opponents)

	def graph_value(graph, player):
		if not graph.V:
			return sys.maxsize
		scores = [PercolationPlayer.heuristic_one(graph, v, player) for v in graph.V if v.color == player]
		return sum(scores)

	def count_nonzero(graph):
		return sum(sum(col>0 for col in row) for row in graph)

	def MiniMaxWIP(graph_dict, removed_dict, graph_values, current_matrix_dict, current_graph, player, vertices, depth = 2, maximizing = True):
		currents_indices = [v.index for v in vertices if v.color == player if v.index in list(current_matrix_dict.keys())]
		value = PercolationPlayer.adjacency_graph_value(current_graph, currents_indices, current_matrix_dict)
		current_key = tuple(tuple(row) for row in current_graph)
		graph_values[current_key] = value
		
		if depth == 0 or PercolationPlayer.count_nonzero(current_graph) == 0:
			return value
			
		if maximizing:
		# value = -1000000
			for child, child_matrix_index_dict, removed in PercolationPlayer.AdjacencyNeighbors(current_graph, 
																player, currents_indices, current_matrix_dict):
				child_key = tuple(tuple(row) for row in child)
				graph_dict[child_key] = current_key
				removed_dict[child_key] = removed
				value = max(value, PercolationPlayer.MiniMaxWIP(graph_dict, removed_dict, graph_values, child_matrix_index_dict, 
					child, abs(1-player), vertices, depth-1, not maximizing))
			return value
		else: #minimizing player
		# value = 1000000
			for child, child_matrix_index_dict, removed in PercolationPlayer.AdjacencyNeighbors(current_graph, 
																player, currents_indices, current_matrix_dict):
				child_key = tuple(tuple(row) for row in child)
				graph_dict[child_key] = current_key
				removed_dict[child_key] = removed
				value = min(value, PercolationPlayer.MiniMaxWIP(graph_dict, removed_dict, graph_values, child_matrix_index_dict, 
					child, abs(1-player), vertices, depth-1, not maximizing))
			return value

	def GraphToAdjacency(graph):
		indices = [i.index for i in graph.V]
		n = len(indices)
		vertex_to_index = dict(zip(indices, range(n)))
		# print(vertex_to_index)
		adjacency = [[0]*n for _ in range(n)]
		# print(adjacency)
		for e in graph.E:
			a = vertex_to_index[e.a.index]
			b = vertex_to_index[e.b.index]
			adjacency[a][b] = 1
			adjacency[b][a] = 1
		
		return adjacency, vertex_to_index

	def remove_vertex(graph, index, vertex_to_index):
		# n
		copy = [[j for j in i] for i in graph]
		index_to_remove = vertex_to_index[index]
		copy[index_to_remove] = ["*"]*len(copy)
		for x in copy:
			x[index_to_remove] = "*"
		# for i in copy:
		# 	print(i)
		# print()
		copy = [[j for j in i if type(j) is int] for i in copy]
		copy = [i for i in copy if i != []]
		# for i in copy:
		# 	print(i)
		keys = list(vertex_to_index.keys())
		keys.remove(index)
		vertex_to_index = dict(zip(keys, range(len(copy))))
		return copy, vertex_to_index, index

	def AdjacencyNeighbors(graph, player, my_indices, vertex_to_index):
		to_ret = []

		for index in my_indices:
			to_ret.append(PercolationPlayer.remove_vertex(graph, index, vertex_to_index))
		return to_ret

	def adjacency_color_heuristic(graph, v_index, my_indices, uncolored_indices):
		index_to_vertex = dict(zip(list(vertex_to_index.values()), list(vertex_to_index.keys())))
		index_to_remove = vertex_to_index[v_index]
		neighbor_indices = [i for i, e in enumerate(graph[index_to_remove]) if e != 0]

		if len(neighbor_indices) == 0:
			return -1

		score = 0
		for v in neighbor_indices:
			if index_to_vertex[v] in uncolored_indices:
				score += 1
			if index_to_vertex[v] in my_indices:
				score -= 2
			else:
				score += 4
		return score
	#52.1
	def adjacency_heuristic_one(graph, my_indices, vertex_to_index):
		# neighbors = [i for i in graph[v_index]]
		# print(v_index)
		# print(vertex_to_index)
		index_to_vertex = dict(zip(list(vertex_to_index.values()), list(vertex_to_index.keys())))
		# index_to_remove = vertex_to_index[v_index]
		# neighbor_indices = [i for i, e in enumerate(graph[index_to_remove]) if e != 0]
		# print(neighbor_indices)

		score = 0
		# print(my_indices)
		for v in vertex_to_index:
			# print(v, my_indices)
			# print(v in my_indices)
			if v in my_indices:
				score -= 2
			else:
				score += 4
		return score

	def adjacency_heuristic_two(graph, my_indices, vertex_to_index):
		# neighbors = [i for i in graph[v_index]]
		# print(v_index)
		# print(vertex_to_index)
		# print(my_indices)
		index_to_vertex = dict(zip(list(vertex_to_index.values()), list(vertex_to_index.keys())))
		

		score = 0
		# print(my_indices)
		for v in vertex_to_index:
			# print(v, my_indices)
			# print(v in my_indices)
			if v in my_indices:
				score += 1
			else:
				score -= 2
		return score

	def adjacency_heuristic_three(graph, v_index, my_indices, vertex_to_index):
		# neighbors = [i for i in graph[v_index]]
		# print(v_index)
		# print(vertex_to_index)
		index_to_vertex = dict(zip(list(vertex_to_index.values()), list(vertex_to_index.keys())))
		index_to_remove = vertex_to_index[v_index]
		neighbor_indices = [i for i, e in enumerate(graph[index_to_remove]) if e != 0]
		# print(neighbor_indices)

		if len(neighbor_indices) == 0:
			return -1

		score = 1
		# print(my_indices)
		for v in neighbor_indices:
			# print(v, my_indices)
			# print(v in my_indices)
			if index_to_vertex[v] not in my_indices:
				score += 4 
		return score

	def adjacency_heuristic_five(my_indices, vertex_to_index):
		# neighbors = [i for i in graph[v_index]]
		# print(v_index)
		# print(vertex_to_index)
		# index_to_vertex = dict(zip(list(vertex_to_index.values()), list(vertex_to_index.keys())))
		# index_to_remove = vertex_to_index[v_index]
		# neighbor_indices = [i for i, e in enumerate(graph[index_to_remove]) if e != 0]
		# print(neighbor_indices)
		return  len(my_indices)- len(vertex_to_index)

	def adjacency_graph_value(graph, my_indices, vertex_to_index):
		# print(my_indices)
		if PercolationPlayer.count_nonzero(graph) == 0:
			return (sys.maxsize, sys.maxsize, sys.maxsize)
		if len(my_indices) == 0:
			return (-sys.maxsize, -sys.maxsize, -sys.maxsize)
		
		return (PercolationPlayer.adjacency_heuristic_five(my_indices, vertex_to_index), 
				PercolationPlayer.adjacency_heuristic_two(graph, my_indices, vertex_to_index), 
				PercolationPlayer.adjacency_heuristic_one(graph, my_indices, vertex_to_index))
		# print(potential)
		# if len(potential) == 0:
		# 	print(my_indices, vertex_to_index)
		# return potential[-1]
		# return sum(potential)

	def reverse_dict(dict_in):
		to_ret = defaultdict(list)
		for key in dict_in:
			to_ret[dict_in[key]].append(key)
		return to_ret

	def reconstruct(graph_dict, removed_dict, graph_values, depth):
		reversed_graph_dict = PercolationPlayer.reverse_dict(graph_dict)
		temp = list(graph_dict.keys())[0]
		graph_path = []
		path = []
		while len(path) < depth and len(reversed_graph_dict[temp]) > 0:
			path.append(removed_dict[temp])
			graph_path.append(temp)
			# print(temp, reversed_graph_dict[temp])
			temp = max(reversed_graph_dict[temp], key = lambda x: graph_values[x])
		path.append(removed_dict[temp])
		graph_path.append(temp)
		# print(removed_dict[graph_dict[list(removed_dict.keys())[-1]]])
		# if len(path) <= 0:
		# print(graph_dict)
		# print(removed_dict)
		# print(path)
		# print(removed_dict[path[-2]])
		# print([removed_dict[i] for i in path])
		if len(path) <= 1:
			# print(graph_dict)
			# print(graph_values)
			# print(path)
			return None
		return path[1]
		# return path, graph_path

	def ChooseVertexToColor(graph, player):
		potential = sorted([v for v in graph.V if v.color == -1], 
			key = lambda v: (PercolationPlayer.heuristic_one(graph, v, player), 
				PercolationPlayer.heuristic_five(graph, v, player),
				PercolationPlayer.color_heuristic(graph, v, player)
				)) 
		return potential[-1]

	def ChooseVertexToRemove(graph, player):
		potential = sorted([v for v in graph.V if v.color == player], 
			key = lambda v: (PercolationPlayer.heuristic_one(graph, v, player), 
				PercolationPlayer.heuristic_five(graph, v, player),
				PercolationPlayer.heuristic_three(graph, v, player)
				))
		# print(potential)
		# vertices = graph.V
		# my_indices = [v.index for v in vertices if v.color == player]
		# adjacency_graph, vertex_to_index = PercolationPlayer.GraphToAdjacency(graph)
		# other =[PercolationPlayer.adjacency_remove_heuristic(adjacency_graph, v, my_indices, vertex_to_index) for v in my_indices]
		# print([PercolationPlayer.heuristic_one(graph, v, player) for v in vertices if v.color == player], other)
		return potential[-1]

		#this code has yet to achive a better wr than the one above
		try:
			with PercolationPlayer.Timeout(seconds = 0.490):
				num_nodes = len(graph.V)
				if not graph.E and num_nodes > 30:
					# potential = sorted([v for v in graph.V if v.color == player], 
					# 	key = lambda v: PercolationPlayer.heuristic_one(graph, v, player))
					# print(potential)
					return potential[-1]

				
				if num_nodes > 25:
					depth = 2
				elif num_nodes > 15:
					depth = 3
				else:
					depth = 4

				vertices = graph.V
				n = len(vertices)
				
				adjacency_graph, vertex_to_index = PercolationPlayer.GraphToAdjacency(graph)
				graph_key = tuple(tuple(row) for row in adjacency_graph)
				graph_dict = {graph_key:None}
				removed_dict = {graph_key:None}
				graph_values = {graph_key: None}
				PercolationPlayer.MiniMaxWIP(graph_dict, removed_dict, graph_values, vertex_to_index, adjacency_graph, player, 
					vertices, depth)
				vertex =  PercolationPlayer.reconstruct(graph_dict, removed_dict, graph_values, depth)
				if not vertex:
					return potential[-1]
				return PercolationPlayer.GetVertex(graph, vertex)
				
		except TimeoutError as e:
			# potential = sorted([v for v in graph.V if v.color == player], 
			# key = lambda v: PercolationPlayer.heuristic_one(graph, v, player))
			return potential[-1]

def print_dict(dict_in):
	for i in dict_in:
		print(str(i) + " -- " + str(dict_in[i]))

def reverse_dict(dict_in):
	to_ret = defaultdict(list)
	for key in dict_in:
		to_ret[dict_in[key]].append(key)
	return to_ret

def main():
	a = Vertex(0, 0)
	b = Vertex(1, 1)
	c = Vertex(2, 1)
	d = Vertex(3, 0)
	e1 = Edge(a, b)
	e2 = Edge(a, c)
	e3 = Edge(b, d)
	V = {a, b, c, d} # the vertex set
	E = {e1, e2, e3}
	G = Graph(V, E)
	player = 1
	# print(G)

	# start = time.time()
	# G_copy = deepcopy(G)
	# PercolationPlayer.RemoveVertex(G_copy, 0)
	# print("G remove vertex time: " + str(time.time()-start))

	adjacency, vertex_to_index = PercolationPlayer.GraphToAdjacency(G)
	# print(vertex_to_index)
	# print()
	# PercolationPlayer.remove_vertex(adjacency, c.index, vertex_to_index)
	# print(vertex_to_index)

	# for i in adjacency:
	# 	print(i)
	# print(vertex_to_index)
	# print(PercolationPlayer.ChooseVertexToColor(G, player))
	vertices = G.V
	my_indices = [v.index for v in vertices if v.color == player]
	# print(my_indices, "\n")

	# print("Remove a:")
	# print(PercolationPlayer.adjacency_remove_heuristic(adjacency, 0, my_indices, vertex_to_index))
	# print(PercolationPlayer.heuristic_one(G, a, player))

	# print("\nRemove c:")
	# print(PercolationPlayer.adjacency_remove_heuristic(adjacency, 2, my_indices, vertex_to_index))
	# print(PercolationPlayer.heuristic_one(G, c, player))

	depth = 3
	n = len(vertices)
	
	graph_key = tuple(tuple(row) for row in adjacency)
	graph_dict = {graph_key:0}
	removed_dict = {graph_key:None}
	graph_values = {graph_key: None}
	PercolationPlayer.MiniMaxWIP(graph_dict, removed_dict, graph_values, vertex_to_index, adjacency, player, 
		vertices, PercolationPlayer.adjacency_heuristic_three, depth)
	graph_values[graph_key] = None
	
	for i in graph_dict:
		print(str(removed_dict[i]) + " -- " + str(graph_values[i]) + " -- " + str(i))

	# reversed_graph = reverse_dict(graph_dict)
	# print("\nreversed dict")
	# print_dict(reversed_graph)

	# print()
	# path = PercolationPlayer.reconstruct(graph_dict, removed_dict, graph_values, depth)
	# print("\n", path)
	# for i in graph_path:
	# 	print(str(i) + " -- " + str(graph_values[i]) + " -- " + str(removed_dict[i]))


if __name__ == "__main__":
	main()
