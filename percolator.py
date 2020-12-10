import random
import time
import sys
import signal
from util import Vertex
from util import Edge
from util import Graph

"""
You'll want to implement a smarter decision logic. This is skeleton code that you should copy and replace in your repository.
"""
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
		incident_edges = PercolationPlayer.IncidentEdges(graph, v)
		neighbors = [e.a if e.a != v else e.b for e in incident_edges]
		if not neighbors:
			return 1 #disconnected vertices are high priority because they can never be removed by opponent
		score = 0
		for v in neighbors:
			if v.color == -1:
				score += 1
			elif v.color == player:
				score -= 2
			else:
				score += 3
		return score

	def heuristic_one(graph, v, player):
		incident_edges = PercolationPlayer.IncidentEdges(graph, v)
		neighbors = [e.a if e.a != v else e.b for e in incident_edges]
		if not neighbors:
			return -1
		score = 0
		for v in neighbors:
			if v.color == player:
				score -= 2
			else:
				score += 4
		return score

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

	def heuristic_three(graph, v, player):
		incident_edges = PercolationPlayer.IncidentEdges(graph, v)
		neighbors = [e.a if e.a != v else e.b for e in incident_edges]
		if not neighbors:
			return -1
		score = 0
		for v in neighbors:
			if v.color != player:
				score += 4
		return score

	def graph_value(graph, player):
		if not graph.V:
			return sys.maxsize
		scores = [PercolationPlayer.remove_heuristic(graph, v, player) for v in graph.V if v.color == player]
		return sum(scores)

	def count_nonzero(graph):
		return sum(sum(col>0 for col in row) for row in graph)

	def MiniMaxWIP(graph_dict, removed_dict, current_matrix_dict, current_graph, player, vertices, depth = 2, maximizing = True):
		currents_indices = [v.index for v in vertices if v.color == player if v.index in list(current_matrix_dict.keys())]
		value = PercolationPlayer.adjacency_graph_value(current_graph, currents_indices, current_matrix_dict)

		if depth == 0 or PercolationPlayer.count_nonzero(current_graph) == 0:
			return value
		
		current_key = tuple(tuple(row) for row in current_graph)
		if maximizing:
		# value = -1000000
			for child, child_matrix_index_dict, removed in PercolationPlayer.AdjacencyNeighbors(current_graph, 
																player, currents_indices, current_matrix_dict):
				child_key = tuple(tuple(row) for row in child)
				graph_dict[child_key] = current_key
				removed_dict[child_key] = removed
				value = max(value, PercolationPlayer.MiniMaxWIP(graph_dict, removed_dict, child_matrix_index_dict, child, abs(1-player), vertices, 
										depth-1, not maximizing))
			return value
		else: #minimizing player
		# value = 1000000
			for child, child_matrix_index_dict, removed in PercolationPlayer.AdjacencyNeighbors(current_graph, 
																player, currents_indices, current_matrix_dict):
				child_key = tuple(tuple(row) for row in child)
				graph_dict[child_key] = current_key
				removed_dict[child_key] = removed
				value = min(value, PercolationPlayer.MiniMaxWIP(graph_dict, removed_dict, child_matrix_index_dict, child, abs(1-player), vertices, 
										depth-1, not maximizing))
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
		neighbors = [i for i in np.flatnonzero(graph[v_index])]
		score = 0
		for v in neighbors:
			if v in uncolored_indices:
				score += 1
			elif v in my_indices:
				score -= 2
			else:
				score += 3
		return score

	def adjacency_remove_heuristic(graph, v_index, my_indices, vertex_to_index):
		# neighbors = [i for i in graph[v_index]]
		# print(v_index)
		# print(vertex_to_index)
		index_to_vertex = dict(zip(list(vertex_to_index.values()), list(vertex_to_index.keys())))
		index_to_remove = vertex_to_index[v_index]
		neighbor_indices = [i for i, e in enumerate(graph[index_to_remove]) if e != 0]
		# print(neighbor_indices)

		if len(neighbor_indices) == 0:
			return -1

		score = 0
		# print(my_indices)
		for v in neighbor_indices:
			# print(v, my_indices)
			# print(v in my_indices)
			if index_to_vertex[v] in my_indices:
				score -= 2
			else:
				score += 4
		return score

	def adjacency_graph_value(graph, my_indices, vertex_to_index):
		# print(my_indices)
		if PercolationPlayer.count_nonzero(graph) == 0:
			return sys.maxsize
		if len(my_indices) == 0:
			return -sys.maxsize
		potential = sorted([v for v in my_indices], 
				key = lambda v: PercolationPlayer.adjacency_remove_heuristic(graph, v, my_indices, vertex_to_index))
		# print(potential)
		# if len(potential) == 0:
		# 	print(my_indices, vertex_to_index)
		return potential[-1]
		# return sum(potential)

	def reconstruct(graph_dict, removed_dict):
		temp = list(graph_dict.keys())[-1]
		path = []
		while graph_dict[temp]:
			path.append(removed_dict[temp])
			temp = graph_dict[temp]
		# print(removed_dict[graph_dict[list(removed_dict.keys())[-1]]])
		if len(path) <= 0:
			print(graph_dict)
			print(removed_dict)
		# print(path)
		# print(removed_dict[path[-2]])
		# print([removed_dict[i] for i in path])
		return path[-1]

	def ChooseVertexToColor(graph, player):
		# temp_graph = deepcopy(graph)
		# print([v for v in graph.V if v.color == -1])
		potential = sorted([v for v in graph.V if v.color == -1], 
			key = lambda v: PercolationPlayer.color_heuristic(graph, v, player)) 
		# print(potential)
		# sorted([v for v in graph.V if v.color == -1])
		return potential[-1]

	def ChooseVertexToRemove(graph, player):
		potential = sorted([v for v in graph.V if v.color == player], 
			key = lambda v: PercolationPlayer.heuristic_one(graph, v, player))
		
		# print(potential)
		# vertices = graph.V
		# my_indices = [v.index for v in vertices if v.color == player]
		# adjacency_graph, vertex_to_index = PercolationPlayer.GraphToAdjacency(graph)
		# other =[PercolationPlayer.adjacency_remove_heuristic(adjacency_graph, v, my_indices, vertex_to_index) for v in my_indices]
		# print([PercolationPlayer.heuristic_one(graph, v, player) for v in vertices if v.color == player], other)

		# return potential[-1]

		try:
			with PercolationPlayer.Timeout(seconds = 0.490):
				if not graph.E:# or len(graph.V) > 30:
					# potential = sorted([v for v in graph.V if v.color == player], 
					# 	key = lambda v: PercolationPlayer.heuristic_one(graph, v, player))
					# print(potential)
					return potential[-1]

				num_nodes = len(graph.V)
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
				PercolationPlayer.MiniMaxWIP(graph_dict, removed_dict, vertex_to_index, adjacency_graph, player, vertices, depth)
				return PercolationPlayer.GetVertex(graph, PercolationPlayer.reconstruct(graph_dict, removed_dict))
				
		except TimeoutError as e:
			# potential = sorted([v for v in graph.V if v.color == player], 
			# key = lambda v: PercolationPlayer.heuristic_one(graph, v, player))
			return potential[-1]

	# def RemoveVertex(graph, value):
	# 	v = GetVertex(graph, value)
	# 	graph.V.remove(v)
	# 	graph.E.difference_update(IncidentEdges(graph, v))
	# 	# print(graph)
	# 	connected_vertices = {e.a for e in graph.E}
	# 	connected_vertices.update({e.b for e in graph.E})
	# 	# print(connected_vertices)
	# 	graph.V =  connected_vertices
	# 	# print(graph)
	# 	return graph, v
	
	# def CountVertices(graph):
	# 	zeroes, ones = 0, 0
	# 	for v in graph.V:
	# 		if v.color == 0:
	# 			zeroes += 1
	# 		else:
	# 			ones += 1
	# 	return zeroes, ones

	# def ComputeNeighbors(graph, player):
	# 	to_ret = []
	# 	# print(graph.V)
	# 	for v in graph.V:
	# 		if v.color == player:
	# 			to_ret.append(PercolationPlayer.RemoveVertex(deepcopy(graph), v.index))
	# 	return to_ret

	# def BFS(graph, player, max_depth = 10, me = True):
	# 	game_states = {graph:me}
	# 	depth_per_state = {graph:0}
	# 	path = {graph:None}
	# 	frontier = [graph]
	# 	discovered = set()
	# 	# print(player)
	# 	# print(graph)
	# 	while frontier:
	# 		current = frontier.pop(0)
	# 		# print(current.V)
	# 		my_move = game_states[current]
	# 		depth = depth_per_state[current]
	# 		discovered.add(current)
	# 		# print(current)
	# 		if not my_move:
	# 			player = 1 - player

	# 		if not bool(current.V) and not my_move:
	# 			reconstruct = []
	# 			while path[current]:
	# 				reconstruct.append(current)
	# 				current = path[current]
	# 			# print(reconstruct[-1].V, graph.V)
	# 			# print(graph.V.difference(reconstruct[-1].V))

	# 			to_ret = [x for x in [v.index for v in graph.V] if x not in [v.index for v in reconstruct[-1].V]][0]
	# 			# print(to_ret)
	# 			# for v in graph.V:
	# 			# 	if v.index == to_ret
	# 			return [v for v in graph.V if v.index == to_ret[0]][0]

	# 		if max_depth+1 in depth_per_state.keys():
	# 			last_row = [i for i in depth_per_state if depth_per_state[i] == max_depth]
	# 			last_row_counts = [(PercolationPlayer.CountVertices(g)) for g in last_row]

	# 			if player == 0:
	# 				scores = [score[0]-score[1] for score in last_row_counts]
	# 			else:
	# 				scores = [score[1]-score[0] for score in last_row_counts]

	# 			game_to_score = dict(zip(last_row, scores))
	# 			game_to_score = {k: v for k, v in sorted(game_to_score.items(), key=lambda item: item[1])}

	# 			to_ret = [x for x in [v.index for v in graph.V] if x not in [v.index for v in next(iter(game_to_score))]][0]
	# 			# print(to_ret)
	# 			# for v in graph.V:
	# 			# 	if v.index == to_ret
	# 			return [v for v in graph.V if v.index == to_ret[0]][0]

	# 		for i in PercolationPlayer.ComputeNeighbors(current, player):
	# 			if i not in discovered:
	# 				frontier.append(i)
	# 				discovered.add(i)
	# 				path[i] = current
	# 				game_states[i] = not my_move
	# 				depth[i] = depth + 1
		
	# def MiniMax(graph, depth, maximizing, player):
	# 	if depth == 0 or not graph[-1].V:
	# 		return (PercolationPlayer.graph_value(graph[-1], player), graph)
	# 	value = PercolationPlayer.graph_value(graph[-1], player)
	# 	if maximizing:
	# 	# value = -1000000
	# 		for child in PercolationPlayer.ComputeNeighbors(graph[-1], player):
	# 			value, graph = max((value, graph), PercolationPlayer.MiniMax(graph+[child], depth-1, False, 1-player))
	# 			# graph.append(max_graph)
	# 		return value, graph
	# 	else: #minimizing player
	# 	# value = 1000000
	# 		for child in PercolationPlayer.ComputeNeighbors(graph[-1], player):
	# 			value, graph = min((value, graph), PercolationPlayer.MiniMax(graph+[child], depth-1, True, 1-player))
	# 			# graph.append(min_graph)
	# 		return value, graph


# import numpy as np
# Feel free to put any personal driver code here.
def main():
	a = Vertex(0, 1)
	b = Vertex(1, 0)
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
	print(vertex_to_index)
	print()
	# PercolationPlayer.remove_vertex(adjacency, c.index, vertex_to_index)
	# print(vertex_to_index)

	# for i in adjacency:
	# 	print(i)
	# print(vertex_to_index)
	
	vertices = G.V
	my_indices = [v.index for v in vertices if v.color == player]
	print(my_indices, "\n")

	print("Remove a:")
	print(PercolationPlayer.adjacency_remove_heuristic(adjacency, 0, my_indices, vertex_to_index))
	print(PercolationPlayer.heuristic_one(G, a, player))

	print("\nRemove c:")
	print(PercolationPlayer.adjacency_remove_heuristic(adjacency, 2, my_indices, vertex_to_index))
	print(PercolationPlayer.heuristic_one(G, c, player))



	# neighbors = PercolationPlayer.AdjacencyNeighbors(adjacency, player, my_indices, vertex_to_index)
	# for pair in neighbors:
	# 	for i in pair[0]:
	# 		print(i)
	# 	print(pair[1])
	# 	print("removed: " + str(pair[2]))
	# 	print()
	# # print(neighbors)
	# for i in neighbors:
	# 	print(i)
	# 	print()
	# graph_key = tuple(tuple(row) for row in adjacency)
	# graph_dict = {graph_key:None}
	# removed_dict = {graph_key:None}
	# PercolationPlayer.MiniMaxWIP(graph_dict, removed_dict, vertex_to_index, adjacency, player, vertices)
	# print(PercolationPlayer.ChooseVertexToRemove(G, player))
	# path = []
	# temp = list(graph_dict.items())[-1][0]
	# # print(graph_dict)
	# # print(removed_dict)
	# while graph_dict[temp]:
	# 	path.append(temp)
	# 	temp = graph_dict[temp]
	# # reversed(path)
	# path.append(temp)
	# path = list(reversed(path))
	# # print()
	# print(path)
	# print([removed_dict[i] for i in path])

	# start = time.time()
	# neighbors = PercolationPlayer.ComputeNeighbors(G, player)
	# print("G neighbors time: " + str(time.time()-start)) 

	# 	for j in i:
	# 		print(j)
	# 	print()
	# print(PercolationPlayer.ChooseVertexToColor(G, player))
	# to_remove = PercolationPlayer.ChooseVertexToRemove(G, player)
	# # print()
	# PercolationPlayer.RemoveVertex(G, to_remove)
	# PercolationPlayer.RemoveVertex(G, 3)
	# to_remove = PercolationPlayer.ChooseVertexToRemove(G, player)
	# # PercolationPlayer.RemoveVertex(G, "a")
	# print(G)
	# value, graph = PercolationPlayer.MiniMax([G], 2, True, 1)
	# print(graph)
	# print(graph[0].V, graph[-1].V)
	# graph_dict = {G:None}
	# removed_dict = {G:None}
	# import time
	# start = time.time()
	# value = PercolationPlayer.MiniMaxWIP(graph_dict, removed_dict, G, player)
	# # print(graph_dict)
	# print("MiniMax time: " + str(time.time()-start))
	# start = time.time()
	# # for g in graph_dict.keys():
	# # 	print(g, PercolationPlayer.graph_remove_heuristic(g, player))
	# # 	player = abs(1-player)
	# # print(removed_dict)
	# path = []
	# temp = list(graph_dict.items())[-1][0]
	# while graph_dict[temp]:
	# 	path.append(temp)
	# 	temp = graph_dict[temp]
	# # reversed(path)
	# path.append(temp)
	# path = list(reversed(path))
	# # print()
	# # print(path)
	# remove_order = [removed_dict[g] for g in path]
	# print(remove_order)
	# print("MiniMax time: " + str(time.time()-start))

	
if __name__ == "__main__":
	main()
