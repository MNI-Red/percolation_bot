import random
import time
import sys
from copy import deepcopy
from util import Vertex
from util import Edge
from util import Graph
from util import IncidentEdges, GetVertex
import numpy as np

"""
You'll want to implement a smarter decision logic. This is skeleton code that you should copy and replace in your repository.
"""
class PercolationPlayer:

	def RemoveVertex(graph, value):
		v = GetVertex(graph, value)
		graph.V.remove(v)
		graph.E.difference_update(IncidentEdges(graph, v))
		# print(graph)
		connected_vertices = {e.a for e in graph.E}
		connected_vertices.update({e.b for e in graph.E})
		# print(connected_vertices)
		graph.V =  connected_vertices
		# print(graph)
		return graph, v
	
	def CountVertices(graph):
		zeroes, ones = 0, 0
		for v in graph.V:
			if v.color == 0:
				zeroes += 1
			else:
				ones += 1
		return zeroes, ones

	def ComputeNeighbors(graph, player):
		to_ret = []
		# print(graph.V)
		for v in graph.V:
			if v.color == player:
				to_ret.append(PercolationPlayer.RemoveVertex(deepcopy(graph), v.index))
		return to_ret

	def BFS(graph, player, max_depth = 10, me = True):
		game_states = {graph:me}
		depth_per_state = {graph:0}
		path = {graph:None}
		frontier = [graph]
		discovered = set()
		# print(player)
		# print(graph)
		while frontier:
			current = frontier.pop(0)
			# print(current.V)
			my_move = game_states[current]
			depth = depth_per_state[current]
			discovered.add(current)
			# print(current)
			if not my_move:
				player = 1 - player

			if not bool(current.V) and not my_move:
				reconstruct = []
				while path[current]:
					reconstruct.append(current)
					current = path[current]
				# print(reconstruct[-1].V, graph.V)
				# print(graph.V.difference(reconstruct[-1].V))

				to_ret = [x for x in [v.index for v in graph.V] if x not in [v.index for v in reconstruct[-1].V]][0]
				# print(to_ret)
				# for v in graph.V:
				# 	if v.index == to_ret
				return [v for v in graph.V if v.index == to_ret[0]][0]

			if max_depth+1 in depth_per_state.keys():
				last_row = [i for i in depth_per_state if depth_per_state[i] == max_depth]
				last_row_counts = [(PercolationPlayer.CountVertices(g)) for g in last_row]

				if player == 0:
					scores = [score[0]-score[1] for score in last_row_counts]
				else:
					scores = [score[1]-score[0] for score in last_row_counts]

				game_to_score = dict(zip(last_row, scores))
				game_to_score = {k: v for k, v in sorted(game_to_score.items(), key=lambda item: item[1])}

				to_ret = [x for x in [v.index for v in graph.V] if x not in [v.index for v in next(iter(game_to_score))]][0]
				# print(to_ret)
				# for v in graph.V:
				# 	if v.index == to_ret
				return [v for v in graph.V if v.index == to_ret[0]][0]

			for i in PercolationPlayer.ComputeNeighbors(current, player):
				if i not in discovered:
					frontier.append(i)
					discovered.add(i)
					path[i] = current
					game_states[i] = not my_move
					depth[i] = depth + 1
		
	def MiniMax(graph, depth, maximizing, player):
		if depth == 0 or not graph[-1].V:
			return (PercolationPlayer.graph_value(graph[-1], player), graph)
		value = PercolationPlayer.graph_value(graph[-1], player)
		if maximizing:
		# value = -1000000
			for child in PercolationPlayer.ComputeNeighbors(graph[-1], player):
				value, graph = max((value, graph), PercolationPlayer.MiniMax(graph+[child], depth-1, False, 1-player))
				# graph.append(max_graph)
			return value, graph
		else: #minimizing player
		# value = 1000000
			for child in PercolationPlayer.ComputeNeighbors(graph[-1], player):
				value, graph = min((value, graph), PercolationPlayer.MiniMax(graph+[child], depth-1, True, 1-player))
				# graph.append(min_graph)
			return value, graph

	def color_heuristic(graph, v, player):
		incident_edges = IncidentEdges(graph, v)
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

	def remove_heuristic(graph, v, player):
		incident_edges = IncidentEdges(graph, v)
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

	def graph_value(graph, player):
		if not graph.V:
			return sys.maxsize
		scores = [PercolationPlayer.remove_heuristic(graph, v, player) for v in graph.V if v.color == player]
		return sum(scores)

	def MiniMaxWIP(graph_dict, removed_dict, current_matrix_dict, current_graph, player, vertices, depth = 2, maximizing = True):
		currents_indices = [v.index for v in vertices if v.color == player if v.index in list(current_matrix_dict.keys())]
		value = PercolationPlayer.adjacency_graph_value(current_graph, currents_indices, current_matrix_dict)

		if depth == 0 or np.count_nonzero(current_graph) == 0:
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
		matrix_to_vertex = dict(zip(indices, range(n)))
		# print(matrix_to_vertex)
		adjacency = [[0]*n for _ in range(n)]
		# print(adjacency)
		for e in graph.E:
			a = matrix_to_vertex[e.a.index]
			b = matrix_to_vertex[e.b.index]
			adjacency[a][b] = 1
			adjacency[b][a] = 1
		# import numpy as np
		return adjacency, matrix_to_vertex

	def remove_vertex(graph, index, matrix_to_vertex):
		# n
		copy = [[j for j in i] for i in graph]
		index_to_remove = matrix_to_vertex[index]
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
		keys = list(matrix_to_vertex.keys())
		keys.remove(index)
		matrix_to_vertex = dict(zip(keys, range(len(copy))))
		return copy, matrix_to_vertex, index

	def AdjacencyNeighbors(graph, player, my_indices, matrix_to_vertex):
		to_ret = []

		for index in my_indices:
			to_ret.append(PercolationPlayer.remove_vertex(graph, index, matrix_to_vertex))
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

	def adjacency_remove_heuristic(graph, v_index, my_indices, matrix_to_vertex):
		# neighbors = [i for i in graph[v_index]]
		# print(v_index)
		# print(matrix_to_vertex)
		index_to_remove = matrix_to_vertex[v_index]
		neighbor_indices = [i for i, e in enumerate(graph[index_to_remove]) if e != 0]
		score = 0
		# print(my_indices)
		for v in neighbor_indices:
			# print(v, my_indices)
			# print(v in my_indices)
			if v in my_indices:
				score -= 2
			else:
				score += 4
		return score

	def adjacency_graph_value(graph, my_indices, matrix_to_vertex):
		# print(my_indices)
		if np.count_nonzero(graph) == 0:
			return sys.maxsize
		scores = [PercolationPlayer.adjacency_remove_heuristic(graph, v, my_indices, matrix_to_vertex) for v in my_indices]
		return sum(scores)

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

		if not graph.E:# or len(graph.V) > 30:
			potential = sorted([v for v in graph.V if v.color == player], 
				key = lambda v: PercolationPlayer.remove_heuristic(graph, v, player))
			# print(potential)
			return potential[-1]
		num_nodes = len(graph.V)
		if num_nodes > 25:
			depth = 2
		elif num_nodes > 17:
			depth = 3
		elif num_nodes > 9:
			depth = 4
		else:
			depth = 5

		vertices = graph.V
		n = len(vertices)
		
		adjacency_graph, matrix_to_vertex = PercolationPlayer.GraphToAdjacency(graph)
		graph_key = tuple(tuple(row) for row in adjacency_graph)
		graph_dict = {graph_key:None}
		removed_dict = {graph_key:None}
		PercolationPlayer.MiniMaxWIP(graph_dict, removed_dict, matrix_to_vertex, adjacency_graph, player, vertices, depth)
		return GetVertex(graph, PercolationPlayer.reconstruct(graph_dict, removed_dict))

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

	adjacency, matrix_to_vertex = PercolationPlayer.GraphToAdjacency(G)
	print(matrix_to_vertex)
	print()
	# PercolationPlayer.remove_vertex(adjacency, c.index, matrix_to_vertex)
	# print(matrix_to_vertex)

	# for i in adjacency:
	# 	print(i)
	# print(matrix_to_vertex)
	
	vertices = G.V
	my_indices = [v.index for v in vertices if v.color == player]
	print(my_indices, "\n")

	# print(PercolationPlayer.adjacency_remove_heuristic(adjacency, 2, my_indices))
	# print(PercolationPlayer.adjacency_remove_heuristic(adjacency, 0, my_indices))


	# neighbors = PercolationPlayer.AdjacencyNeighbors(adjacency, player, my_indices, matrix_to_vertex)
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
	graph_key = tuple(tuple(row) for row in adjacency)
	graph_dict = {graph_key:None}
	removed_dict = {graph_key:None}
	# PercolationPlayer.MiniMaxWIP(graph_dict, removed_dict, matrix_to_vertex, adjacency, player, vertices)
	print(PercolationPlayer.ChooseVertexToRemove(G, player))
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
