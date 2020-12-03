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

	def MiniMaxWIP(graph_dict, removed_dict, current_graph, player, vertices, depth = 2, maximizing = True):
		currents_indices = [v.index for v in vertices if v.color == player]
		
		if depth == 0 or np.count_nonzero(current_graph) == 0:
			return PercolationPlayer.adjacency_graph_value(current_graph, currents_indices)
		
		value = PercolationPlayer.adjacency_graph_value(current_graph, currents_indices)
		if maximizing:
		# value = -1000000
			for child, removed in PercolationPlayer.AdjacencyNeighbors(current_graph, currents_indices, player):
				graph_dict[child.tobytes()] = tuple(current_graph)
				removed_dict[child.tobytes()] = removed
				value = max(value, PercolationPlayer.MiniMaxWIP(graph_dict, removed_dict, child, abs(1-player), vertices, 
										depth-1, not maximizing))
			return value
		else: #minimizing player
		# value = 1000000
			for child, removed in PercolationPlayer.AdjacencyNeighbors(current_graph, currents_indices, player):
				graph_dict[child.tobytes()] = current_graph.tobytes()
				removed_dict[child.tobytes()] = removed
				value = min(value, PercolationPlayer.MiniMaxWIP(graph_dict, removed_dict, child, abs(1-player), vertices, 
										depth-1, not maximizing))
			return value

	def GraphToAdjacency(graph):
		n = len(graph.V)
		adjacency = [[0]*n for _ in range(n)]
		# print(adjacency)
		for e in graph.E:
			adjacency[int(e.a.index)][int(e.b.index)] = 1
			adjacency[int(e.b.index)][int(e.a.index)] = 1
		# import numpy as np
		return np.array(adjacency)

	def AdjacencyNeighbors(graph, my_indices, player):
		to_ret = []
		for i in my_indices:
			if np.sum(graph[i]) != 0:
				to_mult = np.ones(len(graph))
				to_mult[i] = 0 
				neighbor = np.multiply(np.multiply(graph, to_mult), np.array([to_mult]).T)
				to_ret.append((neighbor, i))
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

	def adjacency_remove_heuristic(graph, v_index, my_indices):
		neighbors = [i for i in np.flatnonzero(graph[v_index])]
		score = 0
		# print(my_indices)
		for v in neighbors:
			# print(v, my_indices)
			# print(v in my_indices)
			if v in my_indices:
				score -= 2
			else:
				score += 4
		return score

	def adjacency_graph_value(graph, my_indices):
		# print(my_indices)
		if np.count_nonzero(graph) == 0:
			return sys.maxsize
		scores = [PercolationPlayer.adjacency_remove_heuristic(graph, v, my_indices) for v in my_indices]
		return sum(scores)

	def reconstruct(graph_dict, removed_dict):
		temp = list(graph_dict.keys())[-1]
		path = []
		while graph_dict[temp]:
			path.append(graph_dict[temp])
			temp = graph_dict[temp]
		print(removed_dict[path[1]])

	def ChooseVertexToColor(graph, player):
		# temp_graph = deepcopy(graph)
		# print([v for v in graph.V if v.color == -1])
		potential = sorted([v for v in graph.V if v.color == -1], 
			key = lambda v: PercolationPlayer.color_heuristic(graph, v, player)) 
		# print(potential)
		# sorted([v for v in graph.V if v.color == -1])
		return potential[-1]

	def ChooseVertexToRemove(graph, player):

		# potential = sorted([v for v in graph.V if v.color == player], 
		# 	key = lambda v: PercolationPlayer.remove_heuristic(graph, v, player))
		# # print(potential)
		# return potential[-1]
		
		vertices = graph.V
		n = len(vertices)
		graph = PercolationPlayer.GraphToAdjacency(graph)
		graph_dict = {graph.tobytes():None}
		removed_dict = {graph.tobytes():None}
		PercolationPlayer.MiniMaxWIP(graph_dict, removed_dict, graph, player, vertices)
		# PercolationPlayer.reconstruct(graph_dict, removed_dict)

		# print(graph_dict)
		print(removed_dict)

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

	# start = time.time()
	# adjacency = PercolationPlayer.GraphToAdjacency(G)
	# print(adjacency)
	# print()
	# print("adjacency remove vertex time: " + str(time.time()-start))
	# byte =  adjacency.tobytes()
	# print(byte)
	# y = np.frombuffer(byte, dtype = int).reshape(4,4)
	# print(y)
	# for e in G.E:
	# 	print((e.a.index, e.b.index))

	# adjacency = PercolationPlayer.GraphToAdjacency(G)
	# print(adjacency)
	# for i in adjacency:
	# 	print(i)
	# my_indices = [v.index for v in G.V if v.color == player]
	# print(my_indices)
	# print(2 in my_indices)
	
	# # print(adjacency[0][:])

	# start = time.time()
	# neighbors = PercolationPlayer.AdjacencyNeighbors(adjacency, my_indices)
	# print("adjacency neighbors time: " + str(time.time()-start)) 

	# # print(neighbors)
	# for i in neighbors:
	# 	print(i)
	# 	print()

	# start = time.time()
	# neighbors = PercolationPlayer.ComputeNeighbors(G, player)
	# print("G neighbors time: " + str(time.time()-start)) 

	# 	for j in i:
	# 		print(j)
	# 	print()
	# print(PercolationPlayer.ChooseVertexToColor(G, player))
	print(PercolationPlayer.ChooseVertexToRemove(G, player))
	# PercolationPlayer.RemoveVertex(G, "a")
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
