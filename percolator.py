import random
from copy import deepcopy
from util import Vertex
from util import Edge
from util import Graph
from util import IncidentEdges
"""
You'll want to implement a smarter decision logic. This is skeleton code that you should copy and replace in your repository.
"""
class PercolationPlayer:

	def RemoveVertex(graph, value):
		for copied_v in graph.V:
			if copied_v.index == value:
				v = copied_v
		graph.V.remove(v)
		# print(graph)
		# to_remove = []
		# for e in graph.E:
		# 	if e.a is v or e.b is v:
		# 		to_remove.append(e)
		graph.E.difference_update(IncidentEdges(graph, v))
		return graph
	
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
		
	# def MiniMax(graph, depth, maximizing):
	# 	pass
	# 	if depth == 0 or not graph.V:
	#         to_ret = -1
	#         if maximizing:
	#         	to_ret = 1
	#         return to_ret, graph
	#     if maximizingPlayer:
	#         value = −1000000
	#         for child in ComputeNeighbors(graph)
	#             value = max(value, minimax(child, depth − 1, False))
	#         return value
	#     else #minimizing player
	#         value = 1000000
	#         for child in ComputeNeighbors(graph)
	#             value = min(value, minimax(child, depth − 1, True))
	#         return value

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
				score += 3
		return score

	def ChooseVertexToColor(graph, player):
		temp_graph = deepcopy(graph)
		# print([v for v in graph.V if v.color == -1])
		potential = sorted([v for v in graph.V if v.color == -1], 
			key = lambda v: PercolationPlayer.color_heuristic(graph, v, player)) 
		# print(potential)
		# sorted([v for v in graph.V if v.color == -1])
		return potential[-1]

	def ChooseVertexToRemove(graph, player):
		# if len(graph.V) > 1:
		# 	return PercolationPlayer.BFS(graph, player)
		# else:
		# temp_graph = deepcopy(graph)

		potential = sorted([v for v in graph.V if v.color == player], 
			key = lambda v: PercolationPlayer.remove_heuristic(graph, v, player)) 
		# print(potential)
		return potential[-1]

# Feel free to put any personal driver code here.
def main():
	a = Vertex("a", 1)
	b = Vertex("b", 0)
	c = Vertex("c", 1)
	e1 = Edge(a, b)
	e2 = Edge(a, c)
	V = {a, b, c} # the vertex set
	E = {e1, e2}
	G = Graph(V, E)
	# print(G)
	# print(PercolationPlayer.ChooseVertexToColor(G, 1))
	print(PercolationPlayer.ChooseVertexToRemove(G, 1))
	# PercolationPlayer.RemoveVertex(G, a)
	# print(G)

if __name__ == "__main__":
	main()
