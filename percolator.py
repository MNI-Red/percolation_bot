import random
from copy import deepcopy
from util import Vertex
from util import Edge
from util import Graph
"""
You'll want to implement a smarter decision logic. This is skeleton code that you should copy and replace in your repository.
"""
class PercolationPlayer:

	def RemoveVertex(graph, v):
		graph.V.remove(v)
		to_remove = []
		for e in graph.E:
			if e.a is v or e.b is v:
				to_remove.append(e)
		graph.E.difference_update(to_remove)
		return graph

	def ComputeNeighbors(graph):
		to_ret = []
		for v in graph.V:
			to_ret.append(deepcopy(graph).RemoveVertex(v))
		

	def BFS(graph):
		pass
		frontier = [graph]
		discovered = {}
		while frontier:
			current = frontier.pop(0)


	def DegreeCalculator(graph, vertex):
		pass

	def ChooseVertexToColor(graph, player):
		temp_graph = deepcopy(graph)
		sorted([v for v in graph.V if v.color == -1])
		return random.choice([v for v in graph.V if v.color == -1])

	def ChooseVertexToRemove(graph, player):
		temp_graph = deepcopy(graph)
		return random.choice([v for v in graph.V if v.color == player])

# Feel free to put any personal driver code here.
def main():
	a = Vertex("a")
	b = Vertex("b")
	c = Vertex("c")
	e1 = Edge(a, b)
	e2 = Edge(a, c)
	V = {a, b, c} # the vertex set
	E = {e1, e2}
	G = Graph(V, E)
	# print(G)
	# PercolationPlayer.RemoveVertex(G, a)
	# print(G)

if __name__ == "__main__":
	main()
