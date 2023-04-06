# from graphs import getGraph
# from PageRank import PageRank
# from TrustRank import TrustRank
from feature_rank_utils.PageRank.PageRank import PageRank
from feature_rank_utils.PageRank.graphs import getGraph
from feature_rank_utils.PageRank.TrustRank import TrustRank
import sys

def run(edge_file, node_num, beta=0.85, epsilon=1e-6, max_iterations=20):
	"""Calls various ranking functions and print the rank_vectors.
	
	
	Parameters
	----------
	edge_file : string
		Path to the file where edges of web-graph are stored.

	node_num : int
		Number of nodes in the web-graph.
	
	beta : float, optional
		Probability with which teleports will occur.
		Default value : 0.85
	
	epsilon : float, optional
		A small value and total error in ranks should be less than epsilon.
		Default value : 1e-6
	
	max_iterations : int, optional
		Maximum number of times to apply power iteration.
		Default value : 20

	
	Returns
	-------
	None

	"""
	gg = getGraph(edge_file)
	edges = gg.get_connections()

	#print("got edges...")

	pr = PageRank(beta, edges, epsilon, max_iterations, node_num)
	PageRank_vector = pr.pageRank()


	tr = TrustRank(beta, edges, epsilon, max_iterations, node_num, 
		PageRank_vector)
	TrustRank_vector = tr.trustRank()
	#print(TrustRank_vector, sum(TrustRank_vector))
	return TrustRank_vector

if __name__ == '__main__':
	location_of_the_edge_file = "./data/test"
	number_of_nodes_in_web_graph = 9

	# location_of_the_edge_file = "./data/WikiTalk.data"
	# number_of_nodes_in_web_graph = 2394385
	run(location_of_the_edge_file, number_of_nodes_in_web_graph)