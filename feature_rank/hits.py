import networkx as nx
import matplotlib.pyplot as plt

G=nx.DiGraph()

edges = [(0, 2),
         (0, 4),
         (0, 6),
         (1, 4),
         (1, 5),
         (1, 6),
         (2, 3),
         (2, 4),
         (3, 0),
         (3, 1),
         (3, 5),
         (3, 6),
         (4, 5),
         (4, 8),
         (6, 5),
         (6, 3),
         (6, 1),
         (7, 0),
         (7, 5),
         (7, 8),
         (8, 0),
         (8, 5),
         (8, 4)]
G.add_edges_from(edges)

h,a=nx.hits(G)
nx.draw(G,with_labels=True)

#plt.show() # display
print(h,a)

pr = nx.pagerank(G)
print(pr)