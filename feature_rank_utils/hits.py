import networkx as nx
import matplotlib.pyplot as plt

G=nx.DiGraph()

edges = [(2, 0),
         (4, 0),
         (6, 0),
]
G.add_edges_from(edges)

h,a=nx.hits(G)
#nx.draw(G,with_labels=True)

#plt.show() # display
print(h,a)

pr = nx.pagerank(G)
print(pr)