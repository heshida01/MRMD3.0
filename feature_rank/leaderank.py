import networkx as nx
import matplotlib.pyplot as plt

def leaderrank(graph):
    # graph = nx.DiGraph()
    # graph.add_edges_from(edges)

    #节点个数
    num_nodes = graph.number_of_nodes()
    # 节点
    nodes = graph.nodes()

    graph.add_node(0)
    for node in nodes:
        graph.add_edge(0, node)

    print("graph.edges:",graph.edges())

    # LR值初始化
    LR = dict.fromkeys(nodes, 1.0)
    LR[0] = 0.0

    # 迭代从而满足停止条件
    while True:

        tempLR = {}
        for node1 in graph.nodes():
            s = 0.0
            for node2 in graph.nodes():
                if node2 in graph.neighbors(node1):
                    s += 1.0 / graph.degree([node2])[node2] * LR[node2]
            tempLR[node1] = s
        # 终止条件:LR值不在变化

        error = 0.0
        for n in tempLR.keys():
            error += abs(tempLR[n] - LR[n])
        print(error)
        if error <= 0.00001:
            break
        LR = tempLR
    # 节点g的LR值平均分给其它的N个节点并且删除节点
    avg = LR[0] / num_nodes
    LR.pop(0)
    for k in LR.keys():
        LR[k] += avg
    return  LR


if __name__ == '__main__':

    LR = leaderrank()
    #print(sorted(leaderrank(graph).items(), key=lambda item: item[1]))
    print(sorted(LR.items(), key=lambda item: item[1]))

