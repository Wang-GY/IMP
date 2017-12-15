import numpy as np
import os
import time
import argparse


def read_seed_info(path):
    if os.path.exists(path):
        try:
            f = open(path, 'r')
            txt = f.readlines()
            seeds = list()
            for line in txt:
                seeds.append(int(line))
            return seeds
        except IOError:
            print
            'IOError'
    else:
        print
        'file can not found'


# read and analyse the data in the file to obtain a graph object
def read_graph_info(path):
    if os.path.exists(path):
        parents = {}
        children = {}
        edges = {}
        nodes = set()

        try:
            f = open(path, 'r')
            txt = f.readlines()
            header = str.split(txt[0])
            node_num = int(header[0])
            edge_num = int(header[1])

            for line in txt[1:]:
                row = str.split(line)

                src = int(row[0])
                des = int(row[1])
                nodes.add(src)
                nodes.add(des)

                if children.get(src) is None:
                    children[src] = []
                if parents.get(des) is None:
                    parents[des] = []

                weight = float(row[2])
                edges[(src, des)] = weight
                children[src].append(des)
                parents[des].append(src)

            return list(nodes), edges, children, parents, node_num, edge_num
        except IOError:
            print 'IOError'
    else:
        print 'file can not found'


def happen_with_prop(rate):
    rand = np.random.ranf()
    if rand <= rate:
        return True
    else:
        return False


class Graph:
    nodes = None
    edges = None
    children = None
    parents = None
    node_num = None
    edge_num = None

    def __init__(self, (nodes, edges, children, parents, node_num, edge_num)):
        self.nodes = nodes
        self.edges = edges
        self.children = children
        self.parents = parents
        self.node_num = node_num
        self.edge_num = edge_num

    def get_children(self, node):
        ch = self.children.get(node)
        if ch is None:
            self.children[node] = []
        return self.children[node]

    def get_parents(self, node):
        pa = self.parents.get(node)
        if pa is None:
            self.parents[node] = []
        return self.parents[node]

    def get_weight(self, src, dest):
        weight = self.edges.get((src, dest))
        if weight is None:
            return 0
        else:
            return weight

    # return true if node1 is parent of node 2 , else return false
    def is_parent_of(self, node1, node2):
        if self.get_weight(node1, node2) != 0:
            return True
        else:
            return False

    # return true if node1 is child of node 2 , else return false
    def is_child_of(self, node1, node2):
        return self.is_parent_of(node2, node1)

    def get_out_degree(self, node):
        return len(self.get_children(node))

    def get_in_degree(self, node):
        return len(self.get_parents(node))


# graph : Graph
# seed : list
# sample_num : int
def influence_spread_computation_IC(graph, seeds, sample_num=10000):
    influence = 0
    for i in range(sample_num):
        actived = seeds
        new_actived = actived
        while (len(new_actived) != 0):
            actived = new_actived
            new_actived = []
            for node in actived:
                node_children = graph.get_children(node)
                for j in range(graph.get_out_degree(node)):
                    if happen_with_prop(graph.get_weight(node, node_children[j])):
                        new_actived.append(node_children[j])
                        influence = influence + 1
    return int(influence / sample_num) + len(seeds)


def influence_spread_computation_LT(graph, seeds, sample_num=10000):
    influence = 0
    for i in range(sample_num):
        thresholds = np.random.rand(graph.node_num)
        actived = seeds
        new_actived = actived
        while (len(new_actived) != 0):
            actived = new_actived
            new_actived = []
            activity_vector = np.zeros(graph.node_num)
            for node in actived:
                node_children = graph.get_children(node)
                for child in node_children:
                    activity_vector[child - 1] = activity_vector[child - 1] + graph.get_weight(node, child)
            for i in range(graph.node_num):
                if activity_vector[i] >= thresholds[i]:
                    new_actived.append(i + 1)
                    influence = influence + 1
    return int(influence / sample_num) + len(seeds)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='CARP instance file', dest='graph_path')
    parser.add_argument('-s', help='seed set', dest='seed_path')
    parser.add_argument('-m', help='diffusion model', dest='model')
    parser.add_argument('-b',
                        help='specifies the termination manner and the value can only be 0 or 1. If it is set to 0, '
                             'the termination condition is as the same defined in your algorithm. Otherwise, '
                             'the maximal time budget specifies the termination condition of your algorithm.',
                        dest='type')
    parser.add_argument('-t', type=int, help='termination', dest='timeout')
    parser.add_argument('-r', type=int, help='random seed', dest='random_seed')

    args = parser.parse_args()
    graph_path = args.graph_path
    seed_path = args.seed_path
    model = args.model
    type = args.type
    timeout = args.timeout
    random_seed = args.random_seed

    np.random.seed(random_seed)

    graph = Graph(read_graph_info(graph_path))
    seeds = read_seed_info(seed_path)

    if model == 'IC':
        print(influence_spread_computation_IC(graph=graph, seeds=seeds, sample_num=10000))
    elif model == 'LT':
        print(influence_spread_computation_LT(graph=graph, seeds=seeds, sample_num=10000))
    else:
        print('Type err')
