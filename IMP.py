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


def read_graph_info(path):
    if os.path.exists(path):
        try:
            f = open(path, 'r')
            txt = f.readlines()
            header = str.split(txt[0])
            node_num = int(header[0])
            edge_num = int(header[1])
            adjacent_matrix = np.zeros((node_num, node_num))
            for line in txt[1:]:
                row = str.split(line)
                adjacent_matrix[int(row[0]) - 1][int(row[1]) - 1] = float(row[2])
            return adjacent_matrix, node_num, edge_num
        except IOError:
            print
            'IOError'
    else:
        print
        'file can not found'


def happen_with_prop(rate):
    rand = np.random.ranf()
    if rand <= rate:
        return True
    else:
        return False


def print_seeds(seeds):
    for seed in seeds:
        print(seed)


class Graph:
    adjacent_matrix = 0  # np.array i,j represent the weight of edge(i+1,j+1)
    node_num = 0  # number of nodes
    # edge_num = 0# number of edges
    children_buffer = {}  # buffer result
    parents_buffer = {}

    def __init__(self, adjacent_matrix):
        self.adjacent_matrix = adjacent_matrix
        self.node_num = np.shape(adjacent_matrix)[0]
        # self.edge_num = edge_num

    def get_children(self, node):
        children = self.children_buffer.get(node)
        if children is None:
            children = list()
            for i in range(self.node_num):
                if self.adjacent_matrix[node - 1][i] != 0:
                    children.append(i + 1)
            self.children_buffer.update({node: children})
            return children
        else:
            return children

    def get_parents(self, node):
        parents = self.parents_buffer.get(node)
        if parents is None:
            parents = list()
            for i in range(self.node_num):
                if self.adjacent_matrix[i][node - 1] != 0:
                    parents.append(i + 1)
            self.parents_buffer.update({node: parents})
            return parents
        else:
            return parents

    def get_weight(self, src, dest):
        return self.adjacent_matrix[src - 1][dest - 1]

    # return true if node1 is parent of node 2 , else return false
    def is_parent_of(self, node1, node2):
        if self.adjacent_matrix[node1 - 1][node2 - 1] != 0:
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


graph = None


# Heuristics algorithm for IC model
def degree_discount_ic(k):
    seeds = set()
    ddv = np.zeros(graph.node_num)
    tv = np.zeros(graph.node_num)
    for i in range(graph.node_num):
        ddv[i] = graph.get_out_degree(i + 1)
    for i in range(k):
        u = ddv.argmax() + 1
        ddv[u - 1] = -1  # never used
        seeds.add(u)
        children = graph.get_children(u)
        for child in children:
            if child not in seeds:
                tv[child - 1] = tv[child - 1] + 1
                ddv[child - 1] = ddv[child - 1] - 2 * tv[child - 1] - (graph.get_out_degree(child) - tv[child - 1]) * \
                                                                      tv[child - 1] * graph.get_weight(u, child)
    return list(seeds)


def to_sub_node_set(nodes):
    sub_node_set = np.zeros(graph.node_num)
    for node in nodes:
        sub_node_set[node - 1] = 1
    return sub_node_set


# sub_node_set: np.array
def is_in_sub_node_set(node, sub_node_set):
    if sub_node_set[node - 1] == 1:
        return True
    else:
        return False


def init_D():
    D = list()
    for i in range(graph.node_num + 1):
        D.append([])
    return D


# input: Q,D,spd,pp,r,W,U
# Q:list
# D:list(list) D[x]: explored neighbor of x
# spd ,pp,r: float
# W,U node_set np.array
# spdW_
def forward(Q, D, spd, pp, r, W, U=None, spdW_=None):
    x = Q[-1]
    if U is None:
        U = []
    children = graph.get_children(x)
    q = to_sub_node_set(Q)
    count = 0
    while True:
        # any suitable chid is ok

        for child in range(count, len(children)):
            if is_in_sub_node_set(children[child], W) and (not is_in_sub_node_set(children[child], q)) and (
                        children[child] not in D[x]):
                y = children[child]
                break
            count = count + 1

        # no such child:
        if count == len(children):
            return Q, D, spd, pp

        if pp * graph.get_weight(x, y) < r:
            D[x].append(y)
        else:
            Q.append(y)
            pp = pp * graph.get_weight(x, y)
            spd = spd + pp
            D[x].append(y)
            x = Q[-1]
            for v in U:
                if is_in_sub_node_set(v, q) == False:
                    spdW_[v] = spdW_[v] + pp
            children = graph.get_children(x)
            q = to_sub_node_set(Q)
            count = 0


def backtrack(u, r, W, U=None, spdW_=None):
    Q = [u]
    spd = 1
    pp = 1
    D = init_D()
    while len(Q) != 0:
        Q, D, spd, pp = forward(Q, D, spd, pp, r, W, U, spdW_)
        u = Q.pop()
        D[u] = []
        if len(Q) != 0:
            v = Q[-1]
            pp = pp / graph.get_weight(v, u)
    return spd


def simpath_spread(S, r, U=None, spdW_=None):
    spread = 0
    # W: V-S
    W = np.ones(graph.node_num)
    for s in S:
        W[s - 1] = 0
    for u in S:
        # W = W-u
        W[u - 1] = 1
        spread = spread + backtrack(u, r, W, U, spdW_)
        # W = W+u
        W[u - 1] = 0
    return spread


# simpath algorithm with plan greedy algorithm
def simpath_greedy(k):
    S = []
    spd = 0
    marginal_gain = np.zeros(graph.node_num + 1)
    for i in range(k):
        for node in range(1, graph.node_num + 1):
            marginal_gain[node] = simpath_spread(S + [node], 0.001) - spd
        for t in S:
            marginal_gain[t] = -1
        u = marginal_gain.argmax()
        S.append(u)
        # print('add node %d' % u)
        spd = simpath_spread(S, 0.001)
    return S


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='CARP instance file', dest='graph_path')
    parser.add_argument('-k', type=int, help='predefined size of the seed set', dest='seed_size')
    parser.add_argument('-m', help='diffusion model', dest='model')
    parser.add_argument('-b',
                        help='specifies the termination manner and the value can only be 0 or 1. If it is set to 0, '
                             'the termination condition is as the same defined in your algorithm. Otherwise, '
                             'the maximal time budget specifies the termination condition of your algorithm.',
                        dest='type')
    parser.add_argument('-t', type=int, help='time budget', dest='timeout')
    parser.add_argument('-r', type=int, help='random seed', dest='random_seed')

    args = parser.parse_args()
    graph_path = args.graph_path
    seed_size = args.seed_size
    model = args.model
    termination_type = args.type
    timeout = args.timeout
    random_seed = args.random_seed

    np.random.seed(random_seed)

    graph = Graph(read_graph_info(graph_path)[0])

    if model == 'IC':
        seeds = degree_discount_ic(k=seed_size)
        print_seeds(seeds)
    elif model == 'LT':
        seeds = simpath_greedy(k=seed_size)
        print_seeds(seeds)
    else:
        print('Type err')
