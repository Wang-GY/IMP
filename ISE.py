import numpy as np
import os
import argparse
import threading
import multiprocessing
import Queue

spd = None


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
# graph : Graph
# seed : list
# sample_num : int
# use multiple thread
def influence_spread_computation_IC(graph, seeds, sample_num=10000):
    influence = 0
    for i in range(sample_num):
        node_list = list()
        node_list.extend(seeds)
        checked = np.zeros(graph.node_num)
        for node in node_list:
            checked[node - 1] = 1
        while len(node_list) != 0:
            current_node = node_list.pop(0)
            influence = influence + 1
            children = graph.get_children(current_node)
            for child in children:
                if checked[child - 1] == 0:
                    if happen_with_prop(graph.get_weight(current_node, child)):
                        checked[child - 1] = 1
                        node_list.append(child)
    return influence


def influence_spread_computation_IC_Mu(graph, seeds, n=multiprocessing.cpu_count()):
    pool = multiprocessing.Pool()
    results = []
    sub = int(10000 / n)
    for i in range(n):
        result = pool.apply_async(influence_spread_computation_IC, args=(graph, seeds, sub))
        results.append(result)
    pool.close()
    pool.join()
    influence = 0
    for result in results:
        influence = influence + result.get()
    return influence / 10000


# input: Q,D,spd,pp,r,W,U
# Q:list
# D:list(list) D[x]: explored neighbor of x
# spd ,pp,r: float
# W node_set np.array
# U: list
# spdW_
def forward(Q, D, spd, pp, r, W, U, spdW_u, graph):
    x = Q[-1]
    if U is None:
        U = []
    children = graph.get_children(x)
    count = 0
    while True:
        # any suitable chid is ok

        for child in range(count, len(children)):
            if (children[child] in W) and (children[child] not in Q) and (children[child] not in D[x]):
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
                if v not in Q:
                    spdW_u[v] = spdW_u[v] + pp
            children = graph.get_children(x)
            count = 0


def backtrack(u, r, W, U, spdW_, graph):
    Q = [u]
    spd = 1
    pp = 1
    D = init_D(graph)

    while len(Q) != 0:
        Q, D, spd, pp = forward(Q, D, spd, pp, r, W, U, spdW_, graph)
        u = Q.pop()
        D[u] = []
        if len(Q) != 0:
            v = Q[-1]
            pp = pp / graph.get_weight(v, u)
    return spd


def simpath_spread(S, r, U, graph, spdW_=None):
    spread = 0
    # W: V-S
    W = set(graph.nodes).difference(S)
    if U is None or spdW_ is None:
        spdW_ = np.zeros(graph.node_num + 1)
        # print 'U None'
    for u in S:
        W.add(u)
        # print spdW_[u]
        spread = spread + backtrack(u, r, W, U, spdW_[u], graph)
        # print spdW_[u]
        W.remove(u)
    return spread


def influence_spread_computation_LT(graph, seeds, r=0.01):
    return simpath_spread(seeds, r, None, graph)


def init_D(graph):
    D = list()
    for i in range(graph.node_num + 1):
        D.append([])
    return D


def main(args, q):
    graph_path = args.graph_path
    seed_path = args.seed_path
    model = args.model
    random_seed = args.random_seed
    np.random.seed(random_seed)

    graph = Graph(read_graph_info(graph_path))
    seeds = read_seed_info(seed_path)

    if model == 'IC':
        spd = influence_spread_computation_IC_Mu(seeds=seeds, graph=graph)
        q.put(spd)
    elif model == 'LT':
        spd = influence_spread_computation_LT(graph=graph, seeds=seeds, r=0.001)
        q.put(spd)
    else:
        raise ValueError("Mole type should be IC or LT")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='CARP instance file', dest='graph_path')
    parser.add_argument('-s', help='seed set', dest='seed_path')
    parser.add_argument('-m', help='diffusion model', dest='model')
    parser.add_argument('-b', type=int,
                        help='specifies the termination manner and the value can only be 0 or 1. If it is set to 0, '
                             'the termination condition is as the same defined in your algorithm. Otherwise, '
                             'the maximal time budget specifies the termination condition of your algorithm.',
                        dest='type')
    parser.add_argument('-t', type=float, help='termination', dest='timeout')
    parser.add_argument('-r', type=int, help='random seed', dest='random_seed')

    args = parser.parse_args()
    graph_path = args.graph_path
    seed_path = args.seed_path
    model = args.model
    type = args.type
    timeout = args.timeout
    random_seed = args.random_seed

    np.random.seed(random_seed)

    # finish by algorithm
    if type == 0:
        graph = Graph(read_graph_info(graph_path))
        seeds = read_seed_info(seed_path)
        if model == 'IC':
            print influence_spread_computation_IC_Mu(seeds=seeds, graph=graph)
        elif model == 'LT':
            print influence_spread_computation_LT(graph=graph, seeds=seeds, r=0.001)
        else:
            print('Type err')

    # finish by interrupt
    elif type == 1:
        if timeout < 60:
            print 'Given time should not less than 60s!'
            exit(1)
        q = Queue.Queue()
        t = threading.Thread(target=main, args=(args, q))
        t.setDaemon(True)
        t.start()
        t.join(timeout - 1)
        result = list()
        while not q.empty():
            result.append(q.get())
        if len(result) != 0:
            spd = result[-1]
            print(spd)
        else:
            print("Given time was too short")
