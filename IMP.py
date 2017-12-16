import numpy as np
import os
import heapq
import time
import argparse
import threading
import multiprocessing

graph = None
seeds = None


# timer real time
def settimeout(terment_time):
    time.sleep(terment_time)
    print_seeds()
    exit(1)


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
            print 'IOError'
    else:
        print 'file can not found'


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


def print_seeds():
    for seed in seeds:
        print(seed)


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


# base on heap queue
# The CELF queue is maintained in decreasing order of the marginal
# gains and thus, no other node can have a larger marginal gain.
class CELFQueue:
    # create if not exist
    nodes = None
    q = None
    nodes_gain = None

    def __init__(self):
        self.q = []
        self.nodes_gain = {}

    def put(self, node, marginalgain):
        self.nodes_gain[node] = marginalgain
        heapq.heappush(self.q, (-marginalgain, node))

    def update(self, node, marginalgain):
        self.remove(node)
        self.put(node, marginalgain)

    def remove(self, node):
        self.q.remove((-self.nodes_gain[node], node))
        self.nodes_gain[node] = None
        heapq.heapify(self.q)

    def topn(self, n):
        top = heapq.nsmallest(n, self.q)
        top_ = list()
        for t in top:
            top_.append(t[1])
        return top_

    def get_gain(self, node):
        return self.nodes_gain[node]


def get_sample_graph(graph):
    nodes = graph.nodes
    edges = {}
    children = {}
    parents = {}
    node_num = graph.node_num
    edge_num = None
    for edge in graph.edges:
        if happen_with_prop(graph.edges[edge]):
            edges[edge] = graph.edges[edge]
            src = edge[0]
            des = edge[1]

            if children.get(src) is None:
                children[src] = []
            if parents.get(des) is None:
                parents[des] = []

            children[src].append(des)
            parents[des].append(src)
    return Graph((nodes, edges, children, parents, node_num, len(edges)))


def BFS(graph, nodes, get_checked_array=False):
    node_list = list()
    node_list.extend(nodes)
    result_list = list()
    checked = np.zeros(graph.node_num)
    for node in node_list:
        checked[node - 1] = 1
    while len(node_list) != 0:
        current_node = node_list.pop(0)
        result_list.append(current_node)
        children = graph.get_children(current_node)
        for child in children:
            if checked[child - 1] == 0:
                checked[child - 1] = 1
                node_list.append(child)
    if get_checked_array:
        return result_list, checked
    return result_list


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


# k: seed size
def new_greedyIC(graph, k, R=20000):
    seeds = set()
    for i in range(k):
        sv = np.zeros(graph.node_num)
        afficted_nodes, is_afficted = BFS(graph, list(seeds), True)
        # for seed in seeds:
        for i in range(R):
            sample_graph = get_sample_graph(graph)
            for v in range(graph.node_num):
                if is_afficted[v] == 0:
                    sv[v] = sv[v] + len(BFS(sample_graph, [v + 1]))
        for i in range(graph.node_num):
            sv[i] = sv[i] / R
        seeds.add(sv.argmax() + 1)
    return list(seeds)


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
                ddv[child - 1] = ddv[child - 1] - 2 * tv[child - 1] - (graph.get_out_degree(child) - tv[
                    child - 1]) * \
                                                                         tv[child - 1] * graph.get_weight(u, child)
    return list(seeds)


def init_D():
    D = list()
    for i in range(graph.node_num + 1):
        D.append([])
    return D


def get_vertex_cover():
    # dv[i] out degree of node i+1
    dv = np.zeros(graph.node_num)
    # e[i,j] = 0: edge (i+1,j+1),(j+1,i+1) checked
    check_array = np.zeros((graph.node_num, graph.node_num))
    checked = 0
    edges = set()
    for i in range(graph.node_num):
        # for a edge (i,j) and (j,i) may be count twice but the algorithm is to find a vertex cover. it doesn't mater
        dv[i] = graph.get_out_degree(i + 1) + graph.get_in_degree(i + 1)
    # V: Vertex cover
    V = set()
    while checked < graph.edge_num:
        s = dv.argmax() + 1
        V.add(s)
        # make sure that never to select this node again
        children = graph.get_children(s)
        parents = graph.get_parents(s)
        for child in children:
            if check_array[s - 1][child - 1] == 0:
                check_array[s - 1][child - 1] = 1
                checked = checked + 1
        for parent in parents:
            if check_array[parent - 1][s - 1] == 0:
                check_array[parent - 1][s - 1] = 1
                checked = checked + 1
        dv[s - 1] = -1
    return list(V)


# input: Q,D,spd,pp,r,W,U
# Q:list
# D:list(list) D[x]: explored neighbor of x
# spd ,pp,r: float
# W node_set np.array
# U: list
# spdW_
def forward(Q, D, spd, pp, r, W, U, spdW_u):
    x = Q[-1]
    if U is None:
        U = []
    children = graph.get_children(x)
    count = 0
    while True:
        # any suitable chid is ok

        for child in range(count, len(children)):
            # if is_in_sub_node_set(children[child],W) and (not is_in_sub_node_set(children[child],q)) and (children[child] not in D[x]):
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


def backtrack(u, r, W, U, spdW_):
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


def simpath_spread(S, r, U, spdW_=None):
    spread = 0
    # W: V-S
    W = set(graph.nodes).difference(S)
    if U is None or spdW_ is None:
        spdW_ = np.zeros(graph.node_num + 1)
        # print 'U None'
    for u in S:
        W.add(u)
        # print spdW_[u]
        spread = spread + backtrack(u, r, W, U, spdW_[u])
        # print spdW_[u]
        W.remove(u)
    return spread


def simpath(k, r, l):
    C = set(get_vertex_cover())
    V = set(graph.nodes)

    V_C = V.difference(C)
    # spread[x] is spd of S + x
    spread = np.zeros(graph.node_num + 1)
    spdV_ = np.ones((graph.node_num + 1, graph.node_num + 1))
    for u in C:
        U = V_C.intersection(set(graph.get_parents(u)))
        spread[u] = simpath_spread(set([u]), r, U, spdV_)
    for v in V_C:
        v_children = graph.get_children(v)
        for child in v_children:
            spread[v] = spread[v] + spdV_[child][v] * graph.get_weight(v, child)
        spread[v] = spread[v] + 1
    celf = CELFQueue()
    # put all nodes into celf queqe
    # spread[v] is the marginal gain at this time
    for node in range(1, graph.node_num + 1):
        celf.put(node, spread[node])
    S = set()
    W = V
    spd = 0
    # mark the node that checked before during the same Si
    checked = np.zeros(graph.node_num + 1)

    while len(S) < k:
        U = celf.topn(l)
        spdW_ = np.ones((graph.node_num + 1, graph.node_num + 1))
        spdV_x = np.zeros(graph.node_num + 1)
        simpath_spread(S, r, U, spdW_=spdW_)
        for x in U:
            for s in S:
                spdV_x[x] = spdV_x[x] + spdW_[s][x]
        for x in U:
            if checked[x] != 0:
                S.add(x)
                W = W.difference(set([x]))
                spd = spread[x]
                # print spread[x],simpath_spread(S,r,None,None)
                checked = np.zeros(graph.node_num + 1)
                celf.remove(x)
                break
            else:
                spread[x] = backtrack(x, r, W, None, None) + spdV_x[x]
                checked[x] = 1
                celf.update(x, spread[x] - spd)
    return S


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='CARP instance file', dest='graph_path')
    parser.add_argument('-k', type=int, help='predefined size of the seed set', dest='seed_size')
    parser.add_argument('-m', help='diffusion model', dest='model')
    parser.add_argument('-b', type=int,
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
    graph = Graph(read_graph_info(graph_path))
    if termination_type == 0:
        if model == 'IC':
            start = time.time()
            if graph.node_num <= 200:
                seeds = new_greedyIC(graph, k=seed_size, R=10000)
            else:
                seeds = degree_discount_ic(k=seed_size)
            # seeds = new_greedyIC(graph, k=seed_size, R=10000)
            print_seeds()
            run_time = (time.time() - start)
        elif model == 'LT':
            start = time.time()
            seeds = simpath(seed_size, 0.001, 7)
            run_time = (time.time() - start)
            print_seeds()
            print run_time
        else:
            print('Model type err')
    elif termination_type == 1:
        if timeout < 0:
            print 'Given time should not less than 60s!'
            exit(1)
        timer = threading.Thread(target=settimeout, args=(timeout - 1,))
        timer.start()

        if model == 'IC':
            if graph.node_num <= 200:
                seeds = new_greedyIC(graph, k=seed_size, R=10000)
            else:
                seeds = degree_discount_ic(k=seed_size)
                seeds = new_greedyIC(graph, k=seed_size, R=10000)
        elif model == 'LT':
            seeds = simpath(seed_size, 0.001, 7)
        else:
            print('Model type err')
