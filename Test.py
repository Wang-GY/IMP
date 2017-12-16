import multiprocessing
import ISE
import time
import numpy as np

graph_info = ISE.read_graph_info('NetPHY.txt')
graph = ISE.Graph(graph_info)


# graph = None

def influence_spread_computation_IC_Mu(seeds, n=multiprocessing.cpu_count()):
    pool = multiprocessing.Pool()
    # cpus = multiprocessing.cpu_count()
    results = []
    sub = int(10000 / n)
    for i in range(n):
        result = pool.apply_async(influence_spread_computation_IC, args=(seeds, sub))
        results.append(result)
    pool.close()
    pool.join()
    influence = 0
    for result in results:
        influence = influence + result.get()
    return influence / 10000


# graph : Graph
# seed : list
# sample_num : int
# graph : Graph
# seed : list
# sample_num : int
# use multiple thread
def influence_spread_computation_IC(seeds, sample_num=10000):
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
                    if ISE.happen_with_prop(graph.get_weight(current_node, child)):
                        checked[child - 1] = 1
                        node_list.append(child)
    return influence


if __name__ == '__main__':
    seeds = [13541,
             15303,
             5192,
             22764,
             22762,
             25580,
             5551,
             17527,
             25578,
             13439]
    start = time.time()
    print influence_spread_computation_IC_Mu(seeds)
    run_time = (time.time() - start)
    print run_time

    start = time.time()
    print influence_spread_computation_IC(seeds, 10000) / 10000
    run_time = (time.time() - start)
    print run_time
