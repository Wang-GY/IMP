# <center>Influence maximize problem</center>
 <center> 王戈扬 </center >  
<center> 11510050</center >
## Describe
Kempe et al formulated influence maximization as a discrete optimization problem: Given a directed social graph with users as nodes, edge weights reflecting influence between users, and a number k (called budget), find k users, called the seed set, such that by activating them, the expected spread of the influence (just spread for brevity) according to a propagation model is maximized. The influence diffusion process unfolds in discrete time steps, as captured by the propagation model.  
Two classical propagation models  are the Linear Threshold (LT) Model and the Independent Cascade (IC) Model, both taken from mathematical sociology. In both models, at any time step, a user is either active (an adopter of the product) or inactive. In the IC model, when an inactiveuser becomes active at a time step t, it gets exactly one chance to independently activate its currently inactive neighbors at time t +1. In the LT model, the sum of incoming edge weights on any node is assumed to be at most 1 and every user chooses an activation threshold uniformly at random from [0, 1]. At any time step, if the sum of incoming influence (edge weights) from the active neighbors of an inactive user exceeds its threshold, it becomes active. In both models, influence propagates until no more users can become active.  
## 1.Preliminaries
*Software:* Pycharm  
*Language:*  python 2.7.14  
*Algorithm Used:*  
1.SIMPATH (LT)  
2.NewGreedyIC (IC)  
3.DegreeDiscountIC (IC)  
4.SingleDiscount (LT)
## 2.	Methodology
### 2.1 SIMPATH  
SIMPATH is an algorithm for the LT model. It uses two novel optimizations to reduce the number of spread estimation calls. The first one is called *Vertex Cover Optimization*,it divided the graph into two parts, a vertex cover and the rest part.we need to compute the spread of every node in the vertex cover and simultaneously compute the spread of the rest part of nodes. Another Optimization is called *Look Ahead Optimization* which addresses this issue and keeps the running time of subsequent iterations small. Specifically, using a parameter l, it picks the top-l most promising seed candidates in the start of an iteration and shares the marginal gain computation of those candidates.
### 2.2 NewGreedyIC  
NewGreedyIC is an optimization of the General greedy algorithm. The core idea is to generate a subgraph with a subset of edges. Each edge will be keeped with the probability equals the weight of the edge. A single subgraph can be used to compute marginal gain of every node. Instead of resampling to calculate each marginal gain of a node.
### 2.3 DegreeDiscount & DegreeDiscountIC
DegreeDiscountIC is a heurisitics algorithm for the IC model. DegreeDiscount is a (bad) gheurisitics algorithm for every model. we may think that nodes with more outdegree are more like to become the seed. These algorithm are essencially doing the same thing in a more accurate way. The core idea is that when a neighbor of a node u becomes a seed. The gian of choosing the ndoe u to will decress. In the DegreeDiscount algorithm it will decress by 1. and in the DegreeDiscountIC algorithm it will be calculate by a estimation.
## 3.Empirical Verification  
I used the given data to estimate the performance of my algorithm. THe SIMPATH algorithm is very fast and accurate. but the NewGreedyIC will be slow. In order to find 4 seeds in network, The SIMPATH,DegreeDiscountIC algorithm will get the result within 0.1 sec. but the NewGreedyIC will cost about 11 sec. I don't have time to do other test.
## 4. References  
[1] W. Chen, Y. Wang, and S. Yang, “Efficient influence maximization in social networks,” in KDD 2009.  
[2] Amit Goyal, Wei Lu, Laks V. S. Lakshmanan,SIMPATH: An Efficient Algorithm for Influence Maximization under the Linear Threshold Model in IEEE 2011.
