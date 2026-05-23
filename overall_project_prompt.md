# overall project prompt

This project is for large-scale network analysis and a custom technique that I will name Guided Label Propagation, which is a spin-off of other types of label propagation techniques for community detection. All code should be written in Python. At the current point the project needs to contain three main features that should be possible to use independently of the other:

1. Constructing and doing basic analysis of potentially very large network graphs based on data on social media behavior such as sharing similar content, following each other and mentioning users or forwarding each others content. Network functions should as much as possible draw on the Python package Networkit which is developed specifically for large-scale networks. In general everything should be optimized for computation speed the scales very well. Memory efficiency should also be considered, though the option to sacrifice memory efficiency for speed should be widely available. The network construction and analysis needs to support weigthed edges and directed and undirected graphs, but NOT multi-edge graphs or multi-layer graphs. In constructing graphs information about the original IDs of all nodes need to be retained. The construction of a graph should be based on a simple list of sources and targets (a CSV document or dataframe). It should be possible to add a third column with custom weights, though if no weight column is present a function should be apple to calculate weights based on duplicate source-target pairs. 

2. Applying guided lable propagation to detect communities and calculate community belonging probabilities based on a set of input nodes. It can be considered a method for semi-supervised community detection in large-scale networks. For complex networks containing tens of thousands of unknown entities, it can become difficult to ascertain what role they play in the network. Guided label propagation is a method to search for patterns pertaining specifically to categories of interest for the given study. Rather than labelling nodes based on their belonging to clusters of a network partition, where the separation between clusters is determined by network structure (e.g. Blondel et al., 2008), guided label propagation instead relies on a small sample of input nodes with known affinities such as accounts belonging to political right- and left-wing actors. The labels of these known accounts (left and right wing) are then propagated throughout the network in order to uncover how strongly the rest of the accounts are affiliated with the position represented by those known accounts. The advantage of the approach lies in its ability to help researchers to identify the role of particular clusters in social media based information sharing networks by focusing on accounts’ affinity towards known categories of interest rather than clusters that are initially arbitrary. 

3. Time series analysis of networks where constructed graphs are created using the same basic techniques as in the first feature described above, but where each source-target pair contains a datetime value in another column.

The concrete functions that need to be availabe for basic construction and analysis (though it should be possible to add many more), are the following:

- Create a graph based on an edge list of source and target pairs, with the option of a weight column. It should be possible to create both a uni partite and bi-partite graph. It should also be possible to first create a bi-partite graph and then collapse it to a unipartite graph.

- It should be possible to apply various backboning techniques on the graph ideally setting levels based on the amount of nodes or edges the user wishes to retain

- It should be possible to extract a range of typical centrality metrics from the graph

- It should be possible to perform community detection efficiently on very large graphs, to start with a louvain based community detection might be enough as Networkit has an efficient implementation of this. The community detection procedure should have the option to run multiple iterations of community detection with various settings and export the results based on the similarity between partitions in each iteration.

- It should be possible to apply custom filters based on simple metrics such as degrees, edge weights, giant component, specific nodes and edges as well as any other centrality metric that has already been extracted from the Graph.

- It should be possible to export the graph in various formats such as .gexf and combine with metadata about the nodes which should be possible to import into the graph based on the original IDs in the edge list of source-target pairs.



For Guided Label Propagation the following functions should be available:

- Based on a constructed graph, either bi-partite or unipartite in combination with either an undirected or directed graph, a set of labels and probabilities that a node should have a certain label should be outputted (see the function stlp in the file net_utils.py for inspiration). The probability for a node to belong to a specific label should be based on the strength of connection to a, potentially very small, set of input nodes corresponding to the original IDs of the original edge list of source-target pairs. If the graph is directed the results of the label propagation should pertain to versions based on both the out-degree connections of the nodes and the in-degree connections. Ideally calculating the probalities and propagating them throughout the network should be done with matrix calculation for maximal efficiency, however is there is an even more efficient way to do it, that would be fine as well.

- Based on the Guided Label Propagation it should be possible to evaluate the results against both a test set comprised of a split between the originally inputted nodes as well as a fully external validation set.


For the time-series network analysis, the following functions should be available:

- Construct a series of network-slices based on either daily, weekly, monthly or yearly intervals. It should also be possible to specify rolling average and how many intervals to consider for the rolling average. Additionally it should be possible to construct both a cummulative and non-cummulative graph. The time-series analysis should make it possible to export metrics pertaining to all nodes for each of the slices. For each time slice, it should also be possible to export data about connections between various types of nodes based on categories that they belong to based on metadata tied to the original IDs in the edgelist of source-target paris.


## Technical details

1. The code should rely as much as possible on the following python frameworks for computational efficiency:

- Numpy
- Networkit
- Polars

2. All functions should be multi-process when it is the most efficient option for computational scalability

3. The codebased can have a object based or functional or hybrid approach depending on what is most efficient and logical from an architecture standpoint.

## Progress details

For now the codebase should only be available as a backend API, though it should be developed with the possibility of expanding a frontend as well. For now we don't need a CLI for querying the API using the terminal, instead this should be done via a notebook or test.py script, for now.








