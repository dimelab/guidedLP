
import networkit as nxk
import pandas as pd
import re
import networkx as nx
import time
import numpy as np
from multiprocessing import Pool
import multiprocessing
from itertools import chain
import polars as pl
import os
import math
from copy import copy, deepcopy
import datetime
import random
import sys
import psutil
import time
import warnings
from collections import defaultdict
from scipy.stats import norm
warnings.filterwarnings("ignore", category=DeprecationWarning)

def timer(func):
	def wrapper(*args, **kwargs):
			start_time = time.time()
			result = func(*args, **kwargs)
			end_time = time.time()
			execution_time = end_time - start_time
			print(f"Execution time: {execution_time} seconds")
			return result
	return wrapper

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def _stlp_get_n_idx_ws_sorted(g,ns,out_weights=True):

	def _transform_w(w):
		return tanh((w+1)/100.0)

	sorted_n_idx = defaultdict(list)
	sorted_ws = defaultdict(list)
	sorted_ns = defaultdict(list)
	if out_weights:
		wneighs = [g.iterNeighborsWeights(n) for n in ns]
	else:
		wneighs = [g.iterInNeighborsWeights(n) for n in ns]
	for wneigh,nn in zip(wneighs,ns):
		n_idx = []
		ws = []
		for n,w in wneigh:
			w = _transform_w(w)
			n_idx.append(n)
			ws.append(w)
		nlen = len(n_idx)
		sorted_n_idx[nlen].append(n_idx)
		sorted_ws[nlen].append(ws)
		sorted_ns[nlen].append(nn)
	sorted_ws = {nlen:np.array(v) for nlen,v in sorted_ws.items()}
	sorted_n_idx = {nlen:np.array(v,dtype='int32') for nlen,v in sorted_n_idx.items()}
	return sorted_n_idx,sorted_ws,sorted_ns,g

def _stlp_get_n_idx_ws_chunk(n_idxs, ws, aff_map, dist_map):
	ii, jj = np.meshgrid(np.arange(n_idxs.shape[0]), np.arange(n_idxs.shape[1]), indexing='ij')
	affs = aff_map[n_idxs[ii, jj], :]
	dists = dist_map[n_idxs[ii, jj], :]
	wss = ws

	return affs, dists, wss

def _compute_affinity(affs, dists, wss):
	modified_affs = affs * (dists ** 1)
	new_aff = np.zeros((modified_affs.shape[0], modified_affs.shape[2]))
	for i in range(modified_affs.shape[2]):
		new_aff[:, i] = np.sum(wss * modified_affs[:, :, i], axis=1)
	naff_sum = new_aff.sum(axis=1)
	new_aff = np.nan_to_num(new_aff / naff_sum[:, np.newaxis], nan=0.0)
	return new_aff

def _update_affs(nn_idx,nn_ws,nn_ns,aff_map,dist_map):

	new_aff_stack = []
	ncount = 0
	save_all_counts = {}
	nlens = list(nn_ns.keys())
	random.shuffle(nlens)
	for nlen in nlens:
		affs,dists,wss = _stlp_get_n_idx_ws_chunk(nn_idx[nlen],nn_ws[nlen],aff_map,dist_map)
		for i,n in enumerate(nn_ns[nlen]):
			save_all_counts[n]=ncount
			ncount+=1
		new_aff = _compute_affinity(affs,dists,wss)
		new_aff_stack.append(new_aff)

	new_aff_stack = np.concatenate(new_aff_stack,axis=0)
	new_aff_stack = new_aff_stack[[v for k,v in sorted(save_all_counts.items())]]

	return new_aff_stack

def get_djik_distances(g,org_nodes,labels):

	all_dists = {v:[] for v in labels.values()}
	for nl,lab in labels.items():
		#djik = nxk.distance.MultiTargetDijkstra(nxk.graphtools.toUndirected(g), nl, org_nodes)
		djik = nxk.distance.Dijkstra(g, nl)
		djik.run()
		dists = djik.getDistances()
		all_dists[lab].append(dists)
	return all_dists

def get_rev_direction_affinities(g,labels,org_nodes,dist_map,aff_map,out_affinities,title,net_idx,aff_cats):

	rev_net_ids = create_rev_net_idx(g,net_idx)
	dist_map = np.ones([dist_map.shape[0],dist_map.shape[1]])
	aff_map = np.zeros([aff_map.shape[0],aff_map.shape[1]])
	nn_idx, nn_ws, nn_ns, g = _stlp_get_n_idx_ws_sorted(g,org_nodes,out_weights=False)
	for label,ns in out_affinities.items():
		l = label.split("_")[-1]
		for n,v in ns.items():
			aff_map[net_idx[n]][aff_cats[l]]=v
	aff_map = _update_affs(nn_idx,nn_ws,nn_ns,aff_map,dist_map)
	out_affinities = {title+"_"+lab+"_out_degree":{k:0.0 for k in net_idx.keys()} for lab in set(labels.values())}
	for lab in set(labels.values()):
		for i,n in enumerate(aff_map):
			out_affinities[title+"_"+lab+"_out_degree"][rev_net_ids[i]]=n[aff_cats[lab]]
	return out_affinities


def set_noise_labels(g,net_idx,org_labels,sample_def=None,sample_coeff=1.25):

	if sample_def is not None:
		noise_labels = {n:"noise" for n in random.sample([n for n in net_idx.values() if n not in org_labels and g.hasNode(n) and n in sample_def],int(sample_coeff*len(org_labels)))}
	else:
		noise_labels = {n:"noise" for n in random.sample([n for n in net_idx.values() if n not in org_labels and g.hasNode(n)],int(sample_coeff*len(org_labels)))}
	return noise_labels

@timer
def stlp(g,labels,net_idx,title="test",num_cores=1,its=5,epochs=3,stochasticity=1,verbose=False,sample_def=None):

	if net_idx is not None: rev_net_ids = create_rev_net_idx(g,net_idx)
	#max_deg = nxk.graphtools.maxDegree(g)
	org_nodes = list([n for n in g.iterNodes()])
	affinities = []
	distances = []
	all_dists = {}
	org_labels = {net_idx[n]:l for n,l in labels.items() if n in net_idx}
	labels = {}
	if len(org_labels) < 1:
		return None, None
	if its < 0:
		diam = nxk.distance.EffectiveDiameterApproximation(nxk.graphtools.toUnweighted(nxk.graphtools.toUndirected(g)))
		diam.run()
		its = int(diam.getEffectiveDiameter())+1
		print (f"Diameter is: {its}")

	print ("generating neighbour idxs")
	nn_idx, nn_ws, nn_ns, g = _stlp_get_n_idx_ws_sorted(g,org_nodes)
	for epoch in range(epochs):
		print ("getting distances")
		if len(all_dists) > 0:
			for n,_ in noise_labels.items():
				del labels[n]
			noise_labels = set_noise_labels(g,net_idx,org_labels,sample_def=sample_def)
			labels.update(noise_labels)
			all_dists["noise"]=get_djik_distances(nxk.graphtools.toUnweighted(nxk.graphtools.toUndirected(g)),org_nodes,noise_labels)["noise"]
		else:
			noise_labels = set_noise_labels(g,net_idx,org_labels,sample_def=sample_def)
			labels.update(org_labels)
			labels.update(noise_labels)
			all_dists = get_djik_distances(nxk.graphtools.toUnweighted(nxk.graphtools.toUndirected(g)),org_nodes,labels)
		print ("distances set")

		aff_cats = {c:j for j,c in enumerate(sorted(list(set(labels.values()))))}
		dist_map = np.array([np.mean(((np.array(v)+1)**-1), axis=0) for k,v in sorted(all_dists.items())]).T
		#dist_map = np.array([np.mean(((np.array(v)+1)**-1)**0.001, axis=0) for k,v in sorted(all_dists.items())]).T
		#dist_map = np.array([np.mean(((np.array(v)+1)**-1)**((np.array(v)+1)**-1)**0.55, axis=0) for k,v in sorted(all_dists.items())]).T
		aff_map = np.zeros((len(net_idx),len(all_dists)))
		for n,l in labels.items():
			aff_map[n][aff_cats[l]]=1.0
		n_labels = len(aff_cats)

		start_time = time.time()
		for it in range(its):
			new_aff_stack = _update_affs(nn_idx,nn_ws,nn_ns,aff_map,dist_map)
			#new_aff_stack = np.nan_to_num(new_aff_stack / new_aff_stack.sum(axis=1)[:, np.newaxis], nan=0.0, posinf=0.0, neginf=0.0)
			for n,l in labels.items():
				new_aff_stack[n]=np.zeros(n_labels)
				new_aff_stack[n][aff_cats[l]]=1.0
			if it % 10 == 0:
				print (np.sum(abs(aff_map-new_aff_stack)))
			aff_map = new_aff_stack

		affinities.append(aff_map)
		distances.append(dist_map)

	print ("Tallying means.")
	for i in range(len(affinities)):
		if i > 1:
			print (np.sum(abs(np.mean(np.array(affinities[:i-1]),axis=0)-np.mean(np.array(affinities[:i]),axis=0))))
	affinities = np.mean(np.array(affinities),axis=0)
	distances = np.mean(np.array(distances),axis=0)
	out_affinities = {title+"_"+lab:{k:0.0 for k in net_idx.keys()} for lab in set(labels.values())}
	out_distances = {k:[] for k in net_idx.keys()}
	for lab in set(labels.values()):
		for i,n in enumerate(affinities):
			out_affinities[title+"_"+lab][rev_net_ids[i]]=n[aff_cats[lab]]
	out_rev_affinities = get_rev_direction_affinities(g,labels,org_nodes,dist_map,aff_map,out_affinities,title,net_idx,aff_cats)
	for lab in set(labels.values()):
		if "noise" not in lab:
			for i,d in enumerate(distances):
				out_distances[rev_net_ids[i]].append(np.log(d+2))
	return {title:out_affinities},{title+"_"+"dist":{k:np.mean(np.array(v)) for k,v in out_distances.items()}},out_rev_affinities


def set_affinities(g,affinities):

	for n,affs in affinities.items():
		for aff,val in affs.items():
			if n in g:
				nx.set_node_attributes(g,{n:val},aff)
	return g

def set_labels(g,labels,title="test"):

	nx.set_node_attributes(g,labels,title)
	return g

def remap_nodes_based_on_category(df,node_mapping,keep_self_loops=False):

	df = df.with_columns(pl.col("o").map_dict(node_mapping).alias("o"))
	df = df.with_columns(pl.col("e").map_dict(node_mapping).alias("e"))
	if not keep_self_loops:
		df = df.filter(pl.col("e")!=pl.col("o"))
	return df

def chunks(l, n):
	for i in range(0, n):
			yield l[i::n]

def split_polars_dataframe(dataframe, n_chunks):

	if n_chunks <= 0:
		raise ValueError("Number of chunks (n_chunks) must be greater than 0.")

	total_rows = len(dataframe)
	rows_per_chunk = total_rows // n_chunks
	chunks = []

	for i in range(n_chunks):
		start_idx = i * rows_per_chunk
		end_idx = (i + 1) * rows_per_chunk if i < n_chunks - 1 else total_rows
		chunk = dataframe.slice(start_idx, end_idx)
		chunks.append(chunk)

	return chunks

def filter_df_on_degrees(df,col,mind=2,only=None):

	if only is not None:
		df.filter(~pl.col(col).is_in(df.groupby([col]).agg(pl.count()).filter((pl.col("count")<mind) & (pl.col(col).is_in(only)))[col].to_list()))
	else:
		df = df.filter(~pl.col(col).is_in(df.groupby([col]).agg(pl.count()).filter(pl.col("count")<mind)[col].to_list()))
	return df

def filter_df_on_edge_weight(df,col,minw=2):

	df = df.filter((pl.col(col)>=minw))
	return df

def filter_on_degrees(g,mind=2,skip_nodes={}):

	to_remove = []
	for n in list(g.iterNodes()):
			if g.degree(n) < mind:
				to_remove.append(n)
	for n in to_remove:
		if n not in skip_nodes:
			g.removeNode(n)
	return g

def filter_on_selected(g,keep_nodes=[]):

	to_remove = []
	new_g = nxk.Graph(g,weighted=g.isWeighted(), directed=g.isDirected(), edgesIndexed=False)
	for n in list(new_g.iterNodes()):
		if n not in keep_nodes:
			to_remove.append(n)
	for n in to_remove:
		if new_g.hasNode(n):
			new_g.removeNode(n)
	return new_g

def filter_on_selected_edges(g,net_idx=None,keep_edges=[]):

	for o,e in list(g.iterEdges()):
		if net_idx is not None:
			if (net_idx[o],net_idx[e]) not in keep_edges and (net_idx[e],net_idx[o]) not in keep_edges:
				g.removeEdge(o, e)
	return g


def filter_on_gc(g,is_nxg=False):
	
	if is_nxg:
		gc = set(sorted(nx.connected_components(g.to_undirected()), key = len, reverse=True)[0])
		for node in list(g.nodes()):
			if not node in gc:
				g.remove_node(node)
	else:
		cug = set(list(nxk.components.ConnectedComponents.extractLargestConnectedComponent(nxk.graphtools.toUndirected(g)).iterNodes()))
		g = filter_on_selected(g,keep_nodes=cug)

	return g

def filter_on_edge_weight(nxg,minw=2):

	for o,e,d in list(nxg.edges(data=True)):
		if d["weight"]<minw:
			nxg.remove_edge(o,e)
	return nxg

def filter_on_metric(g,df,net_idx,metrics,metric="partisan_dist",keep_prop=0.2):

	current_total = g.numberOfNodes()
	keep_nodes = set([k for k,v in sorted(metrics[metric].items(), key=lambda x:x[1], reverse=True)[:int(current_total*keep_prop)]])
	keep_nodes.update(set([k for k,v in metrics["custom"].items() if v < 2]))
	keep_nodes = set([net_idx[n] for n in keep_nodes if n in net_idx])
	g = filter_on_selected(g,keep_nodes=keep_nodes)
	df = df.filter((pl.col("o").is_in(keep_nodes)) & (pl.col("e").is_in(keep_nodes)))

	return g,df,net_idx

def filter_based_on_com(nxg,max_nodes=1000,com_var="com",preferred_metric="pagerank_wEnga^2"):

	print (nxg.number_of_nodes())
	print (nxg.number_of_edges())
	print ("filtering based on com")
	nnodes = len(nxg.nodes())
	if nnodes > max_nodes:
		keep_percent = max_nodes/nnodes
		df = []
		for node,dat in nxg.nodes(data=True):
			dat["actor_platform"]=node
			df.append(dat)
		df = pd.DataFrame(df)
		df = df.sort_values([com_var,preferred_metric],ascending=False)
		com_counts = df[[com_var]].groupby(com_var).size().reset_index()
		df = df[df[com_var].isin(set(list(com_counts[com_counts[0]>int(0.003*max_nodes)][com_var])))]
		filtered_dfs = []
		for group,vals in df.groupby(com_var):
			filtered_dfs.append(vals.head(int(keep_percent*len(vals))))
		filtered_dfs = pd.concat(filtered_dfs)
		keep_nodes=set(list(filtered_dfs["actor_platform"]))
		#keep_nodes.update(set([n for n,d in nxg.nodes(data=True) if "custom" in d and d["custom"]==0]))
		for n in list(nxg.nodes()):
			if n not in keep_nodes:
				nxg.remove_node(n)
		print (nxg.number_of_nodes())
		print (nxg.number_of_edges())
		gc = set(sorted(nx.connected_components(nxg.to_undirected()), key = len, reverse=True)[0])
		for node in list(nxg.nodes()):
			if not node in gc:
				nxg.remove_node(node)
		print (nxg.number_of_nodes())
		print (nxg.number_of_edges())
		if nxg.number_of_edges() > 2000000:
			nxg = filter_on_edge_weight(nxg,minw=3)
			nxg = filter_on_gc(nxg,is_nxg=True)
		print (nxg.number_of_nodes())
		print (nxg.number_of_edges())

		return nxg
	else:
		return nxg

def net_to_compact(g,net_idx):

	tmp_node_map = nxk.graphtools.getContinuousNodeIds(g)
	g = nxk.graphtools.getCompactedGraph(g,tmp_node_map)
	new_net_idx = {k:tmp_node_map[v] for k,v in net_idx.items() if v in tmp_node_map}
	return g,new_net_idx

def create_rev_net_idx(g=None,net_idx=None):
	
	if g is not None:
		rev_net_idx = {v:k for k,v in net_idx.items() if g.hasNode(v)}
	else:
		rev_net_idx = {v:k for k,v in net_idx.items()}
	return rev_net_idx

@timer
def to_uni_matrix(g,ntc,di=False,ncores=-1):

	bi_neighbors = []
	for n in ntc:
		if di:
			ns = np.array(list(g.iterNeighbors(n)),dtype=np.int32)
		else:
			ns = np.array(sorted(list(g.iterNeighbors(n))),dtype=np.int32)
		bi_neighbors.append(ns)
		g.removeNode(n)
	random.shuffle(bi_neighbors)
	edge_m = uni_neighbors(bi_neighbors,ncores=ncores)

	return edge_m

@timer
def edge_df_to_graph(g,edge_m,ncores=-1):
	
	new_g = nxk.graphtools.copyNodes(g)
	for row in edge_m.to_numpy():
		new_g.addEdge(int(row[0]), int(row[1]), w=int(row[2]))
	return new_g

@timer
def create_edge_tuples(df):

	if isinstance(df,pd.DataFrame):
			df = df.groupby(["url","actor_platform"]).size().reset_index()
			e_tups = zip(list(df.iloc[:,0]),list(df.iloc[:,1]),list(df.iloc[:,2]))
	else:
			e_tups = zip(df["o"].to_list(),df["e"].to_list(),df["w"].to_list())

	return e_tups

@timer
def get_collapse_node_list(g,df,net_idx,col="o",not_col="e"):
	
	do_not_collapse = set(df[not_col].to_list())
	to_collapse = set(df[col].to_list())
	nodes_to_collapse = [net_idx[str(n)] for n in to_collapse if g.hasNode(net_idx[str(n)]) and n not in do_not_collapse]	
	return nodes_to_collapse

@timer
def df_to_nxk(s_t,di=False):

	g = nxk.graph.Graph(n=0, weighted=True, directed=di, edgesIndexed=False)
	net_idx = {}
	for o,e,w in s_t:
		o = str(o)
		e = str(e)
		if o not in net_idx:
			o_i = g.addNode()
			net_idx[o]=o_i
		else:
			o_i = net_idx[o]
		if e not in net_idx:
			e_i = g.addNode()
			net_idx[e]=e_i
		else:
			e_i = net_idx[e]
		g.increaseWeight(o_i, e_i, w)

	return g, net_idx

def uni_neighbors(X,ncores=-1):

	if ncores < 1:
		ncores = multiprocessing.cpu_count()-2
	Y = np.concatenate(Pool(ncores).map(_bi_permutation,chunks(X,ncores)),axis=0)
	
	return Y

def _bi_permutation(X):

	Y = []
	for i in range(len(X)):
		x = X.pop()
		y = x[np.stack(np.triu_indices(len(x), k=1), axis=-1)]
		Y.append(y)
	Y = np.vstack(Y)
	return Y

@timer
def edge_m_to_df(edge_m,min_w=1):

	edge_m = pl.LazyFrame(edge_m,schema=["o","e"],schema_overrides={"o":pl.Int32,"e":pl.Int32}).groupby(["o", "e"]).agg(pl.count().cast(pl.Int32).alias('weight')).filter(pl.col("weight")>=min_w)
	return edge_m.collect()

def extract_subgraph(graph, partition, community):
	subgraph = nxk.Graph(directed=graph.isDirected())
	nodes_in_community = {node for node, comm in enumerate(partition.getVector()) if comm == community}
	
	# Create a mapping from original node IDs to new subgraph node IDs
	node_mapping = {original_node: subgraph.addNode() for original_node in nodes_in_community}
	
	for u in nodes_in_community:
		for v in graph.iterNeighbors(u):
			if v in nodes_in_community:
				subgraph.addEdge(node_mapping[u], node_mapping[v])
	
	return subgraph

@timer
def noise_corrected(df, undirected = True,weight="weight"):
	 
	src_sum = df.groupby("o").agg(o_sum=pl.sum(weight))
	trg_sum = df.groupby("e").agg(e_sum=pl.sum(weight))
	df = df.join(src_sum,how="left",on="o")
	df = df.join(trg_sum,how="left",on="e")
	df = df.with_columns(df.select(pl.sum(weight))[weight].alias("n.."))
	df = df.with_columns((((pl.col("o_sum") * pl.col("e_sum")) / pl.col("n..")) * (1 / pl.col("n..")) ).alias("mean_prior_probability"))
	df = df.with_columns((pl.col("n..")/(pl.col("o_sum")*pl.col("e_sum"))).alias("kappa"))
	df = df.with_columns((((pl.col("kappa")*pl.col(weight))-1)/((pl.col("kappa")*pl.col(weight))+1)).alias("score"))
	df = df.with_columns(((1/(pl.col("n..")**2))*(pl.col("o_sum">)*pl.col("e_sum")*(pl.col("n..")-pl.col("o_sum"))*(pl.col("n..")-pl.col("e_sum")))/((pl.col("n..")**2)*(pl.col("n..")-1))).alias("var_prior_probability"))
	df = df.with_columns((((pl.col("mean_prior_probability")**2)/pl.col("var_prior_probability"))*(1-pl.col("mean_prior_probability"))-pl.col("mean_prior_probability")).alias("alpha_prior"))
	df = df.with_columns(((pl.col("mean_prior_probability")/pl.col("var_prior_probability"))*(1-(pl.col("mean_prior_probability")**2))-(1-pl.col("mean_prior_probability"))).alias("beta_prior"))
	df.drop_in_place("mean_prior_probability")
	df = df.with_columns((pl.col("alpha_prior")+pl.col(weight)).alias("alpha_post"))
	df.drop_in_place("alpha_prior")
	df = df.with_columns((pl.col("n..")-pl.col(weight)+pl.col("beta_prior")).alias("beta_post"))
	df.drop_in_place("beta_prior")
	df = df.with_columns((pl.col("alpha_post")/(pl.col("alpha_post")+pl.col("beta_post"))).alias("expected_pij"))
	df.drop_in_place("alpha_post")
	df.drop_in_place("beta_post")
	df = df.with_columns((pl.col("expected_pij")*(1-pl.col("expected_pij"))*pl.col("n..")).alias("variance_nij"))
	df.drop_in_place("expected_pij")
	df = df.with_columns(((1.0/(pl.col("o_sum")*pl.col("e_sum")))-(pl.col("n..")*((pl.col("o_sum")+pl.col("e_sum")) / ((pl.col("o_sum")*pl.col("e_sum"))**2)))).alias("d"))
	df = df.with_columns((pl.col("variance_nij")*(((2*(pl.col("kappa")+(pl.col(weight)*pl.col("d")))) / (((pl.col("kappa")*pl.col(weight))+1)**2))**2)).alias("variance_cij"))
	df = df.with_columns((pl.col("variance_cij")**.5).alias("sdev_cij"))
	if undirected:
		df = df.filter(pl.col("o") <= pl.col("e"))
	return df.select(pl.col(["o", "e", weight, "score", "sdev_cij"]))

@timer
def filter_on_backbone(df,threshold=1.0,max_edges=-1,tol=0.2,skip_nodes=[],weight="weight",remove_only={},skip_strict=False):
	
	if max_edges > 1:
		prev_len_new_df = -1
		if skip_strict:
			new_df = df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0) | ((pl.col("o").is_in(skip_nodes)) & (pl.col("e").is_in(skip_nodes))))
		else:
			new_df = df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0) | (pl.col("o").is_in(skip_nodes)) | (pl.col("e").is_in(skip_nodes)))
		while len(new_df) > max_edges:
			threshold = threshold*(tol*np.log(len(new_df)))
			if skip_strict:
				new_df = new_df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0) | ((pl.col("o").is_in(skip_nodes)) & (pl.col("e").is_in(skip_nodes))))
			else:
				new_df = new_df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0) | (pl.col("o").is_in(skip_nodes)) | (pl.col("e").is_in(skip_nodes)))
			print (len(new_df))
			if len(new_df) == prev_len_new_df:
				break
			prev_len_new_df = len(new_df)
	elif len(remove_only) > 0:
		sp_df = df.filter((pl.col("o").is_in(remove_only)) | (pl.col("e").is_in(remove_only)))
		bf_filter = len(sp_df)
		if skip_strict:
			new_df = df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0) | ((pl.col("o").is_in(skip_nodes)) & (pl.col("e").is_in(skip_nodes))))
		else:
			new_df = df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0) | (pl.col("o").is_in(skip_nodes)) | (pl.col("e").is_in(skip_nodes)))
		org_new_df = len(new_df)
		if max_edges > 0:
			while len(new_df) > org_new_df-int(max_edges*bf_filter):
				threshold = threshold*(tol*np.log(len(new_df)))
				if skip_strict:
					new_df = new_df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0) | ((pl.col("o").is_in(skip_nodes)) & (pl.col("e").is_in(skip_nodes))))
				else:
					new_df = new_df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0) | ((pl.col("o").is_in(skip_nodes)) | (pl.col("e").is_in(skip_nodes))))
				print (len(new_df))
	else:
		if skip_nodes:
			if skip_strict:
				new_df = df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0) | ((pl.col("o").is_in(skip_nodes)) & (pl.col("e").is_in(skip_nodes))))
			else:
				new_df = df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0) | (pl.col("o").is_in(skip_nodes)) | (pl.col("e").is_in(skip_nodes)))
		else:
			new_df = df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0))
	return new_df.select(pl.col(["o", "e", weight]))

def _get_node_coms(g,coms,net_idx=None,reverse=False):

	node_coms = {}
	if net_idx is not None: rev_net_ids = create_rev_net_idx(g,net_idx)
	for subset in coms.getSubsetIds():
		if len(coms.getMembers(subset)) > 9:
			for member in coms.getMembers(subset):
				if reverse:
					if subset not in node_coms: node_coms[subset]=set([])
					node_coms[subset].add(member)
				else:
					if net_idx is not None:
						node_coms[rev_net_ids[member]]=subset
					else:
						node_coms[member]=subset
	return node_coms

def incremental_conductance(g, community,community_idx, node, node_impact_dict, community_weights):
	
	# Calculate the change in the weight of internal edges and cut edges
	internal_edge_weight_change = sum([w for neighbor,w in g.iterNeighborsWeights(node) if neighbor in community])
	cut_edge_weight_change = sum([w for neighbor,w in g.iterNeighborsWeights(node)]) - internal_edge_weight_change
	
	# Calculate the change in conductance
	delta_conductance = (cut_edge_weight_change - internal_edge_weight_change) / community_weights[community_idx]

	# Record the impact of the node on the change in conductance
	node_impact_dict[community_idx] =  delta_conductance
	return node_impact_dict

def _find_best_com_fit(g1,gam,its=10):

	all_coms = {}
	coms_scores = defaultdict(list)
	for r in range(its):
		coms = nxk.community.detectCommunities(g1, algo=nxk.community.PLM(g1, refine=True, gamma=gam+random.uniform(-1, 1)*0.05, par='balanced', maxIter=128, turbo=True, recurse=True),inspect=False)
		all_coms[r]=coms
	if its > 1:
		for r1,coms1 in list(all_coms.items()):
			for r2,coms2 in list(all_coms.items()):
				if r1 != r2:
					jac_score = nxk.community.JaccardMeasure().getDissimilarity(g1, coms1, coms2)
					coms_scores[r1].append(jac_score)
		coms_scores = {k:np.mean(np.array(v)) for k,v in coms_scores.items()}
		coms = all_coms[sorted(coms_scores.items(), key=lambda x:x[1], reverse=False)[0][0]]
	print (f"Best partition: (average similarity: {round((1-sorted(coms_scores.items(), key=lambda x:x[1], reverse=False)[0][1])*100,3)}%)")
	nxk.community.inspectCommunities(coms, g1)
	return coms

def _find_best_modularity_gamma(g1,gamma_range):

	gammas = np.linspace(gamma_range[0], gamma_range[1], num=30)
	com_mods = {}
	prev_score = 0
	for gam in gammas:
		mod = nxk.community.Modularity().getQuality(nxk.community.detectCommunities(g1, algo=nxk.community.PLM(g1, refine=True, gamma=gam, par='balanced', maxIter=128, turbo=True, recurse=True),inspect=False),g1)
		score = (np.log(gam+1))*mod
		com_mods[gam]=score
		if score < prev_score:
			break
		prev_score = score
		print (str(gam)+" : "+str(mod)+" - "+str(score))
	return sorted(com_mods.items(), key=lambda x:x[1], reverse=True)[0][0]

@timer
def _get_coms(g,org_deg_df,net_idx=None,base_deg=1,max_deg=1,gamma_range=[0.95,3.0],custom_gamma=None):

	if net_idx is not None: rev_net_ids = create_rev_net_idx(g,net_idx)
	final_node_coms = {}
	final_node_coms_conds = {}
	if g.isDirected(): g = nxk.graphtools.toUndirected(g)
	if max_deg == 4:
		base_nodes = set([net_idx[n] for n,c in org_deg_df.items() if n in net_idx and c <= base_deg])
		print (len(base_nodes))
		base_g = filter_on_selected(g,base_nodes)
		if custom_gamma is not None:
			gam = custom_gamma
		else:
			gam = _find_best_modularity_gamma(base_g,gamma_range)
		coms = _find_best_com_fit(base_g,gam,its=12)
		com_nodes = _get_node_coms(base_g,coms,reverse=True)
		node_coms = {n:set([]) for n in g.iterNodes()}
		for com, nodes in com_nodes.items():
			print (com)
			new_assigned = set(nxk.scd.LocalTightnessExpansion(g, alpha=1.0).expandOneCommunity(list(nodes)))
			com_nodes[com].update(set([n for n in new_assigned if (n in nodes) or (n not in nodes and n not in base_nodes)]))
			for n in com_nodes[com]:
				node_coms[n].add(com)
		community_weights = {community:sum([sum([w for neighbor,w in g.iterNeighborsWeights(n)]) for n in com_nodes[community]]) for community in com_nodes.keys()}
		new_assigned_coms = {k:0 for k in com_nodes.keys()}
		new_assigned_coms[-1]=0
		n_count = 0
		for n,coms in node_coms.items():
			n_count+=1
			if n_count % 10000 == 0: print (n_count)
			if len(coms) == 0:
				real_com = -1
			elif len(coms) == 1:
				real_com = list(coms)[0]
			else:
				com_changes = {}
				for com in coms:
					com_changes = incremental_conductance(g,com_nodes[com],com,n,com_changes,community_weights)
				real_com = sorted(com_changes.items(), key=lambda x:x[1], reverse=False)[0][0]
			final_node_coms[n]=real_com
			if n not in base_nodes: new_assigned_coms[real_com]+=1
		print (new_assigned_coms)
		print (len(new_assigned_coms))
	else:
		if custom_gamma is None:
			gam = _find_best_modularity_gamma(g,gamma_range)
		else:
			gam = custom_gamma
		coms = _find_best_com_fit(g,gam,its=6)
		com_nodes = _get_node_coms(g,coms,reverse=True)
		final_node_coms = _get_node_coms(g,coms,reverse=False)
		
	for com in com_nodes.keys():
		org_com_cond = nxk.scd.SetConductance(g,com_nodes[com])
		org_com_cond.run()
		org_com_cond = org_com_cond.getConductance()
		for n in com_nodes[com]:
			final_node_coms_conds[n]=org_com_cond
	
	final_node_coms = {rev_net_ids[k]:v for k,v in final_node_coms.items()}
	final_node_coms_conds = {rev_net_ids[k]:v for k,v in final_node_coms_conds.items()}
	print (len({k for k,v in final_node_coms.items() if v != -1}))
	print (len({k for k,v in final_node_coms.items() if v == -1}))

	return final_node_coms,final_node_coms_conds

@timer
def get_metrics(g,net_idx=None):

	metrics = {}
	if net_idx is not None: rev_net_ids = create_rev_net_idx(g,net_idx)
	metrics["deg_c"]=nxk.centrality.DegreeCentrality(g, normalized=True)
	metrics["pagerank"]=nxk.centrality.PageRank(g,tol=1e-3, normalized=True)
	#metrics["eig_c"]=nxk.centrality.EigenvectorCentrality(g,tol=1e-3)
	metrics["btw_c"]=nxk.centrality.EstimateBetweenness(g, 10, normalized=True, parallel=True)
	#metrics["btw_c"]=nxk.centrality.Betweenness(g,normalized=True)
	out_metrics = {k:{} for k in metrics.keys()}
	out_metrics.update({"in_degree":{},"out_degree":{}})
	for m,f in metrics.items():
		print (m)
		f.run()
		print (m)
		for n in g.iterNodes():
			score = f.score(n)
			if net_idx is not None:
				n = rev_net_ids[n]
			out_metrics[m][n]=score
	for n in g.iterNodes():
		in_deg = g.degreeIn(n)
		out_deg = g.degreeOut(n)
		if net_idx is not None:
			n = rev_net_ids[n]
		out_metrics["in_degree"][n]=in_deg
		out_metrics["out_degree"][n]=out_deg
	return out_metrics

def get_norm_edge_weights_out(nxg):

	no_e_w = {}
	for o,e,w in nxg.edges(data=True):
		no_e_w[(o,e)]=w["weight"]/nxg.out_degree(o,"weight")
	return no_e_w

def get_norm_edge_weights_full(nxg):

	no_e_w = {}
	max_indeg = max([nxg.in_degree(n,"weight") for n in nxg.nodes()])
	max_outdeg = max([nxg.out_degree(n,"weight") for n in nxg.nodes()])
	for o,e,w in nxg.edges(data=True):
		no_e_w[(o,e)]=(w["weight"]/(nxg.out_degree(o,"weight")))*(w["weight"]/(nxg.in_degree(e,"weight")))
	return no_e_w

def conductance(graph, community_nodes):
	cut_weight = sum(graph.weight(u, v) for u in community_nodes for v in graph.iterNeighbors(u) if v not in community_nodes)
	community_edge_weight = sum(graph.weight(u, v) for u in community_nodes for v in graph.iterNeighbors(u))

	# Compute conductance
	return cut_weight / community_edge_weight

def get_conductance_scores(g,coms,verbose=False):

	scores = {}
	for com in set(coms.getSubsetIds()):
		nodes_in_community = {node for node in coms.getMembers(com)}
		comm_conductance = conductance(g, nodes_in_community)
		if verbose:
			print(f"Conductance of community {com}:", comm_conductance)
		scores[com]=comm_conductance
	return scores

def set_metrics(g,metrics,enga_trans=True,enrich=True):

	for met,vals in metrics.items():
		nx.set_node_attributes(g,vals,met)

	return g

@timer
def filter_urls_based_on_texts(df,num_cores=1):
	
	ut_groups = []
	df = df.sort(['o','text'],descending=False) 
	url_g_df = df.groupby("o").agg(pl.col("text")).sort('o',descending=False)
	url_ts = list(zip(url_g_df["o"].to_list(),url_g_df["text"].to_list()))
	if num_cores > 1:
		results = Pool(num_cores).map(_group_text_similar_urls,chunks(url_ts,num_cores))
	else:
		results = [_group_text_similar_urls(url_ts)]
	for result in results:
		ut_groups.extend(result)
	df = df.with_columns(pl.Series(name="text_group", values=ut_groups))
	df = df.with_columns((pl.col("o")+"_"+pl.col("text_group").cast(pl.Utf8)).alias("text_based_o"))
	df.replace("o",pl.Series(df["text_based_o"].to_list()))
	df.drop_in_place("text")
	df.drop_in_place("text_based_o")
	df.drop_in_place("text_group")
	return df

def output_graph_simple(g,net_idx,save_as,add_custom=None,node_data=None,edge_data=None,return_nx=False):

	rev_net_idx = create_rev_net_idx(g,net_idx)
	g = filter_on_selected(g,keep_nodes=[n for n in net_idx.values()])
	g = nxk.nxadapter.nk2nx(g)
	g = nx.relabel_nodes(g,rev_net_idx)
	if add_custom is not None: nx.set_node_attributes(g,add_custom,"custom")
	if node_data is not None:
		for name,dat in node_data.items():
			if name in ["n_coor_in","n_coor_out"]:
				nx.set_node_attributes(g,0,name)
			nx.set_node_attributes(g,dat,name)
	if edge_data is not None:
		for name,dat in edge_data.items():
			if name != "weight":
				nx.set_edge_attributes(g,"None",name)
			for edge_t,val in dat.items():
				if g.has_edge(edge_t[0],edge_t[1]):
					nx.set_edge_attributes(g,{(edge_t[0],edge_t[1]):val},name)
	print ("WRITING")
	nx.write_gexf(g,save_as)
	if return_nx:
		return g

def output_graph(g,net_idx,save_as,add_custom=None,metrics=None,affinity_map=None,add_edge_data=None,default_com="com_1.0",filter_on_com=True,add_edge_dom=None,exp_filter=True):

	def set_show_labels(nxg,com_var="com",preferred_metric="pagerank_wEnga^2"):

		df = []
		for node,dat in nxg.nodes(data=True):
			dat["actor_platform"]=node
			df.append(dat)
		df = pd.DataFrame(df)
		df = df.sort_values([com_var,preferred_metric],ascending=False)
		com_counts = df[[com_var]].groupby(com_var).size().reset_index()
		filtered_dfs = []
		for group,vals in df.groupby(com_var):
			filtered_dfs.append(vals.head(int(10)))
		filtered_dfs = pd.concat(filtered_dfs)
		keep_nodes=set(list(filtered_dfs["actor_platform"]))
		keep_nodes.update(set([n for n,d in nxg.nodes(data=True) if "custom" in d and d["custom"]==0]))
		nx.set_node_attributes(nxg,False,"show_label")
		nx.set_node_attributes(nxg,{n:True for n in keep_nodes},"show_label")

		return nxg

	def map_sigmoid(input):

		return 1 / (1 + math.exp(0.000005 * (input - 95000)))

	CMAX = 16000
	rev_net_idx = create_rev_net_idx(g,net_idx)
	g = nxk.nxadapter.nk2nx(g)
	g = nx.relabel_nodes(g,rev_net_idx)
	#nx.set_node_attributes(g,rev_net_idx,"Label")
	if add_custom is not None: nx.set_node_attributes(g,add_custom,"custom")
	if metrics is not None: g = set_metrics(g,metrics,enrich=False,enga_trans=True)
	if filter_on_com and g.number_of_nodes() <= CMAX:
		g = filter_based_on_com(g,max_nodes=CMAX,com_var=default_com)
	if filter_on_com and g.number_of_nodes() > CMAX and g.number_of_nodes() < 1000000:
		#g = filter_based_on_com(g,max_nodes=int(g.number_of_nodes()*map_sigmoid(g.number_of_nodes())),com_var="com_1.0")
		g = filter_based_on_com(g,max_nodes=int(4000),com_var=default_com)
	print (len(g.nodes()))
	if exp_filter:
		print ("dsa")
		remove_nodes = []
		for node,dat in g.nodes(data=True):
			if dat[default_com] == -1:
				remove_nodes.append(node)
		g.remove_nodes_from(remove_nodes)
	print (len(g.nodes()))
	if add_edge_data is not None:
		bc_scores = {(o,e):s for o,e,s in zip(add_edge_data["o"].to_list(),add_edge_data["e"].to_list(),add_edge_data["score"].to_list())}
		bc_scores = {(rev_net_idx[k[0]],rev_net_idx[k[1]]):(v+1)*0.5 for k,v in bc_scores.items() if k[0] in rev_net_idx and k[1] in rev_net_idx}
		norm_edge_weights_out = get_norm_edge_weights_out(g)
		norm_edge_weights_full = get_norm_edge_weights_full(g)
		nx.set_edge_attributes(g,norm_edge_weights_out,"norm_weight_out")
		nx.set_edge_attributes(g,norm_edge_weights_full,"norm_weight_full")
		nx.set_edge_attributes(g,bc_scores,"bc_score")
		if add_edge_dom:
			edge_fav_dom={}
			for o,e in zip(add_edge_data["o"].to_list(),add_edge_data["e"].to_list()):
				edge=(rev_net_idx[o],rev_net_idx[e])
				if edge not in edge_fav_dom and (edge[0] in add_edge_dom and edge[1] in add_edge_dom):
					e0_d = add_edge_dom[edge[0]]
					e1_d = add_edge_dom[edge[1]]
					be_d = {k:(e0_d[k]["count"]+e1_d[k]["count"])/2 for k in (e0_d.keys() & e1_d.keys())}
					edge_fav_dom[edge]=str(sorted(be_d.items(), key=lambda x:x[1], reverse=True)[:3])
			nx.set_edge_attributes(g,edge_fav_dom,"fav_doms")
	#g = set_show_labels(g,com_var=default_com)
	print ("writing graph file")
	nx.write_gexf(g,save_as)

def output_data(metrics,net_idx,main_path,title,path_step):

	docs = []
	mets = list(metrics.keys())
	nodes = set(list(metrics[mets[0]].keys()))
	print (len(nodes))
	#nodes = nodes.intersection(*[set(list(v.keys())) for k,v in metrics.items()])
	nodes.union(*[set(list(v.keys())) for k,v in metrics.items()])
	print (len(nodes))
	for node in nodes:
		if node in net_idx:
			doc = {"actor_platform":node}
			for met in mets:
				if node not in metrics[met]:
					doc[met]=0
				else:
					doc[met]=metrics[met][node]
			docs.append(doc)
	df = pl.from_dicts(docs)
	df.write_csv(main_path+f"/{title}_{path_step}steps_data.csv")