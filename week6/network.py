import argparse
import csv
import os
from collections import Counter
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)
from pathlib import Path

def load_edgelist(path):
    # Load edgelist csv as a list of tuples
    with open(path, newline='') as fh_in:
        reader = csv.reader(fh_in)
        edgelist = [tuple(row) for row in reader]

    # Remove headers if these are present
    if edgelist[0] == ('nodeA', 'nodeB'):
        edgelist = edgelist[1:]
    
    return edgelist


def calc_weighted_edges(edgelist):
    # Sum up how many times every unique combination of edges occur
    # and set this number as the weight of the unique edge
    weighted_edges = []

    for key, value in Counter(edgelist).items():
        nodeA = key[0]
        nodeB = key[1]
        weight = value
        weighted_edges.append((nodeA, nodeB, weight))
    
    return weighted_edges

def coreference_resolution(edgelist):
    '''
    Removes edges where the whole value of one node exists inside the other
    this removes edges such as ('Trump', 'Donald Trump'), which is often what we want, since
    you'd usually not use just the last name to refer to another Trump when also talking
    about Donald Trump in the same text, BUT it is possible that this could be
    someone else from the same family.
    '''

    '''
    This also does not collapse duplicates if they are not in the same edge,
    e.g.: ('Trump', 'Clinton') and ('Trump', 'Hillary Clinton') will still be
    two separate edges, although they should not be.
    '''
    filtered_edgelist = edgelist.copy()

    # It is faster to remove items when we get the index with eumerate
    for i, edge in enumerate(filtered_edgelist):
        if edge[0] in edge[1] or edge[1] in edge[0]:
            del filtered_edgelist[i]

    return filtered_edgelist


def create_outpath(dirname, filename):
    # Create viz folder from where this script is run from if it does not already exist where the script is
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Returns file path inside new folder
    return os.path.join(dirname, filename)

def calc_centrality_measures(graph):
    ev = nx.eigenvector_centrality(graph)
    bc = nx.betweenness_centrality(graph)
    dg = graph.degree

    return (ev, bc, dg)

def main(edgelist_path, min_weight = 0, demo = False):
    # Turn csv into list of edges
    edgelist = load_edgelist(edgelist_path)

    # To just demo the script, only run 10k edges
    if demo:
        edgelist = edgelist[:10000]

    # Remove some of the duplicates
    edgelist = coreference_resolution(edgelist)
    
    # Sum up unique edges and save as their weight
    weighted_edges = calc_weighted_edges(edgelist)

    # Turn weighted edgelist into dataframe
    edges_df = pd.DataFrame(weighted_edges, columns=["nodeA", "nodeB", "weight"])
    # Filter off the given amount of edges by min_weight
    weight_filtered_df = edges_df[edges_df["weight"] > min_weight]
    
    # Create folder and visualization path
    graph_outpath = create_outpath('viz', 'network.png')

    # Create a graph object
    graph = nx.from_pandas_edgelist(weight_filtered_df, 'nodeA', 'nodeB', ["weight"])
    
    # Draw and save it as a figure in ./viz/network.png
    nx.draw(graph, with_labels=True, node_size=20, font_size=10)
    plt.savefig(graph_outpath, dpi=300, bbox_inches="tight")


    # Eigenvector, betweenness, degree
    ev, bc, dg = calc_centrality_measures(graph)

    # Convert to dataframes
    ev_df = pd.DataFrame(data = ev.items(), columns = ('node', 'eigenvector_centrality'))
    bc_df = pd.DataFrame(data = bc.items(), columns = ('node', 'betweenness_centrality'))
    dg_df = pd.DataFrame(data = dg, columns = ('node', 'degree'))

    # Join dataframes into one table on the 'node' column
    measures_df = ev_df.join(bc_df.set_index('node'), on = 'node').join(dg_df.set_index('node'), on = 'node')

    # Create outpath and save as csv
    measures_outpath = create_outpath('output', 'measures.csv')
    measures_df.to_csv(measures_outpath, index = False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Generate network visualizations and centrality measures from edgelists.")
    # Edgelist csv path
    parser.add_argument("-p", "--path", required = True, type = Path, help = "Path to the input edgelist csv file. The file must have two columns: 'nodeA' and 'nodeB'.")

    # Min weight for filtering
    parser.add_argument("-w", "--min-weight", type = int, default = 0, help = "Integer representing the minimum weight for an edge to have to not be filtered off the data.")
    
    parser.add_argument("-d", "--demo", dest = 'demo', action = 'store_const', const = True, default = False, help = "Add this flag to only work on 10,000 edges.")

    args = parser.parse_args()

    main(edgelist_path = args.path, min_weight = args.min_weight, demo = args.demo)
