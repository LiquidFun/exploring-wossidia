# import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
import json
import random

import cmasher
import folium
import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

mv = folium.Map(location=[54, 12])

colors = {
    'red' : "#cb3b28",
    'blue' : "#36a3d3",
    'green' : "#6ca724",
    'purple' : "#ac4397",
    'orange' : "#ed922e",
    'darkred' : "#9a3033",
    'lightred' : "#ff8979",
    'beige' : "#ffc98f",
    'darkblue' : "#00639f",
    'darkgreen' : "#6f7e23",
    'cadetblue' : "#406573",
    'darkpurple' : "#573767",
    'white' : "#ffffff",
    'pink' : "#ff8ae8",
    'lightblue' : "#85daff",
    'lightgreen' : "#baf671",
    'gray' : "#575757",
    'black' : "#2f2f2f",
    'lightgray' : "#a3a3a3",
}

assert list(colors)[0] == "red"

# colors = cmasher.take_cmap_colors("tab20", None, return_fmt="hex")


skip = "hexe witch witches hexen orte varia weerwolf weerwolven varulv werewolf " \
       "werwolf wolf wolven hond avond heks forhekse kjælling kjærn hekseri hare" \
       "when where wann wo"

USE_PCA = True
COORD_MULTIPLIER = 1
NUM_CLUSTERS = 17

# def all_places():
#     with open("am_place.json") as file:
#         data = json.load(file)
#         for place in data:
#             try:
#                 lat = place['attributes']['lat']
#                 lon = place['attributes']['lon']
#                 name = place['attributes']['name']
#                 folium.Marker([lat, lon], popup=name).add_to(mv)
#             except KeyError:
#                 pass

# def places_from_dataset(file_path):
#     #witches = pd.read_csv("ISEBEL-Datasets/witches-nodes.csv")
#     with open(file_path) as file:
#         for line in file.readlines():
#             split = line.split(',')
#             try:
#                 if split[1] == "place":
#                     lat = float(split[3])
#                     lon = float(split[5])
#                     name = split[7]
#                     folium.Marker([lat, lon], popup=name).add_to(mv)
#                     # print(lat, lon, name)
#             except (IndexError, ValueError):
#                 pass
    #print(witches)

def plot_on_map(graph):
    for node in graph.nodes:
        if graph.nodes[node]['label'] == "place":
            # print(graph.nodes[node])
            try:
                if "latitude" in graph.nodes[node]:
                    lat = float(graph.nodes[node]['latitude'])
                    lon = float(graph.nodes[node]['longitude'])
                else:
                    lat = float(graph.nodes[node]['lat'])
                    lon = float(graph.nodes[node]['long'])
                lat += (random.random() - 0.5) * 0.001
                lon += (random.random() - 0.5) * 0.001
                name = graph.nodes[node]['name'] + "\n\n" + graph.nodes[node].get("info", "")
                # print(graph.nodes[node]['cluster'])
                icon = folium.Icon(color=list(colors)[graph.nodes[node]['cluster']])
                # print(icon)
                folium.Marker([lat, lon], popup=name, icon=icon).add_to(mv)
            except ValueError:
                pass

def make_graph(node_path, edge_path):
    g = nx.Graph()

    keyword_counter = Counter()
    with open(node_path) as file:
        for line in file.readlines()[1:]:
            id, label, *properties = line.strip().split(',')
            properties = {key: value for key, value in zip(properties[::2], properties[1::2])}
            if label == "keyword":
                kw = properties['name'].strip().lower().replace(" ", "_").replace('"', '')
                keyword_counter[kw] += 1
    # common_keywords = [a[0] for a in keyword_counter.most_common(100)]
    # common_keywords = [k for k, count in keyword_counter.items() if count >= 2]

    with open(node_path) as file:
        for line in file.readlines()[1:]:
            id, label, *properties = line.strip().split(',')
            properties = {key: value for key, value in zip(properties[::2], properties[1::2])}
            #if label == "keyword" and properties["name"] not in common_keywords:
            #    continue
            if label == "place":
                try:
                    lat = float(properties.get('lat', properties.get('latitude', 0)))
                    long = float(properties.get('long', properties.get('longitude', 0)))
                except ValueError:
                    pass
                if not (54.500261 > lat > 53.027714 and  10.863210 < long < 14.627471):
                    continue

            try:
                int(id)
                g.add_node(id, label=label, **properties)
            except ValueError:
                pass

    with open(edge_path) as file:
        lines = file.readlines()[1:]
        counter = Counter()
        for line in lines:
            id1, id2, *_ = line.strip().split(',')
            counter[id1] += 1
            counter[id2] += 1
    common = [c[0] for c in counter.most_common(1000)]
    with open(edge_path) as file:
        for line in file.readlines()[1:]:
            id1, id2, label, *_ = line.strip().split(',')
            if id1 in common or id2 in common or True:
                if id1 in g.nodes and id2 in g.nodes:
                    g.add_edge(id1, id2, label=label)
                    for i1, i2 in [(id1, id2), (id2, id1)]:
                        if g.nodes[i1]["label"] != "place":
                            if "info" not in g.nodes[i1]:
                                g.nodes[i1]["info"] = ""
                            g.nodes[i1]["info"] += f"{i2}:{g.nodes[i2]['label']}\n"
        for e1, e2 in g.edges:
            for i1, i2 in [(e1, e2), (e2, e1)]:
                if g.nodes[i1]["label"] == "place":
                    if "info" not in g.nodes[i1]:
                        g.nodes[i1]["info"] = ""
                    # g.nodes[i1]["info"] += f"{i2}:{g.nodes[i2]['label']}\n"
                    for i3 in g.neighbors(i2):
                        if i3 != i1:
                            # g.nodes[i1]["info"] += f"  {i3}:{g.nodes[i3]['label']}"
                            if g.nodes[i3]['label'] == "keyword":
                                # g.nodes[i1]["info"] += f":{g.nodes[i3]['name']}\n"
                                if "keywords" not in g.nodes[i1]:
                                    g.nodes[i1]["keywords"] = set()
                                kw = g.nodes[i3]['name'].strip().lower().replace(" ", "_").replace('"', '')
                                #if kw not in common_keywords:
                                #    continue
                                if kw not in skip and kw not in g.nodes[i1]["keywords"]:
                                    g.nodes[i1]["keywords"].add(kw)
                                    g.nodes[i1]["info"] += f"[{kw}] "
                    # print("Edge", id1, id2)
    return g

dataset = "witches"


#     g = make_graph(f"ISEBEL-Datasets/{dataset}-nodes.csv", f"ISEBEL-Datasets/{dataset}-edges.csv")
#     #nx.draw(g, node_size=1)
#     #plt.show()
#     from node2vec import Node2Vec
#     node2vec = Node2Vec(g, dimensions=3, walk_length=10, num_walks=10, workers=6)
#     model = node2vec.fit(window=10, min_count=1, batch_words=4)
#     model.wv.save_word2vec_format("node2vec.txt")

# def cluster_node2vec_graph():
#     g = make_graph(f"ISEBEL-Datasets/{dataset}-nodes.csv", f"ISEBEL-Datasets/{dataset}-edges.csv")
#     x = np.loadtxt("node2vec.txt", skiprows=1)
#     x = x[x[:, 0].argsort()]
#     x = x[0:x.shape[0], 1:x.shape[1]]

    # kmeans = KMeans(n_clusters=len(colors), random_state=0).fit(x)
    # labels = kmeans.labels_  # get the cluster labels of the nodes.
    # for label, node in zip(labels, g.nodes):
    #     g.nodes[node]['cluster'] = label
    # print(labels)
    # print(Counter(labels))
    # plot_on_map(g)


# def plot_array_as_scatter_plot(x, clusters):
    # x is an array with 2 columns of x and y coordinates
    # fixed_colors = colors.copy()
    # fixed_colors[fixed_colors.index("darkpurple")] = "#5a0d6d"

def cluster_graph_as_table():
    g = make_graph(f"ISEBEL-Datasets/{dataset}-nodes.csv", f"ISEBEL-Datasets/{dataset}-edges.csv")
    keyword_counts = Counter()
    keywords = set()
    for i in g.nodes:
        if "keywords" in g.nodes[i]:
            keywords |= set(g.nodes[i]["keywords"])
            keyword_counts.update(g.nodes[i]["keywords"])
    for kw, count in keyword_counts.items():
        if count <= 10:
            keywords.remove(kw)
    # print(list(zip(keywords, range(100000))))
    keywords = dict(zip(keywords, range(100000)))
    j = len(keywords)
    len_places = len([n for n in g.nodes if g.nodes[n]["label"] == "place"])
    x = np.zeros((len_places, j + 2))
    index = 0
    place_nodes = []
    for i, node in enumerate(g.nodes):
        if g.nodes[node]["label"] == "place":
            place_nodes.append(g.nodes[node])
            if "keywords" in g.nodes[node]:
                for keyword in g.nodes[node]["keywords"]:
                    if keyword in keywords:
                        x[index, keywords[keyword]] = 1
            try:
                if "latitude" in g.nodes[node]:
                    lat = float(g.nodes[node]['latitude'])
                    lon = float(g.nodes[node]['longitude'])
                    x[index, j] = lat
                    x[index, j+1] = lon
                elif "lat" in g.nodes[node]:
                    lat = float(g.nodes[node]['lat'])
                    lon = float(g.nodes[node]['long'])
                    x[index, j] = lat
                    x[index, j+1] = lon
            except ValueError:
                pass
            index += 1
    assert len(x[:, j:].min(axis=0)) == 2
    # print(x[x[:, j] != 0, j])
    x[x[:, j] == 0, j] = x[x[:, j] != 0, j].min()
    x[x[:, j+1] == 0, j+1] = x[x[:, j+1] != 0, j+1].min()
    print(x[:, j:].min(axis=0))
    x[:, j:] -= x[:, j:].min(axis=0)
    print(x[:, j:].max(axis=0))
    x[:, j:] /= x[:, j:].max(axis=0)
    x[:, j:] *= COORD_MULTIPLIER
    # x *= 1000
    # Do pca on the array x
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(x)
    if USE_PCA:
        pca = PCA(n_components=2)
        pca.fit(x)
        x = pca.transform(x)
    cols = [list(colors.values())[cluster] for cluster in kmeans.labels_]
    plt.scatter(x[:, 0], x[:, 1], s=80, c=cols)
    if USE_PCA:
        index_to_keyword = dict(zip(keywords.values(), keywords.keys()))
        # Get centers of kmeans clusters
        centers = kmeans.cluster_centers_
        centers = pca.transform(centers)
        # Get the inverse transform of the centers
        inverses = pca.inverse_transform(centers)[:, :-2]
        print(len(inverses))
        for index, center in enumerate(centers):
            indices_of_best_keywords = np.argpartition(-inverses[index], kth=5)[:5]
            indices_of_best_keywords = indices_of_best_keywords[np.argsort(inverses[index][indices_of_best_keywords])[::-1]]
            for i, kw_index in enumerate(indices_of_best_keywords):
                plt.text(center[0], center[1]-0.05*i, index_to_keyword[kw_index], fontsize=8-i)
            plt.scatter(center[0], center[1], s=100, marker="x", c=list(colors.values())[index])
        # for keyword, count in (keyword_counts | {"BLOCKSBERG": 1}).items(): #.most_common(NUM_CLUSTERS):
        #     zeros = np.zeros((1, j + 2))
        #     if keyword == "BLOCKSBERG":
        #         for kw in keywords:
        #             if "blocksberg" in kw:
        #                 zeros[0, keywords[kw]] = 1
        #     else:
        #         zeros[0, keywords[keyword]] = 1
        #     pca_point = pca.transform(zeros)
        #     plt.text(pca_point[0, 0], pca_point[0, 1], keyword, fontsize=6)
        #     plt.scatter(pca_point[0, 0], pca_point[0, 1], s=100, marker="x", c="red")
        plt.title("Clusters")
        plt.show()

    # cluster_size = 100
    # centers = kmeans.cluster_centers_
    # centers = centers.reshape(-1, 1, x.shape[-1]).repeat(cluster_size, 1).reshape(-1, x.shape[-1])
    # distance_matrix = cdist(x, centers)
    # labels = clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size

    labels = kmeans.labels_  # get the cluster labels of the nodes.
    for label, node in zip(labels, place_nodes):
        node['cluster'] = label
    print(labels)
    print(Counter(labels))
    plot_on_map(g)
    return g


def plot_cluster_kw_counts(graph):
    totals = defaultdict(int)
    counts = defaultdict(lambda: defaultdict(int))
    for n in graph.nodes:
        node = graph.nodes[n]
        for kw in node.get("keywords", []):
            counts[node["cluster"]][kw] += 1
            totals[kw] += 1
    print(counts)
    print(totals)
    places = [n for n in graph.nodes if graph.nodes[n]["label"] == "place"]
    lat = 54.5
    lon = 11.4
    for cluster, kwcounts in counts.items():
        lon += .1
        len_cluster = len([n for n in places if graph.nodes[n]["cluster"] == cluster])
        curr_counts = sorted([(v, k) for k, v in kwcounts.items()], reverse=True)[:20]
        name = ' \n'.join([f"[{k}/{totals[v]}:{v}]" for k, v in curr_counts])
        name += f"\n{len_cluster}/{len(places)}\n"
        icon = folium.Icon(color=list(colors)[cluster])
        folium.Marker([lat, lon], popup=name, icon=icon).add_to(mv)


# make_node2vec()
graph = cluster_graph_as_table()
plot_cluster_kw_counts(graph)

# all_places()
#for path in Path("ISEBEL-Datasets").glob("*.csv"):
    #places_from_dataset(path)


mv.save("index.html")
print("Done")
