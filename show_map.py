# import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
import json
import random

import folium
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

mv = folium.Map(location=[54, 12])

def all_places():
    with open("am_place.json") as file:
        data = json.load(file)
        for place in data:
            try:
                lat = place['attributes']['lat']
                lon = place['attributes']['lon']
                name = place['attributes']['name']
                folium.Marker([lat, lon], popup=name).add_to(mv)
            except KeyError:
                pass

def places_from_dataset(file_path):
    #witches = pd.read_csv("ISEBEL-Datasets/witches-nodes.csv")
    with open(file_path) as file:
        for line in file.readlines():
            split = line.split(',')
            try:
                if split[1] == "place":
                    lat = float(split[3])
                    lon = float(split[5])
                    name = split[7]
                    folium.Marker([lat, lon], popup=name).add_to(mv)
                    # print(lat, lon, name)
            except (IndexError, ValueError):
                pass
    #print(witches)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
     'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

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
                icon = folium.Icon(color=colors[graph.nodes[node]['cluster']])
                # print(icon)
                folium.Marker([lat, lon], popup=name, icon=icon).add_to(mv)
            except ValueError:
                pass

skip = "hexe witch witches hexen orte varia weerwolf weerwolven varulv werewolf werwolf wolf wolven hond avond heks forhekse kjælling kjærn hekseri hare"

def make_graph(node_path, edge_path):
    g = nx.Graph()

    keyword_counter = Counter()
    with open(node_path) as file:
        for line in file.readlines()[1:]:
            id, label, *properties = line.strip().split(',')
            properties = {key: value for key, value in zip(properties[::2], properties[1::2])}
            if label == "keyword":
                keyword_counter[properties['name']] += 1
    # common_keywords = [a[0] for a in keyword_counter.most_common(100)]

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
                #if not (54.500261 > lat > 53.027714 and  10.863210 < long < 14.627471):
                #    continue

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
                                if kw not in skip and kw not in g.nodes[i1]["keywords"]:
                                    g.nodes[i1]["keywords"].add(kw)
                                    g.nodes[i1]["info"] += f"[{kw}] "
                    # print("Edge", id1, id2)
    return g

dataset = "witches-dk"

def make_node2vec():
    g = make_graph(f"ISEBEL-Datasets/{dataset}-nodes.csv", f"ISEBEL-Datasets/{dataset}-edges.csv")
    #nx.draw(g, node_size=1)
    #plt.show()
    from node2vec import Node2Vec
    node2vec = Node2Vec(g, dimensions=3, walk_length=10, num_walks=10, workers=6)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    model.wv.save_word2vec_format("node2vec.txt")

def cluster_graph():
    g = make_graph(f"ISEBEL-Datasets/{dataset}-nodes.csv", f"ISEBEL-Datasets/{dataset}-edges.csv")
    x = np.loadtxt("node2vec.txt", skiprows=1)
    x = x[x[:, 0].argsort()]
    x = x[0:x.shape[0], 1:x.shape[1]]

    kmeans = KMeans(n_clusters=10, random_state=0).fit(x)
    labels = kmeans.labels_  # get the cluster labels of the nodes.
    for label, node in zip(labels, g.nodes):
        g.nodes[node]['cluster'] = label
    print(labels)
    print(Counter(labels))
    plot_on_map(g)


def cluster_graph_as_table():
    g = make_graph(f"ISEBEL-Datasets/{dataset}-nodes.csv", f"ISEBEL-Datasets/{dataset}-edges.csv")
    keywords = set()
    for i in g.nodes:
        if "keywords" in g.nodes[i]:
            keywords |= set(g.nodes[i]["keywords"])
    # print(list(zip(keywords, range(100000))))
    keywords = dict(zip(keywords, range(100000)))
    j = len(keywords)
    x = np.zeros((len(g.nodes), j + 2))
    for i, node in enumerate(g.nodes):
        if "keywords" in g.nodes[node]:
            for keyword in g.nodes[node]["keywords"]:
                x[i, keywords[keyword]] = 1
        try:
            if "latitude" in g.nodes[node]:
                lat = float(g.nodes[node]['latitude'])
                lon = float(g.nodes[node]['longitude'])
                x[i, j] = lat
                x[i, j+1] = lon
            elif "lat" in g.nodes[node]:
                lat = float(g.nodes[node]['lat'])
                lon = float(g.nodes[node]['long'])
                x[i, j] = lat
                x[i, j+1] = lon
        except ValueError:
            pass
    assert len(x[:, j:].min(axis=0)) == 2
    # print(x[x[:, j] != 0, j])
    x[x[:, j] == 0, j] = x[x[:, j] != 0, j].min()
    x[x[:, j+1] == 0, j+1] = x[x[:, j+1] != 0, j+1].min()
    print(x[:, j:].min(axis=0))
    x[:, j:] -= x[:, j:].min(axis=0)
    print(x[:, j:].max(axis=0))
    x[:, j:] /= x[:, j:].max(axis=0)
    x[:, j:] *= 5
    # x *= 1000
    print(x.max(axis=0))

    kmeans = KMeans(n_clusters=len(colors), random_state=0).fit(x)

    # cluster_size = 100
    # centers = kmeans.cluster_centers_
    # centers = centers.reshape(-1, 1, x.shape[-1]).repeat(cluster_size, 1).reshape(-1, x.shape[-1])
    # distance_matrix = cdist(x, centers)
    # labels = clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size

    labels = kmeans.labels_  # get the cluster labels of the nodes.
    for label, node in zip(labels, g.nodes):
        g.nodes[node]['cluster'] = label
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
    lat = 54.5
    lon = 11.4
    for cluster, kwcounts in counts.items():
        lon += .1
        counts = sorted([(v, k) for k, v in kwcounts.items()], reverse=True)[:20]
        name = ' \n'.join([f"[{k}/{totals[v]}:{v}]" for k, v in counts])
        icon = folium.Icon(color=colors[cluster])
        folium.Marker([lat, lon], popup=name, icon=icon).add_to(mv)


# make_node2vec()
graph = cluster_graph_as_table()
plot_cluster_kw_counts(graph)

# all_places()
#for path in Path("ISEBEL-Datasets").glob("*.csv"):
    #places_from_dataset(path)


mv.save("index.html")
print("Done")
