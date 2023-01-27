from collections import defaultdict
import random

import folium

from show_map import apply_clusters_to_graph, make_graph

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

skip = "hexe witch witches hexen orte varia weerwolf weerwolven varulv werewolf " \
       "werwolf wolf wolven hond avond heks forhekse kjælling kjærn hekseri hare" \
       "when where wann wo hekserij toverij"

def plot_on_map(graph):
    for node in graph.nodes:
        if graph.nodes[node]['label'] == "place":
            # print(graph.nodes[node])
            try:
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


def plot_pca(graph, n_clusters: int = 10, coord_multiplier=1):
    cluster_graph_as_table(graph, n_clusters, coord_multiplier, apply_pca=True, plot_pca=True)


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


def main():
    graph = make_graph(f"ISEBEL-Datasets/{dataset}-nodes.csv", f"ISEBEL-Datasets/{dataset}-edges.csv")
    apply_clusters_to_graph(graph)
    plot_on_map(graph)
    plot_cluster_kw_counts(graph)
    print("Done")


dataset = "witches"

if __name__ == '__main__':
    main()
