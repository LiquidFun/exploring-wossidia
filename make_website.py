from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import dash
from dash import html, dcc, Input, Output, State, dash_table
import plotly.express as px

from show_map import cluster_graph_as_table, make_graph, filter_graph_by_coordinates

curr_path = Path(__file__).parent
is_debug = (curr_path / "DEBUG").exists()

ALLOWED_DATASETS = ("witches", "witches-dk", "werewolf")
ALLOWED_KMEANS = list(range(2, 21))
ALLOWED_AREA_FILTERS = ("Germany", "Netherlands", "Denmark")
ALLOWED_TOGGLES = ("Use PCA",)
ALLOWED_SIZE_MULTIPLIERS = [i*0.1 for i in range(-20, 21)]
ALLOWED_COORD_MULTIPLIERS = list(range(0, 11))
ALLOWED_TABS = ("map-tab", "data-tab", "pca-tab")

def minmax(input_list):
    return min(input_list), max(input_list)

def get_app():
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash("Wossidia", external_stylesheets=external_stylesheets)

    size_marks = {i: f"{2**i}" for i in (-2, -1, 0, 1, 2)}
    app.layout = html.Div([
        html.Div(children=[
            html.H1(f'Wossidia {"(DEBUG)" if is_debug else ""}'),
            html.Label('Dataset'),
            dcc.Dropdown(ALLOWED_DATASETS, 'witches', id="dataset-name"),

            html.Hr(style={"margin": "0.5em"}),
            html.Label('k-Means clusters'),
            dcc.Slider(*minmax(ALLOWED_KMEANS), value=10, id='kmeans-slider', step=1),

            html.Hr(style={"margin": "0.5em"}),
            html.Label('Size multiplier'),
            dcc.Slider(*minmax(ALLOWED_SIZE_MULTIPLIERS), value=0, id='size-slider', step=0.1, marks=size_marks),

            html.Hr(style={"margin": "0.5em"}),
            html.Label('Coordinate importance'),
            dcc.Slider(*minmax(ALLOWED_COORD_MULTIPLIERS), value=4, id='coordinate-slider', step=1),

            html.Hr(style={"margin": "0.5em"}),
            html.H6('Area filter'),
            dcc.Checklist(ALLOWED_AREA_FILTERS, ["Germany"], id="area-filters", inline=True),

            html.Hr(style={"margin": "0.5em"}),
            html.H5('Additional toggles'),
            dcc.Checklist(ALLOWED_TOGGLES, [], id="toggles", inline=True),

            html.Hr(style={"margin": "0.5em"}),
            html.Div(id='cluster-table', style={"flex": 1}),

        ], style={'width': "20%", 'flex': 1, "padding": 10}),

        html.Div(dcc.Tabs(id="tabs", value="map-tab", # style={"height": "30px", "padding": "5px"},
           children=[dcc.Tab(children=[
               dcc.Graph(
                   figure=dict(
                       layout=dict(
                           mapbox=dict(
                               layers=[],
                               center={"lat": 53.66, "lon": 12.0},
                               zoom=8,
                               pitch=0,
                           ),
                           autosize=True,
                       ),
                   ), id="map", style={"height": "88vh"}),
           ], value="map-tab", label="Map"),

           dcc.Tab(children=[
               html.Div(id="data-table", style={"height": "88vh", "overflow": "auto"}),
           ], value="data-tab", label="Data"),

           # dcc.Tab(children=[
           #     html.Div(id="pca-plot", style={"height": "88vh", "overflow": "auto"}),
           # ], value="pca-tab", label="PCA Plot"),

        ]), style={'padding': 10, 'flex': 4})

    ], style={'display': 'flex', 'flex-direction': 'row'})

    return app

app = get_app()

area_filters = {
    "Germany": [53.027714, 54.500261, 10.863210, 14.627471],
    "Netherlands": [50.750000, 53.500000, 3.000000, 7.500000],
    "Denmark": [54.500000, 57.500000, 7.000000, 12.500000],
}

assert set(area_filters.keys()) == set(ALLOWED_AREA_FILTERS)

def filter_graph_by_areas(graph, areas: tuple[str]):
    if areas is None:
        areas = area_filters.keys()
    from_lat = min(area_filters[area][0] for area in areas)
    to_lat = max(area_filters[area][1] for area in areas)
    from_lon = min(area_filters[area][2] for area in areas)
    to_lon = max(area_filters[area][3] for area in areas)
    graph = filter_graph_by_coordinates(graph, from_lat, to_lat, from_lon, to_lon)
    return graph

@cache
def get_places_as_table(dataset: str, kmeans_clusters: int, filters: tuple[str], coord_multiplier: int, pca: bool):
    graph = make_graph(f"ISEBEL-Datasets/{dataset}-nodes.csv", f"ISEBEL-Datasets/{dataset}-edges.csv")
    graph = filter_graph_by_areas(graph, filters)
    cluster_graph_as_table(graph, kmeans_clusters, coord_multiplier, apply_pca=pca)

    places = pd.DataFrame([data for _, data in graph.nodes.data() if data["label"] == "place"])

    # Filter rows with no lat or long
    places = places[places["lat"].notna() & (places["lat"] != '')]
    places["lat"] = places["lat"].astype(float)
    places["long"] = places["long"].astype(float)

    # Add column with size of point
    # Set float values to set() in column "keywords"
    places["keywords"] = places["keywords"].apply(lambda x: Counter() if isinstance(x, float) else x)
    places["size"] = 10 + np.log(places["keywords"].apply(len) + 1) * 4

    # Convert clusters to string because then plotly counts the colors as discrete
    places["cluster"] = places["cluster"].astype(str)
    places["cluster_name"] = places["cluster"]

    cluster_sizes = places["cluster"].value_counts()
    for cluster in set(places["cluster"]):
        # Merge all keyword counters for cluster "cluster"
        keywords = Counter()
        for keywords_set in places[places["cluster"] == cluster]["keywords"]:
            keywords.update(keywords_set)
        cluster_name = f"({cluster_sizes[cluster]})_{keywords.most_common(1)[0][0]}_{cluster}"
        places.loc[places["cluster"] == cluster, "cluster_name"] = cluster_name


    # Add column with number of keywords
    places["num_keywords"] = places["keywords"].apply(lambda counter: sum(counter.values()))

    # Add cluster_info to places data frame
    places["cluster_info"] = ""
    cluster_counts = defaultdict(Counter)
    for i, row in places.iterrows():
        cluster_counts[row["cluster"]].update(row["keywords"])
    for i, row in places.iterrows():
        places.at[i, "cluster_info"] = ', '.join([f"{k}:{c}" for k, c in cluster_counts[row["cluster"]].most_common(5)])

    # Sort by number of keywords
    places.sort_values("cluster", inplace=True)

    return places


@cache
@app.callback(
    [
        Output("map", "figure"),
        Output("data-table", "children"),
        Output("cluster-table", "children"),
    ],
    [
        Input("dataset-name", "value"),
        Input("kmeans-slider", "value"),
        Input("area-filters", "value"),
        Input("toggles", "value"),
        Input("size-slider", "value"),
        Input("coordinate-slider", "value"),
        Input("tabs", "value"),
    ],
    [
        State("map", "figure"),
        State("data-table", "children"),
    ],
)
def on_parameter_change(dataset, kmeans_clusters, area_filters, toggles, size_multiplier, coord_mult, curr_tab,
                        map_state, table_state):

    # Validate user inputs
    assert dataset in ALLOWED_DATASETS, \
        f"Dataset {dataset} not in {ALLOWED_DATASETS}"
    assert kmeans_clusters in ALLOWED_KMEANS, \
        f"Kmeans {kmeans_clusters} not in {ALLOWED_KMEANS}"
    assert set(area_filters) - set(ALLOWED_AREA_FILTERS) == set(), \
        f"Areas {area_filters} not in {ALLOWED_AREA_FILTERS}"
    assert set(toggles) - set(ALLOWED_TOGGLES) == set(), \
        f"Toggles {toggles} not in {ALLOWED_TOGGLES}"
    assert size_multiplier in ALLOWED_SIZE_MULTIPLIERS, \
        f"Size multiplier {size_multiplier} not in {ALLOWED_SIZE_MULTIPLIERS}"
    assert coord_mult in ALLOWED_COORD_MULTIPLIERS, \
        f"Coordinate multiplier {coord_mult} not in {ALLOWED_COORD_MULTIPLIERS}"
    assert curr_tab in ALLOWED_TABS, f"Tab {curr_tab} not in {ALLOWED_TABS}"

    use_pca = "Use PCA" in toggles if toggles is not None else False
    places = get_places_as_table(dataset, kmeans_clusters, tuple(area_filters), coord_mult, use_pca)

    return [
        updated_map(places, size_multiplier, map_state) if curr_tab == "map-tab" else map_state,
        updated_data_table(places) if curr_tab == "data-tab" else table_state,
        updated_cluster_table(places),
    ]


def updated_map(places, size_multiplier, figure):

    fig = px.scatter_mapbox(
        places,
        lat="lat",
        lon="long",
        hover_name="name",
        hover_data=["info", "cluster", "cluster_info"],
        color="cluster_name",
        color_discrete_sequence=px.colors.qualitative.G10,
        center=figure["layout"]["mapbox"]["center"],
        size="size",
        zoom=figure["layout"]["mapbox"]["zoom"],
        mapbox_style="open-street-map",
        opacity=0.95,
        size_max=20 * 2**size_multiplier,
    )
    fig.update_layout(legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.75)",
                                  bordercolor="Black", borderwidth=1, title_text="Cluster"))
    # fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_traces(textposition="middle center")

    return fig

def updated_data_table(places):
    places = places.sort_values("num_keywords", ascending=False)
    places = places[["name", "lat", "long", "cluster", "num_keywords", "info"]]
    return dash_table.DataTable(
        places.to_dict("records"),
        [{"name": c, "id": c} for c in places.columns],
        style_cell={"textAlign": "left"},
        style_table={"overflowX": "scroll"},
    )

def updated_cluster_table(places):
    counts = Counter()
    keywords = {}

    for _, row in places.iterrows():
        if row["cluster_name"] not in keywords:
            keywords[row["cluster_name"]] = Counter()
        keywords[row["cluster_name"]].update(row["keywords"])
        counts[row["cluster_name"]] += 1

    rows = [html.Tr([
        html.Th("Count"),
        html.Th("Keywords"),
    ])]
    for (k, occ), c, color in zip(keywords.items(), counts.values(), px.colors.qualitative.G10):
        fmt = ', '.join([f"{k} ({v})" for k, v in occ.most_common(5)])
        rows.append(html.Tr([
            html.Td(c),
            html.Td(fmt, style={"font-size": "12px"})
        ], style={"background-color": color}))

    return html.Table(rows)

def main():
    app.run_server(debug=is_debug)


if __name__ == '__main__':
    main()
