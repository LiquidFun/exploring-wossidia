from collections import Counter, defaultdict
from functools import cache

import numpy as np
import pandas as pd
# us_cities = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv")

import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

from show_map import cluster_graph_as_table, make_graph, filter_graph_by_coordinates



def get_app():
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash("Wossidia", external_stylesheets=external_stylesheets)

    size_marks = {i: f"{2**i}" for i in (-2, -1, 0, 1, 2)}
    app.layout = html.Div([
        html.Div(children=[
            html.H1('Wossidia'),
            html.Label('Dataset'),
            dcc.Dropdown(['witches', 'witches-dk', 'werewolf'], 'witches', id="dataset-name"),

            # html.Br(),
            html.Hr(style={"margin": "0.5em"}),
            html.Label('k-Means clusters'),
            dcc.Slider(2, 20, value=10, id='kmeans-slider', step=1),

            # html.Br(),
            html.Hr(style={"margin": "0.5em"}),
            html.Label('Size multiplier'),
            dcc.Slider(-2, 2, value=0, id='size-slider', step=0.1, marks=size_marks),

            html.Hr(style={"margin": "0.5em"}),
            # html.Br(),
            html.Label('Coordinate importance'),
            dcc.Slider(0, 10, value=4, id='coordinate-slider', step=1),

            html.Hr(style={"margin": "0.5em"}),
            html.Div([
                html.H6('Area filter'),
                dcc.Checklist(["Germany", "Netherlands", "Denmark"], ["Germany"], id="area-filters", inline=True),
            ], style={"flex": 1}),

            html.Hr(style={"margin": "0.5em"}),
            html.Div([
                html.H5('Additional toggles'),
                dcc.Checklist(["Use PCA", "Show major keywords"], [], id="toggles", inline=True),
            ], style={"flex": 1}),

            html.Hr(style={"margin": "0.5em"}),
            # html.H5('Cluster table'),
            html.Div(id='cluster-table', style={"flex": 1}),
            # dcc.Graph(id="cluster-table", style={"flex": 1, "height": "100%", "width": "100%"}),


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

def filter_graph_by_areas(graph, areas: tuple[str]):
    if areas is None:
        areas = area_filters.keys()
    from_lat = min(area_filters[area][0] for area in areas)
    to_lat = max(area_filters[area][1] for area in areas)
    from_lon = min(area_filters[area][2] for area in areas)
    to_lon = max(area_filters[area][3] for area in areas)
    # graph = graph.query(f"lat > {from_lat} and lat < {to_lat} and lon > {from_lon} and lon < {to_lon}")
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
        # replace new cluster names
        places.loc[places["cluster"] == cluster, "cluster_name"] = cluster_name
        # places["cluster_name"][places["cluster"] == cluster] = cluster_name


    # Add column with number of keywords
    places["num_keywords"] = places["keywords"].apply(lambda counter: sum(counter.values()))

    # Add cluster_info to places data frame
    places["cluster_info"] = ""
    cluster_counts = defaultdict(Counter)
    for i, row in places.iterrows():
        cluster_counts[row["cluster"]].update(row["keywords"])
    for i, row in places.iterrows():
        places.at[i, "cluster_info"] = ', '.join([f"{k}:{c}" for k, c in cluster_counts[row["cluster"]].most_common(5)])
        print(row["cluster_info"])
    print(cluster_counts)

    # Sort by number of keywords
    places.sort_values("cluster", inplace=True)

    return places


@cache
@app.callback(
    Output("map", "figure"),
    [
        Input("dataset-name", "value"),
        Input("kmeans-slider", "value"),
        Input("area-filters", "value"),
        Input("toggles", "value"),
        Input("size-slider", "value"),
        Input("coordinate-slider", "value"),
        Input("tabs", "value"),
    ],
    [State("map", "figure")],
)
def update_figure(dataset, kmeans_clusters, area_filters, toggles, size_multiplier, coord_mult, curr_tab, figure):
    print(curr_tab)
    if curr_tab != "map-tab":
        return figure
    use_pca = "Use PCA" in toggles if toggles is not None else False
    places = get_places_as_table(dataset, kmeans_clusters, tuple(area_filters), coord_mult, use_pca)
    print(len(places))



    # fig = go.Figure(go.Scattermapbox(
    #     lat=places["lat"],
    #     lon=places["long"],
    #     mode="markers+text",
    #     marker=go.scattermapbox.Marker(
    #         size=30,
    #         color="red",
    #         # colorscale=px.colors.qualitative.G10,
    #         # opacity=0.7,
    #     ),
    #     text=places["cluster_name"],
    #     textposition="bottom center",
    #     textfont={"family": "sans serif", "size": 10},
    #     # hoverinfo="text",
    # ))

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
        # text="name",
    )
    # fig.data[0].update(mode="markers+text+lines")
    fig.update_layout(legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.75)",
                                  bordercolor="Black", borderwidth=1, title_text="Cluster"))
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_traces(textposition="middle center")


    # Get centers of clusters
    # for cluster in set(places["cluster_name"]):
    #     cluster_places = places[places["cluster_name"] == cluster]
    #     cluster_center = cluster_places[["lat", "long"]].mean()
    #     print(cluster_center)

    #     keywords = Counter()
    #     for keywords_set in places[places["cluster_name"] == cluster]["keywords"]:
    #         keywords.update(keywords_set)
    #     most_common = keywords.most_common(1)[0]
    #     fig.add_trace(go.Scattergeo(
    #         lon=[cluster_center["long"]],
    #         lat=[cluster_center["lat"]],
    #         mode="markers+text",
    #         name=cluster,
    #         text=[f"{most_common[0]} ({most_common[1]})"] * 2,
    #         textfont={"size": 100, "color": "black"},
    #     ))

    return fig

@cache
@app.callback(
    Output("data-table", "children"),
    [
        Input("dataset-name", "value"),
        Input("kmeans-slider", "value"),
        Input("area-filters", "value"),
        Input("toggles", "value"),
        Input("coordinate-slider", "value"),
        Input("tabs", "value"),
    ],
    [State("data-table", "children")],
)
def update_data_table(dataset, kmeans_clusters, area_filters, toggles, coord_mult, curr_tab, children):
    if curr_tab != "data-tab":
        return children
    use_pca = "Use PCA" in toggles if toggles is not None else False
    places = get_places_as_table(dataset, kmeans_clusters, tuple(area_filters), coord_mult, use_pca)
    places = places.sort_values("num_keywords", ascending=False)
    places = places[["name", "lat", "long", "cluster", "num_keywords", "info"]]
    return dash_table.DataTable(
        places.to_dict("records"),
        [{"name": c, "id": c} for c in places.columns],
        style_cell={"textAlign": "left"},
        style_table={"overflowX": "scroll"},
        # striped=True, bordered=True, hover=True, id="data-table"
    )

@cache
@app.callback(
    Output("cluster-table", "children"),
    [
        Input("dataset-name", "value"),
        Input("kmeans-slider", "value"),
        Input("area-filters", "value"),
        Input("toggles", "value"),
        Input("coordinate-slider", "value"),
    ],
)
def update_cluster_table(dataset, kmeans_clusters, area_filters, toggles, coord_mult):
    use_pca = "Use PCA" in toggles if toggles is not None else False
    places = get_places_as_table(dataset, kmeans_clusters, tuple(area_filters), coord_mult, use_pca)
    # clusters = places.groupby("cluster_name").agg({"name": "count", "keywords": "sum"})
    # cluster_counts = places["cluster"].value_counts()
    # return str(set(reversed(places["cluster_name"])))
    # dash_table.DataTable(
    #     places.to_dict("records"),
    #     [{"name": c, "id": c} for c in ["cluster_name"]],
    #     style_cell={"textAlign": "left"},
    #     style_table={"overflowX": "scroll"},
    #     # striped=True, bordered=True, hover=True, id="data-table"
    # )
    counts = Counter()
    keywords = {}

    for _, row in places.iterrows():
        if row["cluster_name"] not in keywords:
            keywords[row["cluster_name"]] = Counter()
        keywords[row["cluster_name"]].update(row["keywords"])
        counts[row["cluster_name"]] += 1


    rows = [html.Tr([
        # html.Th("Cluster"),
        html.Th("Count"),
        html.Th("Keywords"),
    ])]
    for (k, occ), c, color in zip(keywords.items(), counts.values(), px.colors.qualitative.G10):
        fmt = ', '.join([f"{k} ({v})" for k, v in occ.most_common(5)])
        rows.append(html.Tr([
            # html.Td(k),
            html.Td(c),
            html.Td(fmt, style={"font-size": "12px"})
        ], style={"background-color": color}))

    return html.Table(rows)

    # return go.Figure(go.Table(
    #     header=dict(values=["Cluster", "Count", "Keywords"]),
    #     cells=dict(values=[list(keywords.keys()), list(counts.values()), list(keywords.values())]),
    # ))

def main():
    app.run_server(debug=True)  # Turn off reloader if inside Jupyter


if __name__ == '__main__':
    main()
