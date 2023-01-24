from functools import cache

import numpy as np
import pandas as pd
# us_cities = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv")

import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px

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

            html.Br(),
            html.Label('k-Means clusters'),
            dcc.Slider(2, 20, value=10, id='kmeans-slider', step=1),

            html.Br(),
            html.Label('Size multiplier'),
            dcc.Slider(-2, 2, value=0, id='size-slider', step=0.1, marks=size_marks),

            html.Br(),
            html.Label('Coordinate importance'),
            dcc.Slider(0, 10, value=1, id='coordinate-slider', step=1),

            html.Br(),
            html.Label('Area filter'),
            dcc.Checklist(["Germany", "Netherlands", "Denmark"], ["Germany"], id="area-filters"),
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
                    ), id="map", style={"height": "86vh"}),
            ], value="map-tab", label="Map"),

            dcc.Tab(children=[
                html.Div(id="data-table", style={"height": "86vh", "overflow": "auto"}),
            ], value="data-tab", label="Data"),

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
def get_places_as_table(dataset: str, kmeans_clusters: int, filters: tuple[str], coord_multiplier: int):
    graph = make_graph(f"ISEBEL-Datasets/{dataset}-nodes.csv", f"ISEBEL-Datasets/{dataset}-edges.csv")
    graph = filter_graph_by_areas(graph, filters)
    cluster_graph_as_table(graph, kmeans_clusters, coord_multiplier)

    places = pd.DataFrame([data for _, data in graph.nodes.data() if data["label"] == "place"])

    # Filter rows with no lat or long
    places = places[places["lat"].notna() & (places["lat"] != '')]
    places["lat"] = places["lat"].astype(float)
    places["long"] = places["long"].astype(float)

    # Add column with size of point

    # Set float values to set() in column "keywords"
    places["keywords"] = places["keywords"].apply(lambda x: set() if isinstance(x, float) else x)
    places["size"] = 10 + np.log(places["keywords"].apply(len) + 1) * 4

    # Convert clusters to string because then plotly counts the colors as discrete
    places["cluster"] = places["cluster"].astype(str)
    return places

@cache
@app.callback(
    Output("map", "figure"),
    [
        Input("dataset-name", "value"),
        Input("kmeans-slider", "value"),
        Input("area-filters", "value"),
        Input("size-slider", "value"),
        Input("coordinate-slider", "value"),
        Input("tabs", "value"),
    ],
    [State("map", "figure")],
)
def update_figure(dataset, kmeans_clusters, area_filters, size_multiplier, coord_mult, curr_tab, figure):
    if curr_tab != "map-tab":
        return figure
    places = get_places_as_table(dataset, kmeans_clusters, tuple(area_filters), coord_mult)
    print(size_multiplier)

    fig = px.scatter_mapbox(
        places,
        lat="lat",
        lon="long",
        hover_name="name",
        hover_data=["info", "cluster"],
        color="cluster",
        color_discrete_sequence=px.colors.qualitative.G10,
        center=figure["layout"]["mapbox"]["center"],
        size="size",
        zoom=figure["layout"]["mapbox"]["zoom"],
        mapbox_style="carto-positron",
        opacity=0.95,
        size_max=20 * 2**size_multiplier,
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

@cache
@app.callback(
    Output("data-table", "children"),
    [
        Input("dataset-name", "value"),
        Input("kmeans-slider", "value"),
        Input("area-filters", "value"),
        Input("coordinate-slider", "value"),
        Input("tabs", "value"),
    ],
    [State("data-table", "children")],
)
def update_data_table(dataset, kmeans_clusters, area_filters, coord_mult, curr_tab, children):
    if curr_tab != "data-tab":
        return children
    places = get_places_as_table(dataset, kmeans_clusters, tuple(area_filters), coord_mult)
    places = places[["name", "lat", "long", "cluster"]]
    return dash_table.DataTable(
        places.to_dict("records"),
        [{"name": c, "id": c} for c in places.columns],
        style_table={"overflowX": "scroll", "align": "left"},
        # striped=True, bordered=True, hover=True, id="data-table"
    )


def main():
    app.run_server(debug=True)  # Turn off reloader if inside Jupyter


if __name__ == '__main__':
    main()
