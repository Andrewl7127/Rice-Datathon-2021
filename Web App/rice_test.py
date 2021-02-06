import os
import base64
import re

from pandas import DataFrame
import numpy as np
import pandas as pd
import xgboost as xgb
import zipcodes
from datetime import datetime

pd.options.mode.chained_assignment = None  # default='warn'
from urllib.request import urlopen
from sklearn.cluster import KMeans
from dash.dependencies import Input, Output, State
import plotly.express as px
import json
import dash
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import requests
import io
from sklearn.preprocessing import MinMaxScaler

from dash_extensions import Download
from dash_extensions.snippets import send_data_frame

training = pd.read_csv('training.csv')


months = []
days = list(training['dayOfTheYear'])
for i in days:
    if(i in np.arange(1,31)):
        months.append(1)
    elif(i in np.arange(32,60)):
        months.append(2)
    elif(i in np.arange(60,91)):
        months.append(3)
    elif(i in np.arange(91,121)):
        months.append(4)
    elif(i in np.arange(121,152)):
        months.append(5)
    elif(i in np.arange(152,182)):
        months.append(6)
    elif(i in np.arange(182,213)):
        months.append(7)
    elif(i in np.arange(213,244)):
        months.append(8)
    elif(i in np.arange(144,274)):
        months.append(9)
    elif(i in np.arange(274,305)):
        months.append(10)
    elif(i in np.arange(305,335)):
        months.append(11)
    else:
        months.append(12)

training['Month'] = months

stores = {1000: 'Houston', 2000: 'Austin', 3000: 'College Station', 4000: 'San Antonio'}
convert = {"Overall": "Overall", "Bucket 1 (8AM to 10 AM)": 1, "Bucket 2 (11AM to 1PM)": 2,
           "Bucket 3 (2PM to 4 PM)": 3,
           "Bucket 4 (5PM to 8 PM)": 4}

months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'July', 8: 'Aug', 9: 'Sept', 10: 'Oct',
          11: 'Nov', 12: 'Dec'}

buckets = {1: '8AM to 10AM', 2: '11AM to 1PM', 3: '2PM to 4PM', 4: '5PM to 8 PM'}
coordinates = {'HOUSTON': (29.7604, -95.3698), 'AUSTIN': (30.2672, -97.7431), 'COLLEGE STATION': (30.6280, -96.3344),
               'SAN ANTONiO': (29.4241, -98.4936)}
training['latitude'] = [x[0] for x in training['City'].map(coordinates)]
training['longitude'] = [x[1] for x in training['City'].map(coordinates)]

image_filename = 'Rice_Datathon_Visualizations.png'  # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
# server = app.server

# app.layout = html.Div(children=[
#     html.Div([
#         html.H2(children='Tableau Visualization'),
#         html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
#     ])
# ])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
server = app.server

app.layout = html.Div(children=[
    html.H1('Rice Datathon', style={'font-weight': 'bold', 'font-size': '350%', 'padding-left': '2px'}),
    html.H4('Created by Andrew Liu, Adhvaith Vijay, Kyle Fang, and Siew Fen Eow',
            style={'font-weight': 'bold', 'padding-left': '3px', 'font-size': '180%'}),
    # All elements from the top of the page
    # html.Div([
    #     html.Br(),
    #     html.Br(),
    #     html.H2(children='XGBoost Model'),
    #     html.Label(["Please Select A Bucket",
    #                 dcc.Dropdown(id="numeric_bucket",
    #                              options=[{'label': k, 'value': k} for k in [0,1,2,3]],
    #                              value=2,
    #                              clearable=False,
    #                              multi=False)]),
    #     html.Br(),
    #     html.Label(["Which Store Are You Interested In?",
    #                 dcc.Dropdown(id="store",
    #                              options=[{'label': k, 'value': k} for k in [1000,2000,3000,4000]],
    #                              value=1000,
    #                              clearable=False,
    #                              multi=False)]),
    #     html.Br(),
    #     html.Label(["Enter In a Day (1 - 365): ",
    #                 dcc.Input(id='day', type='text', value='2')]),
    #     html.Br(),
    #     html.Br(),
    #     dbc.Button(id='button2', n_clicks=0, children="Submit", color="primary"),
    #     html.Br(),
    #     html.Br(),
    #     html.Div(id='prediction',
    #              children='')
    # ], style={'text-align': 'center'}),
    html.Div([
        html.Br(),
        html.H2(children='XGBoost Demo'),
        html.Label(["Enter In a Day (1 - 365): ",
                    dcc.Input(id='zipcode', type='text', value='55')]),
        html.Br(),
        html.Label(["Please Select A Bucket",
                    dcc.Dropdown(id="interact",
                                 options=[{'label': k, 'value': k} for k in ["Bucket 1 (8AM to 10AM)",
                                                                             "Bucket 2 (11AM to 1PM)",
                                                                             "Bucket 3 (2PM to 4 PM)",
                                                                             "Bucket 4 (5PM to 8 PM)"]],
                                 value="Bucket 1 (8AM to 10AM)",
                                 clearable=False,
                                 multi=False)]),
        html.Br(),
        html.Label(["Please Select A Store",
                    dcc.Dropdown(id="interact2",
                                 options=[{'label': k, 'value': k} for k in ["Houston","Austin",
                                                                             "College Station",
                                                                             "San Antonio"]],
                                 value="Houston",
                                 clearable=False,
                                 multi=False)]),
        html.Br(),
        dbc.Button(id='prob_button', n_clicks=0, children="Submit", color="primary"),
        html.Br(),
        html.Br(),
        html.Div(id='prob-output',
                 children='',
                 style={'font-weight': 'bold', 'font-size': '320%'}
                 )
    ], style={'text-align': 'center'}),
    html.Br(),
    html.Br(),
    html.Div([
        html.H2(children='Visualize Store Locations'),
        html.Label(["Pick Which Timeframe You Are Interested In:",
                    dcc.Dropdown(id="bucket",
                                 options=[{'label': k, 'value': k} for k in ["Overall", "Bucket 1 (8AM to 10 AM)",
                                                                             "Bucket 2 (11AM to 1PM)",
                                                                             "Bucket 3 (2PM to 4 PM)",
                                                                             "Bucket 4 (5PM to 8 PM)"]],
                                 value="Overall",
                                 clearable=False,
                                 multi=False)]),
        html.Br(),
        dbc.Button(id='button1', n_clicks=0, children="Submit", color="primary"),
        html.Br(),
        html.Br(),
        dcc.Graph(id='bucket_output', figure={})
    ], style={'text-align': 'center'}),
    html.Br(),
    html.Br(),
    html.Div([
        html.H2(children='Tableau Visualizations'),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
    ], style={'text-align': 'center'})
])


@app.callback(
    Output(component_id='prob-output', component_property='children'),
    Input(component_id='prob_button', component_property='n_clicks'),
    [State(component_id='zipcode', component_property='value'),
     State(component_id='interact', component_property='value'),
     State(component_id='interact2', component_property='value')],
    prevent_initial_call=False
)
def get_prob(n, zipcode, interact, interact2):
    zipcode = int(zipcode)
    map_stores = {"Houston":1000, "Austin":2000, "College Station":3000, "San Antonio":4000}
    map_buckets = {"Bucket 1 (8AM to 10AM)":1, "Bucket 2 (11AM to 1PM)":2, "Bucket 3 (2PM to 4 PM)":3, "Bucket 4 (5PM to 8 PM)":4}
    interact2 = map_stores[interact2]
    interact = map_buckets[interact]

    EBT_Site = training[training['StoreNumber'] == interact2].reset_index().at[0, 'EBT Site']
    Alcohol = training[training['StoreNumber'] == interact2].reset_index().at[0, 'Alcohol']
    Carwash = training[training['StoreNumber'] == interact2].reset_index().at[0, 'Carwash']
    Day2 = zipcode % 7

    # store, day, bucket
    mylist = [interact2, zipcode, interact, EBT_Site, Alcohol, Carwash, Day2]

    final_df = pd.DataFrame([mylist])
    loaded_model = xgb.Booster()
    loaded_model.load_model('bestest.model')
    predictions = np.array([round(i) for i in loaded_model.predict(xgb.DMatrix(final_df))])
    return "We Predict You Will Sell " + str(int(predictions[0])) + " Hotdogs."

#
# @app.callback(
#     Output(component_id='prediction', component_property='children'),
#     Input(component_id='button2', component_property='n_clicks'),
#     [State(component_id='numeric_bucket', component_property='value'),
#      State(component_id='store', component_property='value'),
#      State(component_id='day', component_property='value')],
#     prevent_initial_call=False
# )
# def get_prediction(n, numeric_bucket, store, day):
#     day = int(day)
#
#     EBT_Site = training[training['StoreNumber'] == store].at[0, 'EBT Site']
#     Alcohol = training[training['StoreNumber'] == store].at[0, 'Alcohol']
#     Carwash = training[training['StoreNumber'] == store].at[0, 'Carwash']
#     Day2 = int(day) % 7
#
#     # mylist = [store, day, numeric_bucket, EBT_Site, Alcohol, Carwash, Day2]
#
#     # final_df = pd.DataFrame([mylist])
#     # loaded_model = xgb.Booster()
#     # loaded_model.load_model('bestest.model')
#     #
#     # predictions = np.array([round(i) for i in loaded_model.predict(xgb.DMatrix(final_df))])
#     return str(store)


# ------------------------------------------------------------------------------------------------
@app.callback(
    Output(component_id='bucket_output', component_property='figure'),
    Input(component_id='button1', component_property='n_clicks'),
    [State(component_id='bucket', component_property='value')],
    prevent_initial_call=False
)
def viz_bucket(n, bucket):
    bucket_q = convert[bucket]
    if bucket_q == "Overall":
        result = training.groupby(['StoreNumber', 'Month', 'latitude', 'longitude']).agg(
            {'GrossSoldQuantity': sum}).reset_index()
    else:
        training_group = training[training['3HourBucket'] == bucket_q]
        result = training_group.groupby(['StoreNumber', 'Month', 'latitude', 'longitude']).agg(
            {'GrossSoldQuantity': sum}).reset_index()
    result['Month'] = result['Month'].map(months)
    result["StoreNumber"] = result["StoreNumber"].map(stores)

    fig = px.scatter_mapbox(result, lat="latitude", lon="longitude",
                            animation_frame="Month",
                            hover_data={'latitude': False, 'longitude': False},
                            hover_name="StoreNumber",
                            size="GrossSoldQuantity",
                            color_discrete_sequence=["red", "green", "blue", "goldenrod"],
                            color="StoreNumber",
                            opacity=0.80,
                            size_max=40,
                            mapbox_style="carto-positron",
                            zoom=5.7, center={"lat": 29.9055, "lon": -96.8766},
                            title='Hot Dogs Sold Across Stores: {}'.format(bucket))

    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 250
    fig.update_layout(autosize=True, height=550)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
