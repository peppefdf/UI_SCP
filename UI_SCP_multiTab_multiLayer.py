## #!/home/cslgipuzkoa/virtual_machine_disk/anaconda3/envs/SCP_test/bin/python
import dash
from dash import Dash
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
from dash import html, callback_context, ALL
from dash import dcc, Output, Input, State, callback, dash_table

from dash_extensions import Download
from dash_extensions.snippets import send_file
from dash_extensions.snippets import send_data_frame

import dash_leaflet as dl
import dash_leaflet.express as dlx
import dash_daq as daq

from flask import Flask

import plotly.express as px
# plot test data
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import base64
import datetime
import io
from io import StringIO

#import re
import json
import pandas as pd
import geopy.distance
import numpy as np
import geopandas

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

from pylab import cm
import matplotlib

#from google.colab import drive
#drive.mount('/content/drive',  force_remount=True)

import sys
import os
from os import listdir
import shutil




root_dir = 'C:/Users/gfotidellaf/repositories/UI_SCP/assets/'
#root_dir = '/home/cslgipuzkoa/virtual_machine_disk/UI_SCP/assets/'
#sys.path.append('/content/drive/MyDrive/Colab Notebooks')
sys.path.append(root_dir + 'modules')
#"/content/drive/MyDrive/Colab Notebooks/calcroutes_module.py"
#import calcroutes_module -> import inside callback function
#import generate_GTFS_module -> import inside callback function

from dash.long_callback import DiskcacheLongCallbackManager
## Diskcache
import diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

print('Code restarted!')
"""
# the following code caused some problem...
# clean console ###############################################################
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass
###############################################################################
print('Console cleared!')
"""
#"/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/calcroutes_module.py"

#im1 = '/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/CSL_logo.PNG'
#im2 = '/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/DFG_logo.png'
#im1 = '/home/beppe23/mysite/assets/CSL_logo.PNG'
#im2 = '/home/beppe23/mysite/assets/DFG_logo.png'
im1 = root_dir +'images/CSL_logo.PNG'
#im2 = root_dir +'images/DFG_logo.png'
#im3 = root_dir +'images/MUBIL_logo.png'

#stops_file = "/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/GTFS_files_bus_stops_12_02_2024/all_stops_12_02_2024.csv"
#stops_file = "/home/beppe23/mysite/assets/all_stops_12_02_2024.csv"
#stops_file = "C:/Users/gfotidellaf/Desktop/CSL_Gipuzkoa/Accessibility/assets/all_stops_12_02_2024.csv"
stops_file = root_dir +'data/all_bus_stops.csv'


from PIL import Image
image1 = Image.open(im1)
#image2 = Image.open(im2)
#image3 = Image.open(im3)


stops_df = pd.read_csv(stops_file, encoding='latin-1')
stops_lat_lon = stops_df[['stop_lat','stop_lon']].to_numpy()

bus_icon = "https://i.ibb.co/HV0K5Fp/bus-stop.png" 
worker_icon = "https://i.ibb.co/W0H7nYM/meeting-point.png"
coworking_icon = "https://i.ibb.co/J2qXGKN/coworking-icon.png"

center = (43.26852347667122, -1.9741372404905988)
#    iconUrl= 'https://uxwing.com/wp-content/themes/uxwing/download/location-travel-map/bus-stop-icon.png',
#    iconUrl= "https://i.ibb.co/6n1tzcQ/bus-stop.png",
custom_icon_bus = dict(
    iconUrl= bus_icon,
    iconSize=[40,40],
    iconAnchor=[22, 40]
)
custom_icon_worker = dict(
    iconUrl= worker_icon,
    iconSize=[40,40],
    iconAnchor=[22, 40]
)
custom_icon_coworking = dict(
    iconUrl= coworking_icon,
    iconSize=[40,40],
    iconAnchor=[22, 40]
)

server = Flask(__name__)
#app = Dash(name = 'SCP_app', server = server, external_stylesheets=[dbc.themes.BOOTSTRAP],prevent_initial_callbacks=True,suppress_callback_exceptions = True)
app = Dash(name = 'SCP_app', server = server, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME, dbc.icons.BOOTSTRAP],prevent_initial_callbacks=True,suppress_callback_exceptions = True)

#app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],prevent_initial_callbacks=True)

"""
app.layout = html.Div([
    html.Img(src=image1,style={'width':'40%', "display": "inlineBlock", "verticalAlign": "top"}),
    html.Img(src=image2,style={'width':'25%',"display": "inlineBlock", "verticalAlign": "top"}),
    html.Div(dl.Map([dl.TileLayer()],
                     center=center, zoom=12, id="map",
                     style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"})),
    dbc.Button("Load stops", id="load_stops", n_clicks=0),
    html.Div(id='clickdata'),
    dcc.Store(id='internal-value', data=[])
])
"""
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 60,
    "left": 0,
    "bottom": 0,
    "width": "25rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "overflow": "scroll"           # scrollbar
}

"""
"padding": "2rem 1rem",
"background-color": "#f8f9fa",
"""
# the styles for the main content position it to the right of the sidebar and
# add some padding.

CONTENT_STYLE = {
    "margin-left": "5rem",
    "margin-right": "15rem",
}

CONTENT_STYLE_2 = {
    "margin-left": "0rem",
    "margin-right": "10rem",
}


INDICATORS_STYLE = {
    "background-color": "#f8f9fa",
    "position": "fixed",
    "top": 60,
    "right": 40,
    "bottom": 0,
    "width": "40rem",
    "overflow": "scroll"    
}

INDICATORS_STYLE_2 = {
    "background-color": "#f8f9fa",
    "position": "fixed",
    "top": 60,
    "left": 5,
    "bottom": 0,
    "width": "150rem",
    "overflow": "scroll"    
}


mouse_over_mess = """
Shifts stops to closest
<p>existing bus stops</p>"""

mouse_over_mess_stops = """
Proposes bus stops based on
<p>workers Lats/Lons in CSV file</p>
<p>and the number of clusters</p>"""

mouse_over_mess_clusters = """
Clusters by which to group workers"""

mouse_over_mess_depot = """
Uncheck box for removing bus stops!"""

routes = [{'label': 'Route ' +str(i+1), 'value': i} for i in range(3)]

stops_actions = [{'label': 'Delete marker', 'value': 'DM'},
                 {'label': 'Set origin of bus routes', 'value': 'SO'},
                 {'label': 'Set coworking hub', 'value': 'SC'}                   
                ]

interventions = [{'label': 'Company transportation', 'value': 'CT'},
                 {'label': 'Remote working', 'value': 'RW'}                   
                ]

choose_transp_hour = [{'label': "{:02d}".format(i) + ':00' + '-' + "{:02d}".format(i+1) + ':00', 'value': i} for i in range(24)] 


Step_1_text="Step 1:\nLoad and visualize data "
Step_2_text="Step 2:\nSelect type of intervention "
Step_3_text="Step 3:\nSelect time for commuting "
Step_4_text="Run"

collapse_button_1 = html.Div([

            dbc.Button(
                        [html.I(className="fas fa-plus-circle")],
                        id="Load_data_button_1",
                        className="mb-3",
                        color="primary",
                        n_clicks=0,
                        style={
                                "cursor": "pointer",
                                "display": "inline-block",
                                "white-space": "pre",
                        },
                ),
])

collapse_button_2 = html.Div([

        dbc.Button(
            [html.I(className="fas fa-plus-circle")],
            id="Intervention_type_button_1",
            className="mb-3",
            color="primary",
            n_clicks=0,
            style={
                    "cursor": "pointer",
                    "display": "inline-block",
                    "white-space": "pre",
                    },
        ),
])


collapse_button_3 = html.Div([
        dbc.Button(
            [html.I(className="fas fa-plus-circle")],
            id="Select_time_button_1",
            className="mb-3",
            color="primary",
            n_clicks=0,
            style={
                    "cursor": "pointer",
                    "display": "inline-block",
                    "white-space": "pre",
                    },
        ),
])


collapse_button_4 = html.Div([
        dbc.Button(
            [html.I(className="fas fa-plus-circle")],
            id="Advanced_settings_button_1",
            className="mb-3",
            color="primary",
            n_clicks=0,
            style={
                    "cursor": "pointer",
                    "display": "inline-block",
                    "white-space": "pre",
                    },
        ),
])

sidebar_1 =  html.Div(
       [
                 
        dbc.Row([
            dbc.Col([
                        html.P([Step_1_text],style={"font-weight": "bold","white-space": "pre"}),
                    ],
                    style={'height': '80px'},width=8),
            dbc.Col([        
                        collapse_button_1,
                    ],
                    style={'height': '80px'},width=3
                    )                    
        ]),
        dbc.Row([
                dbc.Collapse([
                        dcc.Upload(
                                id='upload-data_1',
                                children=html.Div([
                                html.A('Import Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            # Allow multiple files to be uploaded
                            multiple=True),
                            dbc.Button("Visualize clusters of workers", id="show_workers_1", n_clicks=0,style={"margin-top": "15px","font-weight": "bold"}),
                            html.Br(),        
                            html.P([ html.Br(),'Choose number of clusters'],id='cluster_num_1',style={"margin-top": "15px","font-weight": "bold"}),        
                            dbc.Popover(
                                dbc.PopoverBody(mouse_over_mess_clusters), 
                                target="n_clusters_1",
                                body=True,
                                trigger="hover",style = {'font-size': 12, 'line-height':'2px'},
                                placement= 'right',
                                is_open=False),
                            #dcc.Input(id="n_clusters", type="text", value='19'),
                            dcc.Slider(1, 30, 1,
                            value=19,
                            id='n_clusters_1',
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Br(),   
                        html.Div([
                                dbc.Button("Download data template", id="button_download_template_1", n_clicks=0,style={"margin-top": "15px","font-weight": "bold"}),               
                                Download(id="download_template_1"),
                                ])
                        ],
                        id="Load_data_panel_1",
                        is_open=False,
                    )

        ]),
        dbc.Row([
            dbc.Col([
                        html.P([Step_2_text],style={"font-weight": "bold","white-space": "pre"}),
                    ],
                    style={'height': '80px'},width=8),
            dbc.Col([        
                        collapse_button_2,
                    ],
                    style={'height': '80px'},width=3
                    )                    
        ]),
        dbc.Row([
            dbc.Collapse([
            dcc.Dropdown(interventions, multi=False,style={"margin-top": "15px"}, id='choose_intervention_1'),
            html.P([ html.Br(),'Select action for markers'],id='action_select_1',style={"margin-top": "15px", "font-weight": "bold"}),
            dcc.Dropdown(stops_actions, multi=False,style={"margin-top": "15px"}, id='choose_stop_action_1'),           
            html.Div(id='sidebar_intervention_1', style={"margin-top": "15px"})
            ],
           id="Intervention_type_panel_1",
           is_open=False,
           )
        ]),
        dcc.Store(id='worker_data_1', data=[]),
        dcc.Store(id='root_dir_1', data = root_dir), 
        dbc.Row([
            dbc.Col([
                        html.P([Step_3_text],style={"font-weight": "bold","white-space": "pre"}),
                    ],
                    style={'height': '80px'},width=8),
            dbc.Col([        
                        collapse_button_3,
                    ],
                    style={'height': '80px'},width=3
                    )                    
        ]),
        dbc.Row([
                dbc.Collapse([
                    html.P(['Choose time range'],style={"font-weight": "bold","white-space": "pre"}),
                    html.Div(dcc.Dropdown(choose_transp_hour, multi=False, id='choose_transp_hour_1'))            
                    ],
                    id='Select_time_panel_1',
                    is_open=False,
                ),
        ]),
        dbc.Row([
            dbc.Col([
                        html.P(['Advanced_settings'],style={"font-weight": "bold","white-space": "pre"}),
                    ],
                    style={'height': '80px'},width=8),
            dbc.Col([        
                        collapse_button_4,
                    ],
                    style={'height': '80px'},width=3
                    )                    
        ]),
        html.Br(),
        dbc.Row([
            dbc.Collapse([
                html.P([ html.Br(),'Liters of gasoline per kilometer (car)'],id='gas_km_car_1',style={"margin-top": "15px","font-weight": "bold"}),
                dcc.Slider(0, 5,0.02,
                    value=1./12,
                    id='choose_gas_km_car_1',
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),                   
                html.P([ html.Br(),'Liters of gasoline per kilometer (bus)'],id='gas_km_bus_1',style={"margin-top": "15px","font-weight": "bold"}),
                dcc.Slider(0, 10,0.05,
                    value=1.12,
                    id='choose_gas_km_bus_1',
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),                    
                html.P([ html.Br(),'CO2 Kg per lt'],id='CO2_lt_1',style={"margin-top": "15px","font-weight": "bold"}),
                dcc.Slider(0, 10,0.05,
                    value=2.3,
                    id='choose_CO2_lt_1',
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),            
                ],
                id="Advanced_settings_panel_1",
                is_open=False,
                ),
        ]),
        html.Div(dbc.Button(Step_4_text, 
                            id="run_MCM_1",
                            className="mb-3",
                            size="lg",
                            n_clicks=0,
                            disabled=False, 
                            color='warning'),
                            style={"white-space": "pre"}),
        dbc.Row(
            [                 
              dbc.Col(
                    html.Div(
                              dcc.Upload(id='load-scenario_1',
                                         children=html.Div([
                                         dbc.Button('Load scenario')
                                        ]),
                                        # Allow multiple files to be uploaded
                                        multiple=True
                                        )
                    ),
                    style={"margin-top": "15px"},
                    width='auto'
              ),

              dbc.Col(
                    html.Div([
                            dbc.Button("Download scenario", id='button_download_scenario_1', n_clicks=0),
                            Download(id="download_scenario_1"),
                            Download(id="download_inputs_1"),
                            Download(id="download_StopsCowHubs_1")
                            ]),
                            style={"margin-top": "15px"},
                            width="auto"
              ),  
            ]),
        html.Div(id='outdata_1', style={"margin-top": "15px"}),   
        dcc.Store(id='internal-value_route_opt_done_1', data=0),   
        dcc.Store(id='internal-value_stops_1', data=[]),
        dcc.Store(id='internal-value_coworking_1', data=[]),        
        dcc.Store(id='internal-value_routes_1', data=[]),        
        dcc.Store(id='internal-value_scenario_1', data=[]),        
        dcc.Store(id='internal-value_remote_days_1', data=0),        
        dcc.Store(id='internal-value_remote_workers_1', data=0),
        dcc.Store(id='internal-value_bus_number_1', data = 0) 
        ],
        id='sidebar_1',
        style=SIDEBAR_STYLE)


markers_all_1 = []
markers_remote_1 = []
markers_cow_1 = []
central_panel_1 = html.Div(
       [
          html.Div([
             html.Img(src=image1,style={'width':'40%', "display": "inlineBlock", "verticalAlign": "top"}),
             #html.Img(src=image2,style={'width':'25%',"display": "inlineBlock", "verticalAlign": "top"}),
             #html.Img(src=image3,style={'width':'25%',"display": "inlineBlock", "verticalAlign": "top"})

             ],style= {'verticalAlign': 'top'}),
          dls.Clock(
                    children=[ dl.Map(
                                [dl.ScaleControl(position="topright"),
                                 dl.LayersControl(
                                        [dl.BaseLayer(dl.TileLayer(), name='CO2', checked='CO2'),
                                         dl.BaseLayer(dl.TileLayer(), name='CO2/CO2_target', checked=False),
                                         dl.BaseLayer(dl.TileLayer(), name='weighted_d', checked=False),
                                         dl.BaseLayer(dl.TileLayer(), name='weighted_n', checked=False),
                                         dl.BaseLayer(dl.TileLayer(), name='Family type', checked=False)]  +
                                        [dl.Overlay(dl.LayerGroup(markers_all_1), name="all", id='markers_all_1', checked=False),
                                         dl.Overlay(dl.LayerGroup(markers_remote_1), name="remote", id='markers_remote_1', checked=False),
                                         dl.Overlay(dl.LayerGroup(markers_cow_1), name="coworking", id='markers_cow_1', checked=False)], 
                                        id="lc_1"
                                        )
                                ], 
                                center=center, 
                                zoom=12,
                                id="map_1",style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"})
                    ],
                    color="#435278",
                    speed_multiplier=1.5,
                    width=80,
                    show_initially=False
                    )
    ],
    style=CONTENT_STYLE)


#fig.update_layout(
#    margin=dict(l=100, r=100, t=5, b=5)
#)
#fig.update_layout(width=int(100))
# plot test data
#df = px.data.tips()
#fig = px.pie(df, values='tip', names='day')
#fig.update_layout(showlegend=False)
#fig.update_layout(title_text='Transport share', title_x=0.5)
d_tmp = {'counts': [100, 50, 2], 'distance': [100, 50, 2], 'Mode': ['Car','PT','Walk'], 'color': ['red','blue','green']}
df_tmp = pd.DataFrame(data=d_tmp)

radius_max = 1
x0 = 1
x1 = 30
x2 = 2
x3 = 2
x4 = 7                           
x0_max = 5
x1_max = 100
x2_max = 4
x3_max = 6
x4_max = 15
fig1 = go.Scatterpolar(
            r=[radius_max*x0/x0_max, radius_max*x1/x1_max, radius_max*x2/x2_max, radius_max*x3/x3_max, radius_max*x4/x4_max],
            theta=['Remote working days',
                   'Remote working persons (%)',
                   'Coworking hubs', 
                   'Bus routes',
                   'Bus stops'],
            hovertext= [str(x0),str(x1), str(x2), str(x3), str(x4)],
            fill='toself'
        )


cmap = cm.get_cmap('RdYlGn', 30)    # PiYG
interv = np.linspace(0,1,cmap.N)
j = 0
steps_gauge = []
for i in reversed(range(cmap.N-1)):
    rgba = cmap(i)
    t = {'range':[interv[j],interv[j+1]],'color': matplotlib.colors.rgb2hex(rgba)}
    j+=1
    steps_gauge.append(t)

fig2 = go.Indicator(mode = "gauge+number",
                        value = 0.3,
                       domain = {'x': [0, 1], 'y': [0, 1]},        
                        gauge= {
                                'steps':steps_gauge,
                                'axis':{'range':[0,1]},
                                'bar':{
                                        'color':'black',
                                        'thickness':0.5
                                      }
                                }

                    )


fig3 = go.Pie(labels=df_tmp["Mode"],
                  values=df_tmp["counts"],
                  showlegend=False,
                  textposition='inside',
                  textinfo='label+percent',
                  marker=dict(colors=df_tmp['color']))


fig4 = go.Bar(
            x=df_tmp['distance'],
            y=df_tmp['Mode'],
            orientation='h',
            marker_color=df_tmp['color'])

row_titles = ("Interventions", "CO2 emissions", "Transport share (%)", "Weekly distance share (km)")
column_titles = ()

fig = make_subplots(rows=4, cols=1, 
                    specs=[
                            [{"type": "scatterpolar"}], [{"type": "indicator"}],
                            [{"type": "pie"}], [{"type": "bar"}]],
                    row_heights=[2,1,2,1],
                    subplot_titles=row_titles
                    ) #-> row height is used to re-size plots of specific rows
#fig.for_each_annotation(lambda a:  a.update(y = 1.05) if a.text in column_titles else a.update(x = -0.07) if a.text in row_titles else())
fig.for_each_annotation(lambda a:  a.update(x = 0.2) if a.text in row_titles else())
fig.append_trace(fig1, 1, 1)
fig.append_trace(fig2, 2, 1)
fig.append_trace(fig3, 3, 1)
fig.append_trace(fig4, 4, 1)   

#fig.update_yaxes(title_text="CO2 emissions", row=1, col=1)
#fig.update_yaxes(title_text="Transport share", row=2, col=1)
#fig.update_yaxes(title_text="Distance share", row=3, col=1)
fig.update_annotations(font_size=18)
fig.update_layout(showlegend=False)    
fig.update_layout(polar=dict(radialaxis=dict(visible=False)))
fig.update_polars(radialaxis=dict(range=[0, radius_max]))

"""
          html.Div([
             daq.Gauge(
             color={"gradient":True,"ranges":{"green":[0,0.333],"yellow":[0.333,0.666],"red":[0.666,1.0]}},
             value=0.3,
             size = 140,
             label={'label':'CO2 emissions', 'style':{'font-size':'18px',"font-weight": "bold"}},
             max=1,
             min=0,
             id='CO2_gauge_1',
             style={'width': '40vh', 'height': '25vh'})
             ]),
          html.Div([
              dcc.Graph(
                figure=fig, 
                id='Transport_share',
                style={'width': '40vh', 'height': '30vh'})
            ]),
          html.Div([
              dcc.Graph(
                    #figure=px.bar(df, x='Km', y='Mode', orientation='h'), 
                    figure=px.bar(df, x='distance', y='Mode', orientation='h', labels={'distance':'Total distance (km)'}, color = 'distance', title="Weekly distance share (km)"),
                    id="Km_share",
                    style={'width': '60vh', 'height': '40vh'})       
              ])
"""
# 
#            style={'width': '70vh', 'height': '100vh'})
indicators_1 = html.Div(
        [              
        dcc.Graph(
            figure=fig,
            id = 'Indicator_panel_1',
            style={'width': '60vh', 'height': '100vh'}) 
        ],
        style=INDICATORS_STYLE
        )

central_panel2 = html.Div(
       [
          html.Div([
             html.Img(src=image1,style={'width':'40%', "display": "inlineBlock", "verticalAlign": "top"}),

             ],style= {'verticalAlign': 'top'}),
          dls.Clock(
                    children=[dl.Map([dl.TileLayer(),
                                      dl.ScaleControl(position="topright")], center=center, 
                                      zoom=12,
                                      id="map_2",style={'width': '90vh', 'height': '70vh', "margin": "auto", "display": "block"})
                    ],
                    color="#435278",
                    speed_multiplier=1.5,
                    width=80,
                    show_initially=False
                    )
    ],
    style=CONTENT_STYLE)


x = ['1970-01-01', '1970-01-01', '1970-02-01', '1970-04-01', '1970-01-02',
     '1972-01-31', '1970-02-13', '1971-04-19']

fig = make_subplots(rows=3, cols=3)


trace0 = go.Histogram(x=x, nbinsx=4)
trace1 = go.Histogram(x=x, nbinsx = 8)
trace2 = go.Histogram(x=x, nbinsx=10)
trace3 = go.Histogram(x=x,
                      xbins=dict(
                      start='1969-11-15',
                      end='1972-03-31',
                      size='M18'), # M18 stands for 18 months
                      autobinx=False
                     )
trace4 = go.Histogram(x=x,
                      xbins=dict(
                      start='1969-11-15',
                      end='1972-03-31',
                      size='M4'), # 4 months bin size
                      autobinx=False
                      )
trace5 = go.Histogram(x=x,
                      xbins=dict(
                      start='1969-11-15',
                      end='1972-03-31',
                      size= 'M2'), # 2 months
                      autobinx = False
                      )
 
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 2, 2)
fig.append_trace(trace4, 3, 1)
fig.append_trace(trace5, 3, 2)

indicators2 = html.Div(
        [     
          html.Div([
              dcc.Graph(
                figure=fig, 
                id='graph2', 
                style={'width':'150vh', 'height':'120vh'})
            ], 
            style={'width':'100%'}
            )
        ],
        style=INDICATORS_STYLE_2)

Tab_1 = dbc.Card(
    dbc.CardBody(
        [
        dbc.Row(
            [
                dbc.Col(sidebar_1, width=2, className='bg-light'),
                dbc.Col(central_panel_1, width=7),
                dbc.Col(indicators_1, width=3)
            ])
        ]
    ),
    className="mt-3",
)

Tab_2 = dbc.Card(
    dbc.CardBody(
        [
        dbc.Row(
            [
                dbc.Col(indicators2, width=12)
            ])
        ]
    ),
    className="mt-3",
)



tabs = dbc.Tabs(
    [
        dbc.Tab(Tab_1, label="Calculate scenarios"),
        dbc.Tab(Tab_2, label="Detailed result visualization")
    ]
)

app.layout = html.Div([tabs])


# Folder navigator ###############################################################
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    #root_dir = 'C:/Users/gfotidellaf/repositories/UI_SCP/assets/'
    #temp_file = root_dir + 'data/temp_workers_data.csv'
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    #df.to_csv(temp_file, index=False)
    gdf = geopandas.GeoDataFrame(df, 
                                 geometry = geopandas.points_from_xy(df.O_long, df.O_lat), 
                                 crs="EPSG:4326"
        )
      
    return gdf  


def parse_contents_load_scenario(contents, filename, date):
    content_type, content_string = contents.split(',')    
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    if 'scenario' in filename:
        gdf = geopandas.GeoDataFrame(df, 
                                 geometry = geopandas.points_from_xy(df.O_long, df.O_lat), 
                                 crs="EPSG:4326"
        )
        out = plot_result(gdf)
    else:
        out =  df

    return out
def drawclusters(workers_df,n_clusters):
    from sklearn.cluster import KMeans
    from scipy.spatial import ConvexHull

    workers_lat_lon = workers_df[['O_lat', 'O_long']].values.tolist()
    workers_lat_lon = np.array(workers_lat_lon)
    model = KMeans(n_clusters=n_clusters, max_iter=500).fit(workers_lat_lon)
    clusters_poly = []
    #points_per_cluster = []
    for i in range(n_clusters):
        points = workers_lat_lon[model.labels_ == i]
        hull = ConvexHull(points)
        vert = np.append(hull.vertices, hull.vertices[0])  # close the polygon by appending the first point at the end
        clusters_poly.append(points[vert])
        #points_per_cluster.append(len(points))
    return clusters_poly

def suggest_clusters(wdf):
    #sil_score_max = -100 #this is the minimum possible score
    dist_max = 100
    wdf = wdf[['O_lat', 'O_long']].values.tolist()
    alpha = 0.65
    n_max_clusters = int(19.*len(wdf)/2507)
    #beta = (1-alpha)*19 + alpha*0.63
    sil_score_max = 1

    for n_clusters in range(2,31):
        #model = KMeans(n_clusters = n_clusters, init='k-means++', max_iter=100, n_init=1)
        model = KMeans(n_clusters = n_clusters)
        labels = model.fit_predict(wdf)
        #sil_score = silhouette_score(wdf, labels, sample_size=len(wdf), random_state=42, metric= 'mahalanobis')
        sil_score = silhouette_score(wdf, labels, metric= 'manhattan')
        #db_score = davies_bouldin_score(wdf, labels)
        #ar_score = adjusted_rand_score(wdf, labels)
        #sil_score = silhouette_score(wdf, labels)
        #aver_score = (1 - alpha)*n_clusters/n_max_clusters + alpha*sil_score
        #x = (1-alpha)*n_clusters + alpha*sil_score    
        #aver_score = - (x - beta)**2 + 1
        d0 = (1-alpha)*(n_max_clusters - n_clusters)/n_max_clusters
        d1 = alpha*(sil_score_max - sil_score) 
        dist_to_max = (d0**2 + d1**2)**0.5
        print("The average silhouette and db score for %i clusters are %0.2f; the average score is %0.2f" %(n_clusters,sil_score, dist_to_max))
        #if sil_score > sil_score_max:
        if dist_to_max < dist_max:   
           dist_max = dist_to_max
           best_n_clusters = n_clusters
    return best_n_clusters    

def generate_colors(n):
    import random
    colors = []
    for i in range(n):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color = '#%02x%02x%02x'%(r, g, b)
        if color not in colors:
            colors.append(color)
    return colors 


def interpolate_color(color_start_rgb, color_end_rgb, t):
    """
    Interpolate between two RGB colors.
    
    Parameters:
    - color_start_rgb: Tuple of integers representing the starting RGB color.
    - color_end_rgb: Tuple of integers representing the ending RGB color.
    - t: Float representing the interpolation factor between 0 and 1.
    
    Returns:
    - A tuple representing the interpolated RGB color.
    """
    #print(color_start_rgb)
    #color_start_rgb = color_start_rgb.split('#')[1] 
    #color_end_rgb = color_end_rgb.split('#')[1]
    return tuple(int(start_val + (end_val - start_val) * t) for start_val, end_val in zip(color_start_rgb, color_end_rgb))

def hex_to_rgb(hex_color):
    """
    Convert hex to RGB.
    
    Parameters:
    - hex_color: String representing the hexadecimal color code.
    
    Returns:
    - A tuple of integers representing the RGB values.
    """
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

#def generate_color_gradient(CO2max,CO2_i, n_colors=256, n_min=0):
def generate_color_gradient(CO2max,CO2_i, label=0):
    #from  matplotlib.colors import ListedColormap, Normalize, LogNorm
    import matplotlib as mpl

    """
    cmap = mpl.colors.ListedColormap(['green', 'yellow', 'red'])
     #norm = mpl.colors.Normalize(vmin=0, vmax=100)
    norm = mpl.colors.LogNorm(vmin=0+1, vmax=CO2max+1) 
    # create a scalarmappable from the colormap
    #sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)  
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    #color_hex= mpl.colors.to_hex(sm.to_rgba(idx), keep_alpha=False)
    color_hex= mpl.colors.to_hex(sm.to_rgba(CO2_i))
    """
    """
    ranges = [
        {"start": "2ECC71", "end": "F7DC6F"},
        {"start": "F7DC6F", "end": "E74C3C"}
    ]
    """
    if label==0:
        ranges = [
            {"start": "2ECC71", "end": "F4D03F"},
            {"start": "F4D03F", "end": "C0392B"}
        ]
        color_start_hex = ranges[0]["start"]
        color_end_hex = ranges[0]["end"]
        color_start_rgb = hex_to_rgb(color_start_hex)
        color_end_rgb = hex_to_rgb(color_end_hex)
        # Generate gradient
        gradient1 = [interpolate_color(color_start_rgb, color_end_rgb, t) for t in np.linspace(0, 1, 256)]
        color_start_hex = ranges[1]["start"]
        color_end_hex = ranges[1]["end"]
        color_start_rgb = hex_to_rgb(color_start_hex)
        color_end_rgb = hex_to_rgb(color_end_hex)
        # Generate gradient
        gradient2 = [interpolate_color(color_start_rgb, color_end_rgb, t) for t in np.linspace(0, 1, 256)]
        gradient = gradient1 + gradient2

    elif label== 'Car':
        ranges = [
            {"start": "F1948A", "end": "B03A2E"}
        ]
        color_start_hex = ranges[0]["start"]
        color_end_hex = ranges[0]["end"]
        color_start_rgb = hex_to_rgb(color_start_hex)
        color_end_rgb = hex_to_rgb(color_end_hex)
        # Generate gradient
        gradient = [interpolate_color(color_start_rgb, color_end_rgb, t) for t in np.linspace(0, 1, 256)]
    
    elif label== 'PT':
        ranges = [
            {"start": "D6EAF8", "end": "21618C"}
        ]
        color_start_hex = ranges[0]["start"]
        color_end_hex = ranges[0]["end"]
        color_start_rgb = hex_to_rgb(color_start_hex)
        color_end_rgb = hex_to_rgb(color_end_hex)
        # Generate gradient
        gradient = [interpolate_color(color_start_rgb, color_end_rgb, t) for t in np.linspace(0, 1, 256)]    

    else:
        return '#2ECC71'
    

    N = len(gradient)

    value = int((CO2_i/CO2max)*N)
    idx = np.argmin(np.abs(np.array(range(N))-value))
    color = [gradient[idx][0]/255,gradient[idx][1]/255,gradient[idx][2]/255]
    color_hex = mpl.colors.to_hex(color)
    #if value == N:
    #print(value,idx,color)

    return color_hex

def plot_result(result, NremDays, NremWork, Nbuses, StopsCoords=[], CowFlags=[]):

    radius_max = 1
    x0 = NremDays
    x1 = NremWork
    x2 = sum(CowFlags)
    x3 = Nbuses
    x4 = len(StopsCoords) - sum(CowFlags)                           
    x0_max = 5
    x1_max = 100
    x2_max = 4
    x3_max = 10
    x4_max = 15
    fig1 = go.Scatterpolar(
                r=[radius_max*x0/x0_max, radius_max*x1/x1_max, radius_max*x2/x2_max, radius_max*x3/x3_max, radius_max*x4/x4_max],
                theta=['Remote working days',
                    'Remote working persons (%)',
                    'Coworking hubs', 
                    'Bus routes',
                    'Bus stops'],
                hovertext= [str(x0),str(x1), str(x2), str(x3), str(x4)],
                fill='toself'
            )

    Total_CO2 = result['CO2'].sum()
    Total_CO2_worst_case = result['CO2_worst_case'].sum()
    
    cmap = cm.get_cmap('RdYlGn', 30)    # PiYG
    interv = np.linspace(0,1,cmap.N)
    j = 0
    steps_gauge = []
    for i in reversed(range(cmap.N-1)):
        rgba = cmap(i)
        t = {'range':[interv[j],interv[j+1]],'color': matplotlib.colors.rgb2hex(rgba)}
        j+=1
        steps_gauge.append(t)

    fig2 = go.Indicator(mode = "gauge+number",
                        value = Total_CO2/Total_CO2_worst_case,
                        domain = {'x': [0, 1], 'y': [0, 1]},        
                        gauge  = {
                                    'steps':steps_gauge,
                                    'axis':{'range':[0,1]},
                                    'bar':{
                                            'color':'black',
                                            'thickness':0.5
                                        }
                                    }
                        )

    predicted = result['prediction']
    unique_labels, counts = np.unique(predicted, return_counts=True)
    d = {'Mode': unique_labels, 'counts':counts}
    df = pd.DataFrame(data=d)
    df['Mode'] = df['Mode'].map({0:'Walk',1:'PT',2:'Car'})
    df['color'] = df['Mode'].map({'Walk': 'green','PT': 'blue','Car':'red'})
    fig3 = go.Pie(labels=df["Mode"],
                    values=df["counts"],
                    showlegend=False,
                    textposition='inside',
                    textinfo='label+percent',
                    marker=dict(colors=df['color']))

    temp = result.copy()
    temp['distance_km'] = temp['distance']/1000.
    temp = temp[['Mode','distance_km']]
    Contribs = temp.groupby(['Mode']).sum() 
    Contribs = Contribs.reset_index()
    Contribs['color'] = Contribs['Mode'].map({'Walk': 'green','PT': 'blue','Car':'red'})
    fig4 = go.Bar(
                x=df_tmp['distance'],
                y=df_tmp['Mode'],
                orientation='h',
                marker_color=df_tmp['color'])

    row_titles = ("Interventions", "CO2 emissions", "Transport share (%)", "Weekly distance share (km)")
    column_titles = ()

    fig_total = make_subplots(rows=4, cols=1, 
                        specs=[
                                [{"type": "scatterpolar"}], [{"type": "indicator"}],
                                [{"type": "pie"}], [{"type": "bar"}]],
                        row_heights=[2,1,2,1],
                        subplot_titles=row_titles
                        ) #-> row height is used to re-size plots of specific rows
    #fig.for_each_annotation(lambda a:  a.update(y = 1.05) if a.text in column_titles else a.update(x = -0.07) if a.text in row_titles else())
    fig_total.for_each_annotation(lambda a:  a.update(x = 0.2) if a.text in row_titles else())
    fig_total.append_trace(fig1, 1, 1)
    fig_total.append_trace(fig2, 2, 1)
    fig_total.append_trace(fig3, 3, 1)
    fig_total.append_trace(fig4, 4, 1)   

    #fig.update_yaxes(title_text="CO2 emissions", row=1, col=1)
    #fig.update_yaxes(title_text="Transport share", row=2, col=1)
    #fig.update_yaxes(title_text="Distance share", row=3, col=1)
    fig_total.update_annotations(font_size=18)
    fig_total.update_layout(showlegend=False)    
    fig_total.update_layout(polar=dict(radialaxis=dict(visible=False)))
    fig_total.update_polars(radialaxis=dict(range=[0, radius_max]))


    temp = result.loc[result['Coworking'] == 1]
    Total_CO2 = temp['CO2'].sum()
    Total_CO2_worst_case = temp['CO2_worst_case'].sum()
    fig1 = go.Indicator(mode = "gauge+number",
                        value = Total_CO2/Total_CO2_worst_case,
                       domain = {'x': [0, 1], 'y': [0, 1]},        
                        gauge= {
                                'steps':steps_gauge,
                                'axis':{'range':[0,1]},
                                'bar':{
                                        'color':'black',
                                        'thickness':0.5
                                      }
                                })

    temp = result.loc[result['Rem_work'] == 1]
    Total_CO2 = temp['CO2'].sum()
    Total_CO2_worst_case = temp['CO2_worst_case'].sum()
    fig2 = go.Indicator(mode = "gauge+number",
                       value = Total_CO2/Total_CO2_worst_case,
                       domain = {'x': [0, 1], 'y': [0, 1]},      
                        gauge= {
                                'steps':steps_gauge,
                                'axis':{'range':[0,1]},
                                'bar':{
                                        'color':'black',
                                        'thickness':0.5
                                      }
                                })
        
    temp = result.loc[(result['Rem_work'] == 0) & (result['Coworking'] == 0)]
    Total_CO2 = temp['CO2'].sum()
    Total_CO2_worst_case = temp['CO2_worst_case'].sum()
    fig3 = go.Indicator(mode = "gauge+number",
                       value = Total_CO2/Total_CO2_worst_case,
                       domain = {'x': [0, 1], 'y': [0, 1]},        
                       gauge= {
                                'steps':steps_gauge,
                                'axis':{'range':[0,1]},
                                'bar':{
                                        'color':'black',
                                        'thickness':0.5
                                      }
                                })

    predicted = result.loc[result['Rem_work'] == 1, 'prediction']   
    unique_labels, counts = np.unique(predicted, return_counts=True)
    d = {'Mode': unique_labels, 'counts':counts}
    df = pd.DataFrame(data=d)
    df['Mode'] = df['Mode'].map({0:'Walk',1:'PT',2:'Car'}) 
    df['color'] = df['Mode'].map({'Walk': 'green','PT': 'blue','Car':'red'}) 
    fig4 = go.Pie(labels=df["Mode"],
                  values=df["counts"],
                  showlegend=False,
                  textposition='inside',
                  textinfo='label+percent',
                  marker=dict(colors=df['color']))

    predicted = result.loc[result['Coworking'] == 1, 'prediction']
    unique_labels, counts = np.unique(predicted, return_counts=True)
    d = {'Mode': unique_labels, 'counts':counts}
    df = pd.DataFrame(data=d)
    df['Mode'] = df['Mode'].map({0:'Walk',1:'PT',2:'Car'}) 
    df['color'] = df['Mode'].map({'Walk': 'green','PT': 'blue','Car':'red'})     
    fig5 = go.Pie(labels=df["Mode"],
                  values=df["counts"],
                  showlegend=False,
                  textposition='inside',
                  textinfo='label+percent',
                  marker=dict(colors=df['color']))
 
    predicted = result.loc[(result['Rem_work'] == 0) & (result['Coworking'] == 0), 'prediction']  
    unique_labels, counts = np.unique(predicted, return_counts=True)
    d = {'Mode': unique_labels, 'counts':counts}
    df = pd.DataFrame(data=d)
    df['Mode'] = df['Mode'].map({0:'Walk',1:'PT',2:'Car'}) 
    df['color'] = df['Mode'].map({'Walk': 'green','PT': 'blue','Car':'red'}) 
    fig6 = go.Pie(labels=df["Mode"],
                  values=df["counts"],
                  showlegend=False,
                  textposition='inside',
                  textinfo='label+percent',
                  marker=dict(colors=df['color']))

    temp = result.loc[result['Rem_work'] == 1]
    temp['distance_km'] = temp['distance']/1000.
    temp = temp[['Mode','distance_km']]
    Contribs = temp.groupby(['Mode']).sum() 
    Contribs = Contribs.reset_index()
    Contribs['color'] = Contribs['Mode'].map({'Walk': 'green','PT': 'blue','Car':'red'})
    print(Contribs.head())
    fig7 = go.Bar(
            x=Contribs['distance_km'],
            y=Contribs['Mode'],
            orientation='h',
            marker_color=Contribs['color'])

    temp = result.loc[result['Coworking'] == 1]
    temp['distance_km'] = temp['distance']/1000.
    temp = temp[['Mode','distance_km']]
    Contribs = temp.groupby(['Mode']).sum() 
    Contribs = Contribs.reset_index()
    Contribs['color'] = Contribs['Mode'].map({'Walk': 'green','PT': 'blue','Car':'red'})
    print(Contribs.head())
    fig8 = go.Bar(
            x=Contribs['distance_km'],
            y=Contribs['Mode'],
            orientation='h',
            marker_color=Contribs['color'])


    temp = result.loc[(result['Rem_work'] == 0) & (result['Coworking'] == 0)]
    temp['distance_km'] = temp['distance']/1000.
    temp = temp[['Mode','distance_km']]
    Contribs = temp.groupby(['Mode']).sum() 
    Contribs = Contribs.reset_index()
    Contribs['color'] = Contribs['Mode'].map({'Walk': 'green','PT': 'blue','Car':'red'})
    print(Contribs.head())
    fig9 = go.Bar(
            x=Contribs['distance_km'],
            y=Contribs['Mode'],
            orientation='h',
            marker_color=Contribs['color'])


    #family_types = ['Hogar de una persona', 'Otros hogares sin niños', '2 adultos',
    #                '2 adultos con niño(s)', '1 adulto con niño(s)',
    #                'Otros hogares con niños']    
    no_kids_df = result.loc[(result['Rem_work'] == 1) & (result['Tipo_familia'] <3) & (result['Mode'] == 'Car')]
    kids_df    = result.loc[(result['Rem_work'] == 1) &(result['Tipo_familia'] >2) & (result['Mode'] == 'Car')]
    fig10 = go.Bar(
            x=['No kids', 'Kids'],
            y=[len(no_kids_df['CO2'].index),len(kids_df['CO2'].index)],
            marker_color=['red','orange'])

    no_kids_df = result.loc[(result['Coworking'] == 1) & (result['Tipo_familia'] <3) & (result['Mode'] == 'Car')]
    kids_df    = result.loc[(result['Coworking'] == 1) &(result['Tipo_familia'] >2) & (result['Mode'] == 'Car')]
    fig11 = go.Bar(
            x=['No kids', 'Kids'],
            y=[len(no_kids_df['CO2'].index),len(kids_df['CO2'].index)],
            marker_color=['red','orange'])

    no_kids_df = result.loc[(result['Rem_work'] == 0) & (result['Coworking'] == 0) & (result['Tipo_familia'] <3) & (result['Mode'] == 'Car')]
    kids_df    = result.loc[(result['Rem_work'] == 0) & (result['Coworking'] == 0) & (result['Tipo_familia'] >2) & (result['Mode'] == 'Car')]
    fig12 = go.Bar(
            x=['No kids', 'Kids'],
            y=[len(no_kids_df['CO2'].index),len(kids_df['CO2'].index)],
            marker_color=['red','orange'])


    column_titles = ['Remote working', 'Coworking', 'all-(Remote+Coworking)']
    row_titles = ['CO2 emissions', 'Transport share', 'Distance share', 'Using car']
    #fig = make_subplots(rows=1, cols=3)
    fig_decomp = make_subplots(rows=4, cols=3, 
                        specs=[
                               [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                               [{"type": "pie"}, {"type": "pie"}, {"type": "pie"}],
                               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
                        column_titles = column_titles,
                        row_titles = row_titles,
                        row_heights=[1, 2, 2, 2]
                        ) #-> row height is used to re-size plots of specific rows
    
    #fig = make_subplots(rows=2, cols=3)
    fig_decomp.for_each_annotation(lambda a:  a.update(y = 1.05) if a.text in column_titles else a.update(x = -0.07) if a.text in row_titles else())
    fig_decomp.append_trace(fig1, 1, 1)
    fig_decomp.append_trace(fig2, 1, 2)
    fig_decomp.append_trace(fig3, 1, 3)
    fig_decomp.append_trace(fig4, 2, 1)
    fig_decomp.append_trace(fig5, 2, 2)
    fig_decomp.append_trace(fig6, 2, 3)
    fig_decomp.append_trace(fig7, 3, 1)
    fig_decomp.append_trace(fig8, 3, 2)
    fig_decomp.append_trace(fig9, 3, 3)
    fig_decomp.append_trace(fig10, 4, 1)
    fig_decomp.append_trace(fig11, 4, 2)
    fig_decomp.append_trace(fig12, 4, 3)      

    #fig.update_yaxes(title_text="CO2 emissions", row=1, col=1)
    #fig.update_yaxes(title_text="Transport share", row=2, col=1)
    #fig.update_yaxes(title_text="Distance share", row=3, col=1)
    fig_decomp.update_annotations(font_size=28)
    fig_decomp.update_layout(showlegend=False)    

    #children = [dl.TileLayer()]
    #maxCO2 = result['CO2'].max()
    #maxCO2_worst_case = result['CO2_worst_case'].max()
    Total_CO2 = result['CO2'].sum()
    Total_CO2_worst_case = result['CO2_worst_case'].sum()
    markers_all_1 = []
    markers_remote_1 = []
    markers_cow_1 = []
    markers_comm_1 = []
    for i_pred in result.itertuples():
        #print(i_pred.geometry.y, i_pred.geometry.x)
        #color = generate_color_gradient(maxCO2,i_pred.CO2) 
        #color = generate_color_gradient(i_pred.CO2_worst_case,i_pred.CO2) 
        
        maxCO2 = result.groupby("Mode")['CO2'].max()[i_pred.Mode]
        color = generate_color_gradient(maxCO2,i_pred.CO2, i_pred.Mode) 
        #color = generate_color_gradient(maxCO2_worst_case,i_pred.CO2, i_pred.Mode) 
        #print(color)
        #text = i_pred.Mode
        text = 'CO2: ' + '{0:.2f}'.format(i_pred.CO2) + ' Kg ' + '(' + i_pred.Mode + ')' + '<br>' +  'Weekly distance: ' + '{0:.2f}'.format(i_pred.distance/1000) + ' Km'
        #text = text + '<br>' + 'Remote working: ' + str(i_pred.Rem_work)
        n_cw = int(i_pred.Rem_work)
        text = text + '<br>' + 'Remote working: ' + (['Yes']*n_cw + ['No'])[n_cw-1]
        try:
            n_cw = int(i_pred.Coworking)
        except:
            n_cw = 0    
        text = text + '<br>' + 'Coworking: ' + (['Yes']*n_cw + ['No'])[n_cw-1]          
        marker_i = dl.CircleMarker(
                        id=str(i_pred),
                        children=[dl.Tooltip(content=text)],
                        center=[i_pred.geometry.y, i_pred.geometry.x],
                        radius=10,
                        fill=True,
                        fillColor=color,
                        fillOpacity=1.0,                            
                        stroke=True,
                        weight = 2.0,
                        color='black'
                        )
        #children.append(marker_i)
        markers_all_1.append(marker_i)  
        #print('remote_work_i: ', i_pred.Rem_work)
        try:
            if  i_pred.Rem_work > 0.0:
                markers_remote_1.append(marker_i)
        except:
            pass
        try:
            if  i_pred.Coworking > 0.0:
                markers_cow_1.append(marker_i)
        except:
            pass    
    markers_comm_1 = list(set(markers_all_1) - set(markers_remote_1) - set(markers_cow_1) )

    
    #children.append(dl.ScaleControl(position="topright"))
    children = [ dl.TileLayer(),
                dl.ScaleControl(position="topright"),
                dl.LayersControl(
                                [dl.BaseLayer(dl.TileLayer(), name='CO2', checked=False),
                                 dl.BaseLayer(dl.TileLayer(), name='CO2/CO2_target', checked=False),
                                 dl.BaseLayer(dl.TileLayer(), name='weighted_d', checked=False),
                                 dl.BaseLayer(dl.TileLayer(), name='weighted_n', checked=False),
                                 dl.BaseLayer(dl.TileLayer(), name='Family type', checked=False)] +
                                [dl.Overlay(dl.LayerGroup(markers_all_1), name="all", id= 'markers_all_1', checked=True),
                                 dl.Overlay(dl.LayerGroup(markers_remote_1), name="remote",id= 'markers_remote_1', checked=True),
                                 dl.Overlay(dl.LayerGroup(markers_cow_1), name="coworking",id= 'markers_cow_1', checked=True), 
                                 dl.Overlay(dl.LayerGroup(markers_comm_1), name="home-headquarters",id= 'markers_comm_1', checked=True)], 
                                id="lc_1"
                                )
                ]

    if CowFlags:
        custom_icon_coworking = dict(
        iconUrl= "https://i.ibb.co/J2qXGKN/coworking-icon.png",
        iconSize=[40,40],
        iconAnchor=[22, 40]
        )    
        Cow_markers = []
        for i, pos in enumerate(StopsCoords): 
            if CowFlags[i]==1:
                custom_icon = custom_icon_coworking
                tmp = dl.Marker(dl.Tooltip("Coworking hub"), position=pos, icon=custom_icon, id={'type': 'marker', 'index': i})    
                Cow_markers.append(tmp)  
        children = children + Cow_markers

    new_map = dl.Map(children, center=center,
                                     zoom=12,                        
                                     id="map_1",style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"})

    #return [Total_CO2/Total_CO2_worst_case, fig_total, fig_decomp, new_map]
    return [fig_total, fig_decomp, new_map]

def categorize_Mode(code):
    if 'Andando' in code:
        return 'Walk'
    elif 'Coche' in code:
        return 'Car'
    else:
        return 'PT'
    
def run_MCM(trips_ez, root_Dir, Transh, routeOptDone, gkm_car=1./12, gkm_bus=1.1, co2lt=2.3, NremDays=0, NremWork=0, CowCoords=None):
    import pandas as pd
    import sys    
    root_dir = root_Dir
    sys.path.append(root_dir + 'modules')
    import pp
    import prediction
    import pandas as pd
 
    if Transh == None:
        Transh = 8
        print('Transport hour not selected. Using default (08:00)')
    else:
        print('Chosen transport hour: ',Transh)
    workers_data_dir = 'data/'
    MCM_data_dir = 'data/input_data_MCM/'    
    model_dir = 'modules/models/'
    #trips_ez = pd.read_csv(root_dir + data_dir + 'workers_eskuzaitzeta_2k.csv')
    #trips_ez = pd.read_csv(root_dir + workers_data_dir + 'temp_workers_data.csv')    
    #if baseline == 1:    
    #    trips_ez['Modo'].apply(categorize_Mode)    
    #    trips_ez['Mode'] = trips_ez['Modo']
  
    eliminar = ['Unnamed: 0', 'Com_Ori', 'Com_Des', 'Modo', 'Municipio',
                'Motos','Actividad','Año','Recur', 'Income', 'Income_Percentile'] 
    try:
        trips_ez = trips_ez.drop(columns=eliminar)
    except:
        pass
    #trips_ez.head(10).to_csv(root_dir + workers_data_dir + 'template_workers_data.csv',index=False)
    trips_ez=pp.pp(Transh,trips_ez, routeOptDone, CowCoords, NremWork, NremDays, root_dir, MCM_data_dir) 
    #trips_ez['transit_tt'] = trips_ez['transit_tt'].apply(lambda x: x*0.2)
    #trips_ez['drive_tt'] = trips_ez['drive_tt'].apply(lambda x: x*1)
    prediction=prediction.predict(trips_ez, gkm_car, gkm_bus, co2lt, root_dir + model_dir)  
    return prediction


# Left sidebar subpanels #############################################
@callback(
    Output("Load_data_panel_1", "is_open"),
    [Input("Load_data_button_1", "n_clicks")],
    [State("Load_data_panel_1", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@callback(
    Output("Intervention_type_panel_1", "is_open"),
    [Input("Intervention_type_button_1", "n_clicks")],
    [State("Intervention_type_panel_1", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(
    Output("Select_time_panel_1", "is_open"),
    [Input("Select_time_button_1", "n_clicks")],
    [State("Select_time_panel_1", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(
    Output("Advanced_settings_panel_1", "is_open"),
    [Input("Advanced_settings_button_1", "n_clicks")],
    [State("Advanced_settings_panel_1", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open
#################################################################



# Output('internal-value_scenario','data',allow_duplicate=True),
@callback([
           Output('internal-value_remote_days_1', 'data',allow_duplicate=True),
           Output('internal-value_remote_workers_1', 'data',allow_duplicate=True)
          ],
          [
          Input('choose_remote_days_1','value'),
          Input('choose_remote_workers_1','value')
          ],
          prevent_initial_call=True)
def update_remote_work(rem_days, rem_work):
    return [rem_days, rem_work]


"""
@callback([Output('CO2_gauge_1', 'value',allow_duplicate=True),
           Output('Transport_share','figure',allow_duplicate=True),
           Output('Km_share','figure',allow_duplicate=True),
"""
# Output('internal-value_scenario','data',allow_duplicate=True),
@callback([Output('Indicator_panel_1', 'figure',allow_duplicate=True),
           Output('graph2','figure',allow_duplicate=True),
           Output('map_1','children',allow_duplicate=True),
           Output('internal-value_scenario_1','data',allow_duplicate=True)],
          [
          State('root_dir_1', 'data'),
          State('worker_data_1', 'data'),
          State('internal-value_remote_days_1','data'),
          State('internal-value_remote_workers_1','data'),
          State('internal-value_route_opt_done_1','data'),
          State('internal-value_stops_1','data'),
          State('internal-value_coworking_1','data'),
          State('internal-value_bus_number_1','data'),
          State('choose_transp_hour_1','value'),
          State('choose_gas_km_car_1','value'),
          State('choose_gas_km_bus_1','value'),
          State('choose_CO2_lt_1','value')
          ],
          Input('run_MCM_1', 'n_clicks'),
          prevent_initial_call=True)
def run_MCM_callback(root_dir, workerData, NremDays, NremWork, RouteOptDone, StopsCoords, CowoFlags, Nbuses, TransH, gkm_car, gkm_bus, co2lt, Nclicks):
    print('Cow. Flags:')
    print(CowoFlags)
    CowoIn = np.nonzero(CowoFlags)[0]
    print('Indices:')
    print(CowoIn)
    print('All coords:')
    print(StopsCoords)
    CowoCoords = np.array(StopsCoords)[CowoIn]
    print('Cow coords:')
    print(CowoCoords)
    
    df = pd.DataFrame.from_dict(workerData)    
    result = run_MCM(df, root_dir, TransH, RouteOptDone, gkm_car, gkm_bus, co2lt, NremDays, NremWork, CowoCoords)    
    print()
    print('Coworking:')
    print(result['Coworking'].unique())
    print()
    print('Remote work:')
    print(result['Rem_work'].unique())    
    out = plot_result(result, NremDays, NremWork, Nbuses, StopsCoords, CowoFlags)

    scenario = pd.DataFrame(result.drop(columns='geometry'))
    scenario_json = scenario.to_dict('records') # not working?
    #return [out[0],out[1],out[2],out[3], out[4], scenario_json]
    return [out[0],out[1],out[2], scenario_json]

"""
@callback([Output('CO2_gauge_1', 'value',allow_duplicate=True),
           Output('Transport_share','figure',allow_duplicate=True),
           Output('Km_share','figure',allow_duplicate=True),
           Output('map_1','children',allow_duplicate=True),
           Output('internal-value_scenario_1','data',allow_duplicate=True),
           Output('loading-component_MCM_1','children',allow_duplicate=True)],
          [
          State('root_dir_1', 'data'),
          State('worker_data_1', 'data'),
          State('choose_transp_hour_1','value')
          ],
          Input('run_MCM_baseline_1', 'n_clicks'),
          prevent_initial_call=True)
def run_MCM_baseline_callback(root_dir, workerData, TransH, Nclicks):
    df = pd.DataFrame.from_dict(workerData)    
    result = run_MCM(df, root_dir, TransH)    
    out = plot_result(result)
    scenario = pd.DataFrame(result.drop(columns='geometry'))
    scenario_json = scenario.to_dict('records')
    
    #default_remote_days = 0
    #default_remote_workers = 0
    #default__transp_hour = 8
    #default_gas_km_car = 1./12
    #default_gas_km_bus = 1.1
    #default__CO2_lt = 2.3
    #default_internal_value_stops = []
    #default_internal_value_coworking = []
    default_values = [0,0,8,1./12,1.1,2.3,[],[]]
    #*default_values
    return [out[0],out[1],out[2],out[3], scenario_json, True]
"""



@callback([
          Output('choose_remote_days_1', 'value',allow_duplicate=True),
          Output('choose_remote_workers_1', 'value',allow_duplicate=True),
          Output('worker_data_1', 'data',allow_duplicate=True),
          Output('internal-value_stops_1','data',allow_duplicate=True),
          Output('internal-value_coworking_1','data',allow_duplicate=True),
          Output('internal-value_scenario_1','data',allow_duplicate=True),
          Output('choose_transp_hour_1','value',allow_duplicate=True),
          Output('choose_gas_km_car_1','value',allow_duplicate=True),
          Output('choose_gas_km_bus_1','value',allow_duplicate=True),
          Output('choose_CO2_lt_1','value',allow_duplicate=True)
           ],
          [
          State('root_dir_1','data'),
          State('choose_remote_days_1', 'value'),
          State('choose_remote_workers_1', 'value'),
          State('worker_data_1', 'data'),
          State('internal-value_stops_1','data'),
          State('internal-value_coworking_1','data'),
          State('internal-value_scenario_1','data'),
          State('choose_transp_hour_1','value'),
          State('choose_gas_km_car_1','value'),
          State('choose_gas_km_bus_1','value'),
          State('choose_CO2_lt_1','value')],
          Input('reset_scenario_1', 'n_clicks'),
          prevent_initial_call=True)
def reset_scenario(root_Dir, NremDays, NremWork, WorkerFile, StopsCoords, CowoFlags, Scen, TransH, gkm_car, gkm_bus, co2lt, Nclicks):
          print('resetting variables...')
          NremDays = 0 
          NremWork = 0
          WorkerFile = [] 
          StopsCoords = [] 
          CowoFlags = []
          Scen = [] 
          TransH = 8
          gkm_car = 1./12
          gkm_bus = 1.1
          co2lt = 2.3
          try:
              shutil.rmtree(root_Dir+'data/input_data_MCM/GTFS_feeds/routes_EZ_companies/', ignore_errors=True)
          except:
              pass
          return [NremDays, NremWork, WorkerFile, StopsCoords, CowoFlags, Scen, TransH, gkm_car, gkm_bus, co2lt]


@callback([
           Output("outdata_1", "children",allow_duplicate=True)],
          [
          State('root_dir_1', 'data'),
          State('internal-value_remote_days_1','data'),
          State('internal-value_remote_workers_1','data'),
          State('internal-value_stops_1','data'),
          State('internal-value_coworking_1','data'),
          State('internal-value_scenario_1','data'),
          State('choose_transp_hour_1','value'),
          State('choose_gas_km_car_1','value'),
          State('choose_gas_km_bus_1','value'),
          State('choose_CO2_lt_1','value')],
          Input('save_scenario_1', 'n_clicks'),
          prevent_initial_call=True)
def save_scenario(root_dir, NremDays, NremWork, StopsCoords, CowoFlags, Scen, TransH, gkm_car, gkm_bus, co2lt, Nclicks):
    #root_dir = 'C:/Users/gfotidellaf/repositories/UI_SCP/assets/data/saved_scenarios/'
    
    root_dir = root_dir + 'data/saved_scenarios/'
    scen_label = 'scenario_'
    inputs_label = 'inputs_'
    stops_and_cow_label = 'stops_CowHubs_'

    scen_df = pd.DataFrame(Scen)

    inputs_dict = {'NremDays': NremDays, 'NremWork':NremWork, 
                   'TransH': TransH, 'gkm_car': gkm_car, 
                   'gkm_bus': gkm_bus, 'co2lt': co2lt
                   }
    columns = ['NremDays','NremWork', 
               'TransH', 'gkm_car', 
               'gkm_bus', 'co2lt']
    print('dictionary:')
    print(inputs_dict.items())
    inputs_df = pd.DataFrame(inputs_dict, index=[0])
    #inputs_df = pd.DataFrame.from_dict(inputs_dict, index=[0])


    try:
        files = os.listdir(root_dir)
        versions = [name.split('.csv')[0].split('_')[-1] for name in files]
        print('versions:')
        print(versions)
        last_version = max(list(map(int,versions))) + 1
    except:
        last_version = 0
    
    scen_file = root_dir + scen_label + str(last_version) + '.csv'
    inputs_file = root_dir + inputs_label + str(last_version) + '.csv'
    stops_and_cow_file = root_dir + stops_and_cow_label + str(last_version) + '.csv'

    scen_df.to_csv(scen_file)
    print('Save scenarion to:')
    print(scen_file)
    inputs_df.to_csv(inputs_file, index=False)
    if len(StopsCoords)>0:
        lats, lons = map(list, zip(*StopsCoords))
        stops_and_cow_df = pd.DataFrame(np.column_stack([lats, lons, CowoFlags]), 
                               columns=['lat', 'lon', 'CowHub'])
        stops_and_cow_df.to_csv(stops_and_cow_file)

    return ['Scenario saved into files!'] 


@callback([
           Output('map_1','children',allow_duplicate=True)],
           State('internal-value_scenario_1','data'),
           Input('lc_1', "baseLayer"),
           prevent_initial_call=True)
def switch_layer(Scen, layer):

    #custom_icon_remote = dict(
    #iconUrl= "https://imgur.com/ebj6ogp",
    #iconSize=[10,10],
    #iconAnchor=[0, 0]
    #)   

    markers_all_1 = []
    markers_remote_1 = []
    markers_cow_1 = []
    markers_comm_1 = []
    print('inside switch layer!')
    if Scen:
        scen_df = pd.DataFrame(Scen)
        scen_df = geopandas.GeoDataFrame(
                scen_df, geometry=geopandas.points_from_xy(scen_df.O_long, scen_df.O_lat), crs="EPSG:4326"
                )

        for i_pred in scen_df.itertuples():

            if layer == "CO2":
                #maxCO2 = scen_df['CO2'].max()
                maxCO2 = scen_df.groupby("Mode")['CO2'].max()[i_pred.Mode]

                #color = generate_color_gradient(maxCO2_worst_case,i_pred.CO2, i_pred.Mode) 
                color = generate_color_gradient(maxCO2,i_pred.CO2, i_pred.Mode)
                if i_pred.Mode == 'PT':
                    print('maxCO2: ',maxCO2)
                    print('CO2: ',i_pred.CO2)
                text = 'CO2: ' + '{0:.2f}'.format(i_pred.CO2) + ' Kg ' + '(' + i_pred.Mode + ')' + '<br>' +  'Weekly distance: ' + '{0:.2f}'.format(i_pred.distance/1000) + ' Km'


            elif layer == "CO2/CO2_target":
                #maxCO2 = scen_df.groupby("Mode")['CO2_over_aver'].max()[i_pred.Mode]
                #color = generate_color_gradient(maxCO2,i_pred.CO2_over_aver, i_pred.Mode)
                color = generate_color_gradient(1,i_pred.CO2_over_target, i_pred.Mode)  
                text = 'CO2_over_2030_target: ' + '{0:.2f}'.format(i_pred.CO2_over_target) + ' (' + i_pred.Mode + ')' + '<br>' +  'Weekly distance: ' + '{0:.2f}'.format(i_pred.distance/1000) + ' Km'

            elif layer == "weighted_d":
                #maxCO2 = scen_df.groupby("Mode")['weighted_d'].max()[i_pred.Mode]
                #color = generate_color_gradient(maxCO2,i_pred.weighted_d, i_pred.Mode)
                color = generate_color_gradient(1,i_pred.weighted_d, i_pred.Mode) 
                text = 'weighted_d: ' + '{0:.2f}'.format(i_pred.weighted_d) + ' (' + i_pred.Mode + ')' + '<br>' +  'Weekly distance: ' + '{0:.2f}'.format(i_pred.distance/1000) + ' Km'  

            elif layer == "weighted_n":            
                #maxCO2 = scen_df.groupby("Mode")['weighted_n'].max()[i_pred.Mode] 
                #color = generate_color_gradient(maxCO2,i_pred.weighted_d, i_pred.Mode)
                color = generate_color_gradient(1,i_pred.weighted_d, i_pred.Mode)
                text = 'weighted_n: ' + '{0:.2f}'.format(i_pred.weighted_n) + ' (' + i_pred.Mode + ')' + '<br>' +  'Weekly distance: ' + '{0:.2f}'.format(i_pred.distance/1000) + ' Km'  
            else:
                family_types = ['Hogar de una persona', 'Otros hogares sin niños', '2 adultos',
                                '2 adultos con niño(s)', '1 adulto con niño(s)',
                                'Otros hogares con niños']
                colors = generate_colors(len(family_types))
                color = colors[i_pred.Tipo_familia-1]
                if i_pred.Tipo_familia in range(3) and i_pred.Mode == 'Car':
                    color = '#C0392B'
                text = 'Family type: ' + family_types[i_pred.Tipo_familia-1] + ' (' + i_pred.Mode + ')' + '<br>' +  'Weekly distance: ' + '{0:.2f}'.format(i_pred.distance/1000) + ' Km'  

            n_rw = int(i_pred.Rem_work)
            text = text + '<br>' + 'Remote working: ' + (['Yes']*n_rw + ['No'])[n_rw-1]

            try:
                n_cw = int(i_pred.Coworking)
            except:
                n_cw = 0    
            text = text + '<br>' + 'Coworking: ' + (['Yes']*n_cw + ['No'])[n_cw-1]  
       
            marker_i = dl.CircleMarker(
                            id=str(i_pred.Index),
                            children=[dl.Tooltip(content=text)],
                            center=[i_pred.geometry.y, i_pred.geometry.x],
                            radius=10,
                            fill=True,
                            fillColor=color,
                            fillOpacity=1.0,
                            stroke=True,
                            weight = 2.0,
                            color='black' 
                            )
            #children.append(marker_i)
            markers_all_1.append(marker_i)  
            
            try:
                if  i_pred.Rem_work > 0.0:
                    #marker_i = dl.Marker(dl.Tooltip(""), position=(i_pred.geometry.y, i_pred.geometry.x), icon=custom_icon_remote, id={'type': 'marker', 'index': str(i_pred.Index)})    
                    #markers.append(tmp)  
                    markers_remote_1.append(marker_i)
            except:
                pass
    
            try:
                if  i_pred.Coworking > 0.0:


                    markers_cow_1.append(marker_i)
            except:
                pass

            markers_comm_1 = list(set(markers_all_1) - set(markers_remote_1) - set(markers_cow_1) )      

    Baselayer = [dl.BaseLayer(dl.TileLayer(), name='CO2', checked=False),
                 dl.BaseLayer(dl.TileLayer(), name='CO2/CO2_target', checked=False),
                 dl.BaseLayer(dl.TileLayer(), name='weighted_d', checked=False),
                 dl.BaseLayer(dl.TileLayer(), name='weighted_n', checked=False),
                 dl.BaseLayer(dl.TileLayer(), name='Family type', checked=False)]

    OL1 = dl.LayerGroup(markers_all_1)
    OL2 = dl.LayerGroup(markers_remote_1)
    OL3 = dl.LayerGroup(markers_cow_1)
    OL4 = dl.LayerGroup(markers_comm_1)
  
    children = [ dl.TileLayer(),
                    dl.ScaleControl(position="topright"),
                    dl.LayersControl(Baselayer +
                                    [dl.Overlay(OL1, name="all", checked=True),
                                     dl.Overlay(OL2, name="remote", checked=False),
                                     dl.Overlay(OL3, name="coworking", checked=False),
                                     dl.Overlay(OL4, name="home-headquarters", checked=False)], 
                                     id="lc_1"
                                    )
                    ]
    new_map = dl.Map(children, center=center,
                                        zoom=12,                        
                                        id="map_1",
                                        style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"})
        
    return [new_map]
    


# Download files callbacks ###########################################
# Download files callbacks ###########################################
@callback([Output("download_template_1", "data")],
          [State('root_dir_1','data')],
          Input('button_download_template_1', 'n_clicks'),
          prevent_initial_call=True)
def download_template(rootDir, Nclicks):
    tmpFile = rootDir + 'data/template_data.csv'
    template_df = pd.read_csv(tmpFile)  
    return [send_data_frame(template_df.to_csv, "template_input_data.csv", index=False)]



@callback([Output("download_scenario_1", "data")],
          [
          State('internal-value_scenario_1','data')],
          Input('button_download_scenario_1', 'n_clicks'),
          prevent_initial_call=True)
def download_scenario(Scen, Nclicks):
    scen_df = pd.DataFrame(Scen)
    print('data to download:')
    print(scen_df.head())
    return [send_data_frame(scen_df.to_csv, "scenario.csv", index=False)]

@callback([Output("download_inputs_1", "data")],
          [
          State('internal-value_remote_days_1', 'data'),
          State('internal-value_remote_workers_1', 'data'),
          State('choose_transp_hour_1','value'),
          State('choose_gas_km_car_1','value'),
          State('choose_gas_km_bus_1','value'),
          State('choose_CO2_lt_1','value')],
          Input('button_download_scenario_1', 'n_clicks'),
          prevent_initial_call=True)
def download_inputs(NremDays, NremWork, TransH, gkm_car, gkm_bus, co2lt, Nclicks):
    inputs_dict = {'NremDays': NremDays, 'NremWork':NremWork, 
                   'TransH': TransH, 'gkm_car': gkm_car, 
                   'gkm_bus': gkm_bus, 'co2lt': co2lt
                   }
    columns = ['NremDays','NremWork', 
               'TransH', 'gkm_car', 
               'gkm_bus', 'co2lt']
    inputs_df = pd.DataFrame(inputs_dict, index=[0])
    print('trying to save inputs...')
    return [send_data_frame(inputs_df.to_csv, "inputs.csv", index=False)]

@callback([Output("download_StopsCowHubs_1", "data")],
          [
          State('internal-value_stops_1','data'),
          State('internal-value_coworking_1','data')],
          Input('button_download_scenario_1', 'n_clicks'),
          prevent_initial_call=True)
def download_stops(StopsCoords, CowoFlags, Nclicks):
    if len(StopsCoords)>0:
        lats, lons = map(list, zip(*StopsCoords))
        stops_and_cow_df = pd.DataFrame(np.column_stack([lats, lons, CowoFlags]), 
                               columns=['lat', 'lon', 'CowHub'])
        return [send_data_frame(stops_and_cow_df.to_csv, "StopsCowHubs.csv", index=False)]




@callback([Output('worker_data_1', 'data',allow_duplicate=True),
           Output('n_clusters_1','value',allow_duplicate=True)],
            [Input('upload-data_1', 'contents'),
            State('upload-data_1', 'filename'),
            State('upload-data_1', 'last_modified')],
              prevent_initial_call=True)
def load_worker_data(list_of_contents, list_of_names, list_of_dates):
    #root_dir = 'C:/Users/gfotidellaf/repositories/UI_SCP/assets/'        
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        works_data = children[0]
        #temp_file = root_dir + 'data/' + 'temp_workers_data.csv'
        #df = pd.read_csv(temp_file) 
        #suggested_N_clusters = suggest_clusters(df)
        suggested_N_clusters = suggest_clusters(works_data)
        #print('workers dataframe?', works_data)
        workers = pd.DataFrame(works_data.drop(columns='geometry'))
        workers_dict = workers.to_dict('records') 
        
        return [workers_dict,suggested_N_clusters]
############################################################################################

#           Output('internal-value_stops','data',allow_duplicate=True),
@callback([Output('CO2_gauge_1', 'value',allow_duplicate=True),
           Output('Transport_share','figure',allow_duplicate=True),
           Output('Km_share','figure',allow_duplicate=True),
           Output('map_1','children',allow_duplicate=True),
           Output('internal-value_remote_days_1', 'data',allow_duplicate=True),           
           Output('internal-value_remote_workers_1', 'data',allow_duplicate=True),
           Output('choose_transp_hour_1','value',allow_duplicate=True),
           Output('choose_gas_km_car_1','value',allow_duplicate=True),
           Output('choose_gas_km_bus_1','value',allow_duplicate=True),
           Output('choose_CO2_lt_1','value',allow_duplicate=True),
           Output('internal-value_stops_1','data',allow_duplicate=True),           
           Output('internal-value_coworking_1','data',allow_duplicate=True)],
            [Input('load-scenario_1', 'contents'),
            State('load-scenario_1', 'filename'),
            State('load-scenario_1', 'last_modified')],
              prevent_initial_call=True)
def load_scenario(list_of_contents, list_of_names, list_of_dates):        
    if list_of_contents is not None:
        print()
        print('list of names:')
        print(list_of_names)
        inputs = [
            parse_contents_load_scenario(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates) if 'inputs' in n]
        stops_CowHubs = [
            parse_contents_load_scenario(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates) if 'stops_CowHubs' in n]
        #scenario = []
        scenario = [
            parse_contents_load_scenario(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates) if 'scenario' in n]
        
        try:
            print('inputs:')
            print(inputs)    
            inputs = np.array(inputs[0][:])[0]
            print(inputs)    
            print()
        except:
            pass

        try:
            #print(inputs[int(len(inputs)/2):])
            print('Stops:')
            print(stops_CowHubs)
            #print(stops_CowHubs[:,1],stops_CowHubs[:,2],stops_CowHubs[:,3])
        except:
            pass

        try:
            stops_CowHubs = np.array(stops_CowHubs[0][:])[:]
            print(stops_CowHubs)            
            lats = stops_CowHubs[:,1]
            lons = stops_CowHubs[:,2]
            #StopsCoords = map(list, zip(lats,lons))
            StopsCoords = list(zip(lats,lons))
            print('Stops:')
            print(StopsCoords)
            CowHubs_flags = stops_CowHubs[:,3]
        except:
            StopsCoords = []
            CowHubs_flags = []
        #StopsCoords,CowHubs_flags,
        #return [scenario[0][0],scenario[0][1],scenario[0][2],inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5], StopsCoords, CowHubs_flags, True]
        return [scenario[0][0],scenario[0][1],scenario[0][2],scenario[0][3],*inputs, StopsCoords, CowHubs_flags]


#@app.callback([Output("clickdata", "children")],
#Output("outdata", "children", allow_duplicate=True),
@app.callback([Output('internal-value_stops_1','data',allow_duplicate=True),
               Output('internal-value_coworking_1','data',allow_duplicate=True),
               Output('map_1','children',allow_duplicate=True)],
              State("n_clusters_1", "value"),
              State('worker_data_1', 'data'),
              State('root_dir_1','data'),
              Input("propose_stops_1", "n_clicks"),
              prevent_initial_call=True
              )
def propose_stops(n_clusters,workerData, root_dir, Nclick):
    if Nclick > 0:  
        #root_dir = 'C:/Users/gfotidellaf/repositories/UI_SCP/assets/'
        #root_dir = json.loads(root_dir)['root_dir']
        sys.path.append(root_dir + 'modules')      
        import find_stops_module   
        n_clusters  = int(n_clusters)
        cutoff = 0.8 # cutoff for maximum density: take maxima which are at least cutoff*max
        #root_dir = 'C:/Users/gfotidellaf/repositories/UI_SCP/assets/'
        #workers_DF = pd.read_csv(root_dir + "workers.csv", encoding='latin-1')
        #temp_file = root_dir + 'data/' + 'temp_workers_data.csv'
        #workers_DF = pd.read_csv(temp_file)    
        workers_DF = pd.DataFrame.from_dict(workerData)
        stops_DF = pd.read_csv(root_dir + 'data/'+ "all_bus_stops.csv", encoding='latin-1')
        bus_stops_df,model,yhat = find_stops_module.FindStops(workers_DF, stops_DF, n_clusters, cutoff)
        #df = pd.read_csv(filename)
        #out=St.loc[:'Lat']
        #for i in range(len(St)):
        #    out = out + str(St.loc[i,['Lat']]) + ', ' + str(St.loc[i,['Lon']]) + '; '
        out = ''
        St = []
        Cow = []
        for ind in bus_stops_df.index:
            out = out + str(bus_stops_df['Lat'][ind]) + ',' + str(bus_stops_df['Lon'][ind]) +';'
            St.append((bus_stops_df['Lat'][ind],bus_stops_df['Lon'][ind]))
            Cow.append(0)
        markers = [dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon_bus, id={'type': 'marker', 'index': i}) for i, pos in enumerate(St)]
        newMap = dl.Map([dl.TileLayer(),dl.ScaleControl(position="topright")] + markers,
                        center=center, zoom=12, id="map_1",
                        style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"})
        #return [out,St,newMap]
        #return [out,St,Cow,newMap]
        return [St,Cow,newMap]    

@app.callback([Output('map_1','children',allow_duplicate=True)],
                State("n_clusters_1", "value"),
                State('worker_data_1', 'data'),
               [Input("show_workers_1", "n_clicks")],
              prevent_initial_call=True
              )
def show_workers(n_clusters,workerData, N):
    #root_dir = 'C:/Users/gfotidellaf/repositories/UI_SCP/assets/'
    #temp_file = root_dir + 'data/temp_workers_data.csv'
    #workers_DF = pd.read_csv(temp_file)
    workers_DF = pd.DataFrame.from_dict(workerData)
    """
    St = []
    for ind in workers_DF.index:
         St.append((workers_DF['O_lat'][ind],workers_DF['O_long'][ind]))
    markers = [dl.Marker(dl.Tooltip("Do something?"), position=pos, icon=custom_icon_worker, id={'type': 'marker', 'index': i}) for i, pos in enumerate(St)]
    newMap = dl.Map([dl.TileLayer(),dl.ScaleControl(position="topright")] + markers,
                     center=center, zoom=12, id="map",
                     style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"})
    """ 
    clusters = drawclusters(workers_DF,n_clusters)
    n_max = max(len(x) for x in clusters ) # find maximum size of the clusters
    n_min = min(len(x) for x in clusters ) # find maximum size of the clusters
    #colors = generate_colors(n_clusters)
    n_colors = n_max
    #n_colors = 255
    colors = [generate_color_gradient(n_max, len(clusters[i])) for i in range(len(clusters))]
    #colors = [generate_color_gradient(n_max, len(clusters[i])) for i in range(len(clusters))]
    cluster_shapes = [dl.Polygon(children = dl.Tooltip('Number of workers: '+str(len(clusters[i]))), positions=clusters[i], fill=True, fillColor = colors[i], fillOpacity=0.9) for i in range(n_clusters)]
    newMap = dl.Map([dl.TileLayer(),dl.ScaleControl(position="topright")] + cluster_shapes,
                     center=center, zoom=12, id="map_1",
                     style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"})

    return [newMap]


@app.callback([Output('sidebar_intervention_1','children',allow_duplicate=True)],
              State('internal-value_stops_1','data'),
              State('internal-value_coworking_1','data'),
              Input('choose_intervention_1',"value"),
              prevent_initial_call=True
              )
def choose_intervention(St,Cow,interv):
    print('chosen interv.: ', interv)
           
    if interv == 'CT':
        sidebar_transport = html.Div(
            [           
            dbc.Button("Propose stops", id="propose_stops_1", n_clicks=0,style={"margin-top": "15px","font-weight": "bold"}),
            html.Br(),
            dbc.Popover(dcc.Markdown(mouse_over_mess_stops, dangerously_allow_html=True),
                      target="propose_stops_1",
                      body=True,
                      trigger="hover",style = {'font-size': 12, 'line-height':'2px'}),


            dbc.Button("Match stops", id="match_stops_1", n_clicks=0, style={"margin-top": "15px", "font-weight": "bold"}),
            dbc.Popover(dcc.Markdown(mouse_over_mess, dangerously_allow_html=True),
                      target="match_stops_1",
                      body=True,
                      trigger="hover",style = {'font-size': 12, 'line-height':'2px'}),
            html.P([ html.Br(),'Choose number of buses'],id='bus_num_1',style={"margin-top": "15px","font-weight": "bold"}),
            #dcc.Input(id="choose_buses", type="text", value='3'),
            dcc.Slider(1, 10, 1,
                   value=2,
                   id='choose_buses_1'
            ),
            dbc.Button("Calculate routes", id="calc_routes_1",n_clicks=0, style={"margin-top": "15px"}),
            html.P([ html.Br(),'Select route to visualize'],id='route_select_1',style={"margin-top": "15px", "font-weight": "bold"}),
            dcc.Dropdown(routes, multi=False,style={"margin-top": "15px"},id='choose_route_1'),
            dbc.Button("Visualize routes", id="visualize_routes_1", n_clicks=0,style={"margin-top": "15px"}),
            html.Br(),               
            html.Div(id='outdata_1', style={"margin-top": "15px"}),
            dcc.Store(id='internal-value_stops_1', data=St),
            dcc.Store(id='internal-value_coworking_1', data=Cow),        
            dcc.Store(id='internal-value_routes_1', data=[]),        
            dcc.Store(id='internal-value_scenario_1', data=[])
            ])       
        
        return [sidebar_transport]

    if interv == 'RW':         
        
        sidebar_remote_work = html.Div(
                [
                html.P([ html.Br(),'Choose number of days of remote working'],id='remote_days_num_1',style={"margin-top": "15px","font-weight": "bold"}),
                #dcc.Input(id="choose_buses", type="text", value='3'),
                dcc.Slider(0, 5, 1,
                       value=0,
                       id='choose_remote_days_1'
                ),
                html.P([ html.Br(),'Choose "%" of remote workers'],id='remote_workers_num_1',style={"margin-top": "15px","font-weight": "bold"}),
                #dcc.Input(id="choose_buses", type="text", value='3'),
                dcc.Slider(0, 100, 5,
                       value=0,
                       id='choose_remote_workers_1',
                       marks=None,
                       tooltip={"placement": "bottom", "always_visible": True}                       
                ),
                html.Br(),               
                html.Div(id='outdata_1', style={"margin-top": "15px"}),
                dcc.Store(id='internal-value_stops_1', data=St),
                dcc.Store(id='internal-value_coworking_1', data=Cow),        
                dcc.Store(id='internal-value_routes_1', data=[]),        
                dcc.Store(id='internal-value_scenario_1', data=[])
                ])        

        
        return [sidebar_remote_work]



@app.long_callback([Output("outdata_1", "children",allow_duplicate=True),
               Output("internal-value_route_opt_done_1", 'data',allow_duplicate=True),
               Output('internal-value_routes_1','data',allow_duplicate=True),
               Output('internal-value_bus_number_1','data',allow_duplicate=True),
               Output("choose_route_1", "options",allow_duplicate=True),
               Output('map_1','children',allow_duplicate=True)],
               State('choose_buses_1',"value"),
               State('internal-value_stops_1','data'),
               State('internal-value_coworking_1','data'),
               State('choose_CO2_lt_1','value'),
               State('root_dir_1','data'),
               Input("calc_routes_1", "n_clicks"),
               manager=long_callback_manager,
              prevent_initial_call=True
              )
def calc_routes(Nroutes,St,Cow,CO2km, root_Dir, Nclick):
    print()
    print('inside calc_routes!')
    if Nclick > 0:
      #root_dir = 'C:/Users/gfotidellaf/repositories/UI_SCP/assets/'
      root_dir = root_Dir
      sys.path.append(root_dir + 'modules')  
      import calcroutes_module
      import dash_leaflet as dl
      import generate_GTFS_module as gGTFS
      custom_icon_bus = dict(
      iconUrl= "https://i.ibb.co/HV0K5Fp/bus-stop.png",
      iconSize=[40,40],
      iconAnchor=[22, 40]
      )

      custom_icon_coworking = dict(
      iconUrl= "https://i.ibb.co/J2qXGKN/coworking-icon.png",
      iconSize=[40,40],
      iconAnchor=[22, 40]
      )    
    
      center = (43.26852347667122, -1.9741372404905988)
    
      #list_routes = range(1,int(Nroutes)+1)    
      list_routes = range(int(Nroutes))
      new_menu = [{'label': 'Route ' +str(i+1), 'value': i} for i in list_routes]
      Stops = []
      markers = []
      print('Discriminating bus stops from coworking hubs...')
      for i, pos in enumerate(St): 
        if Cow[i]==1:
             custom_icon = custom_icon_coworking
        else:
             custom_icon = custom_icon_bus
             Stops.append(pos)
        tmp = dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon, id={'type': 'marker', 'index': i})    
        markers.append(tmp)  

      print('List of Stops generated')        
      print('\n')
      print('\n')
      print('Start calculating routes...')
      routes, routes_points_coords, Graph = calcroutes_module.CalcRoutes_module(Stops,int(Nroutes),float(CO2km))
      print('Routes calculated!')
      #print(routes_points_coords)
      gGTFS.gGTFS(routes, Stops, Graph, root_dir)
      route_opt = 1
      # We don't really need to update the map here. We do it just to make the Spinner work: ############ 
      #markers = [dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon_bus, id={'type': 'marker', 'index': i}) for i, pos in enumerate(Stops)]
      newMap = dl.Map([dl.TileLayer(),dl.ScaleControl(position="topright")] + markers,
                     center=center, zoom=12, id="map_1",
                     style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"}) 
      ###################################################################################################   
      #return ["Calculation completed!", routes_coords, new_menu, newMap]
      return ["Calculation completed for: "+str(len(Stops)), route_opt, routes_points_coords, Nroutes, new_menu, newMap]


@app.callback([Output('map_1','children',allow_duplicate=True)],
              [State('choose_route_1',"value"),
              State('internal-value_stops_1','data'),
              State('internal-value_coworking_1','data'),
              State('internal-value_routes_1','data')],
              [Input("visualize_routes_1", "n_clicks")],
              prevent_initial_call=True
              )
def visualize_route(Route,St,Cow,RoutesCoords,Nclick):
    if Nclick > 0:
      print('Start route visualization...')
      #Route = int(Route.split(' ')[1])-1
      Route = int(Route)-1    
      RoutesCoords = RoutesCoords[Route]
      markers = []
      for i, pos in enumerate(St): 
        if Cow[i]==1:
             custom_icon = custom_icon_coworking
        else:
             custom_icon = custom_icon_bus
        tmp = dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon, id={'type': 'marker', 'index': i})    
        markers.append(tmp)     
      #markers = [dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon_bus, id={'type': 'marker', 'index': i}) for i, pos in enumerate(Stops)]
      newMap = dl.Map([dl.TileLayer(), dl.ScaleControl(position="topright"), dl.Polyline(positions=RoutesCoords, pathOptions={'weight':10})] + markers,
                     center=center, zoom=12, id="map_1",
                     style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"})
      return [newMap]


@app.callback([Output("outdata_1", "children",allow_duplicate=True), 
               Output('internal-value_stops_1','data',allow_duplicate=True),
               Output('map_1','children',allow_duplicate=True)],
              [State('internal-value_stops_1','data'),
               State('internal-value_coworking_1','data')],
              Input("match_stops_1", "n_clicks"),
              prevent_initial_call=True
              )
def match_stops(St,Cow,Nclick):
    print('inside callback')
    if Nclick > 0:
      print('inside if...')
      print('matching stops...')  
      bus_stops = []
      out = ''
      for i_st in range(len(St)):
        if Cow[i_st] == 0:  
          #ref = np.array([lat,lon])
          ref = np.array([St[i_st][0],St[i_st][1]])
          ref = np.tile(ref,(len(stops_lat_lon),1)) # generate replicas of ref point
          #d = [sum((p-q)**2)**0.5 for p, q in zip(ref, stops_lat_lon)] # calculate distance of each bus stop to ref point
          d = [geopy.distance.geodesic((p[0],p[1]), (q[0],q[1])).km for p, q in zip(ref, stops_lat_lon)] # calculate distance of each bus stop to ref point

          ind_min = d.index(min(d)) # find index of closest bus stop  
          x = stops_lat_lon[ind_min][0]
          y = stops_lat_lon[ind_min][1]
          bus_stops.append((x, y))
          St[i_st]=(x,y)
          out = out + str(St[i_st][0]) + ', ' + str(St[i_st][1]) + '; '
      markers = []
      for i, pos in enumerate(St): 
           if Cow[i] == 1:
               custom_icon = custom_icon_coworking
               #print('setting coworking icon...')
           else:
               custom_icon = custom_icon_bus
           tmp = dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon, id={'type': 'marker', 'index': i})    
           markers.append(tmp)
      #markers = [dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon_bus, id={'type': 'marker', 'index': i}) for i, pos in enumerate(St)]
      newMap = dl.Map([dl.TileLayer(),dl.ScaleControl(position="topright")] + markers,
                     center=center, zoom=12, id="map_1",
                     style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"})
      return [len(St),St,newMap]

#              [Input('map_1','clickData')],
@app.callback([Output("outdata_1", "children",allow_duplicate=True), 
               Output('internal-value_stops_1','data',allow_duplicate=True),
               Output('internal-value_coworking_1','data',allow_duplicate=True),
               Output('map_1','children',allow_duplicate=True)],
              [State('internal-value_stops_1','data'),
               State('internal-value_coworking_1','data')],
              [Input('map_1','dblclickData')],
              prevent_initial_call=True
              )
def add_marker(St,Cow,clickd):
       print('adding marker...')
       print(clickd)
       marker_lat = clickd['latlng']['lat']
       marker_lon = clickd['latlng']['lng']
       St.append((marker_lat,marker_lon))
       Cow.append(0)
       out=''
       for i in range(len(St)):
           out = out + str(St[i][0]) + ', ' + str(St[i][1]) + '; '
       markers = []
       for i, pos in enumerate(St): 
           if Cow[i] == 1:
               custom_icon = custom_icon_coworking
               #print('setting coworking icon...')
           else:
               custom_icon = custom_icon_bus
           tmp = dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon, id={'type': 'marker', 'index': i})    
           markers.append(tmp)    

       newMap = dl.Map([dl.TileLayer(),dl.ScaleControl(position="topright")] + markers,
                     center=center, zoom=12, id="map_1",
                     style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"})
       return [out,St,Cow,newMap]


@app.callback([Output("outdata_1", "children",allow_duplicate=True),
               Output('internal-value_stops_1','data',allow_duplicate=True),
               Output('internal-value_coworking_1','data',allow_duplicate=True),
               Output("choose_stop_action_1", "value",allow_duplicate=True),
               Output('map_1','children',allow_duplicate=True)],
              [State('internal-value_stops_1','data'), 
               State('internal-value_coworking_1','data'), 
               State('choose_stop_action_1',"value")],
              [Input({"type": "marker", "index": ALL},"n_clicks")],
              prevent_initial_call=True)
def change_marker(St, Cow, stop_operation, *args):

    marker_id = callback_context.triggered[0]["prop_id"].split(".")[0].split(":")[1].split(",")[0]
    n_clicks = callback_context.triggered[0]["value"]

    print('changing marker...')
    #print('marker id?:', marker_id)
    print('requested Marker Operation:')
    print(stop_operation)
    
    if stop_operation == "DM":      
        del St[int(marker_id)]
        del Cow[int(marker_id)]
    
        markers = []
        for i, pos in enumerate(St): 
            if Cow[i]==1:
                custom_icon = custom_icon_coworking
            else:
                custom_icon = custom_icon_bus
            tmp = dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon, id={'type': 'marker', 'index': i})    
            markers.append(tmp)    
        newMap = dl.Map([dl.TileLayer(),dl.ScaleControl(position="topright")] + markers,
                        center=center, zoom=12, id="map_1",
                        style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"})
        #markers = [dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon_bus, id={'type': 'marker', 'index': i}) for i, pos in enumerate(St)]
        #newMap = dl.Map([dl.TileLayer(),dl.ScaleControl(position="topright")] + markers,
        #              center=center, zoom=12, id="map",
        #              style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"})
        return ['Stop deleted!',St,Cow,' ',newMap]

    if stop_operation == "SO":
        print()
        print()
        tmp = St[int(marker_id)]
        St.pop(int(marker_id))
        Cow.pop(int(marker_id))
        St.insert(0, tmp)
        Cow.insert(0, 0)
        print('list modified!')
        print()
        
        markers = [dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon_bus, id={'type': 'marker', 'index': i}) for i, pos in enumerate(St)]
        newMap = dl.Map([dl.TileLayer(),dl.ScaleControl(position="topright")] + markers,
                     center=center, zoom=12, id="map_1",
                     style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"})
        return ['Origin set!',St,Cow,' ',newMap]
        
    if stop_operation == "SC":
       markers = []
       for i, pos in enumerate(St): 
           if i == int(marker_id) or Cow[i]==1:
               custom_icon = custom_icon_coworking
               Cow[i] = 1               
           else:
               custom_icon = custom_icon_bus
           tmp = dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon, id={'type': 'marker', 'index': i})    
           markers.append(tmp)    
       newMap = dl.Map([dl.TileLayer(),dl.ScaleControl(position="topright")] + markers,
                     center=center, zoom=12, id="map_1",
                     style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"})
       return ['Stop deleted!',St,Cow,' ',newMap]



if __name__ == '__main__':
    #app.run(debug=True,port=80)
    app.run_server(debug=True, host='0.0.0.0', port=80)
