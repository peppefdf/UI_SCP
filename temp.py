import dash
from dash import Dash
import dash_bootstrap_components as dbc
#import dash_html_components as html
from dash import html, callback_context, ALL
from dash import dcc, Output, Input, State, callback
import dash_leaflet as dl
#import re
import json
import pandas as pd
import geopy.distance
import numpy as np
import os

#from google.colab import drive
#drive.mount('/content/drive',  force_remount=True)

root_dir = 'C:\\Users\\gfotidellaf\\Desktop\\CSL_Gipuzkoa\\Accessibility\\assets\\'

im1 = root_dir + 'CSL_logo.PNG'
im2 = root_dir + 'DFG_logo.png'
#im1 = '/home/beppe23/mysite/assets/CSL_logo.PNG'
#im2 = '/home/beppe23/mysite/assets/DFG_logo.png'

stops_file = root_dir + "all_stops_12_02_2024.csv"
#stops_file = "/home/beppe23/mysite/assets/all_stops_12_02_2024.csv"

Require_file = root_dir + "Requirements.txt"

print('check..')
from PIL import Image
image1 = Image.open(im1)
image2 = Image.open(im2)

os.system('pip freeze > ' +Require_file)

stops_df = pd.read_csv(stops_file, encoding='latin-1')

stops_lat_lon = stops_df[['stop_lat','stop_lon']].to_numpy()

center = (43.26852347667122, -1.9741372404905988)
#    iconUrl= 'https://uxwing.com/wp-content/themes/uxwing/download/location-travel-map/bus-stop-icon.png',
#    iconUrl= "https://i.ibb.co/6n1tzcQ/bus-stop.png",
custom_icon = dict(
    iconUrl= "https://i.ibb.co/HV0K5Fp/bus-stop.png",
    iconSize=[40,40],
    iconAnchor=[22, 40]
)

app = Dash(prevent_initial_callbacks=True)

"""
app.layout = html.Div([
    html.Img(src=image1,style={'width':'40%', "display": "inlineBlock", "verticalAlign": "top"}),
    html.Img(src=image2,style={'width':'25%',"display": "inlineBlock", "verticalAlign": "top"}),
    html.Div(dl.Map([dl.TileLayer()],
                     center=center, zoom=12, id="map",
                     style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"})),
    dbc.Button("Load stops", id="load_stops", n_clicks=0),
    html.Div(id='clickdata'),
    dcc.Store(id='internal-value', data=[])
])
"""
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
}

mouse_over_mess = """
Shifts chosen stops to
<p>closest existing bus stops</p>"""

sidebar =  html.Div(
       [
        html.Button("Load stops", id="load_stops", n_clicks=0),
        html.Br(),
        html.Button("Match stops", id="match_stops", n_clicks=0, style={"margin-top": "15px", "font-weight": "bold"}),
        dbc.Popover(dcc.Markdown(mouse_over_mess, dangerously_allow_html=True),
                  target="match_stops",
                  body=True,
                  trigger="hover",style = {'font-size': 12, 'line-height':'2px'}),
        html.P([ html.Br(),'Choose number of buses'],id='buses_num',style={"margin-top": "15px","font-weight": "bold"}),
        dcc.Input(id="choose_buses", type="text", value='3'),
        html.P([ html.Br(),'Liters of gasoline per kilometer'],id='gas_km',style={"margin-top": "15px","font-weight": "bold"}),
        dcc.Input(id="choose_gas_km", type="text", value='1.12'),
        html.P([ html.Br(),'CO2 gr per lt'],id='CO2_lt',style={"margin-top": "15px","font-weight": "bold"}),
        dcc.Input(id="choose_CO2_lt (Kg/lt)", type="text", value='2.3'),
        html.Button("Calculate routes", id="calc_routes", n_clicks=0,style={"margin-top": "15px"}),
        html.P([ html.Br(),'Select route to be visualized'],id='route_select',style={"margin-top": "15px", "font-weight": "bold"}),
        dcc.Dropdown(["Route 1", "Route 2", "Route 3"], multi=False,style={"margin-top": "15px"},id='dropdown'),
        html.Div(id='clickdata'),
        dcc.Store(id='internal-value', data=[])
       ],
       style=SIDEBAR_STYLE)

content = html.Div(
    [
          html.Div([
             html.Img(src=image1,style={'width':'40%', "display": "inlineBlock", "verticalAlign": "top"}),
             html.Img(src=image2,style={'width':'25%',"display": "inlineBlock", "verticalAlign": "top"}),
          ]),
          html.Div([dl.Map([dl.TileLayer()], center=center, zoom=12, id="map",style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"})
          ])
    ],
    style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback([Output("dropdown", "options")],
              State('choose_buses',"value"),
              [Input("calc_routes", "n_clicks")]
              )
def calc_CO2(Nroutes,Nclick):
    list_routes = range(1,int(Nroutes)+1)
    new_menu = [{'label': 'Route ' +str(i), 'value': i} for i in list_routes]
    return [new_menu]



@app.callback([Output("clickdata", "children",allow_duplicate=True), Output('internal-value','data',allow_duplicate=True),Output('map','children',allow_duplicate=True)],
              [State('internal-value','data')],
              [Input("match_stops", "n_clicks")]
              )
def match_stops(St,Nclicks):
    bus_stops = []
    out = ''
    for i_st in range(len(St)):
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
    markers = [dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon, id={'type': 'marker', 'index': i}) for i, pos in enumerate(St)]
    newMap = dl.Map([dl.TileLayer()] + markers,
                     center=center, zoom=12, id="map",
                     style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"})
    return [out,St,newMap]


#@app.callback([Output("clickdata", "children")],
@app.callback([Output("clickdata", "children"), Output('internal-value','data',allow_duplicate=True),Output('map','children',allow_duplicate=True)],
              [Input("load_stops", "n_clicks")]
              )
def load_stops(N):
    filename = '/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/Accessibility_Map/INPUT_stops.csv'
    df = pd.read_csv(filename)
    #out=St.loc[:'Lat']
    #for i in range(len(St)):
    #    out = out + str(St.loc[i,['Lat']]) + ', ' + str(St.loc[i,['Lon']]) + '; '
    out = ''
    St = []
    for ind in df.index:
         out = out + str(df['Lat'][ind]) + ',' + str(df['Lon'][ind]) +';'
         St.append((df['Lat'][ind],df['Lon'][ind]))
    markers = [dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon, id={'type': 'marker', 'index': i}) for i, pos in enumerate(St)]
    newMap = dl.Map([dl.TileLayer()] + markers,
                     center=center, zoom=12, id="map",
                     style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"})
    #return [out,St,newMap]
    return [out,St,newMap]


@app.callback([Output("clickdata", "children",allow_duplicate=True), Output('internal-value','data',allow_duplicate=True),Output('map','children',allow_duplicate=True)],
              [State('internal-value','data')],
              [Input('map','clickData')]
              )
def add_marker(St,clickd):
    marker_lat = clickd['latlng']['lat']
    marker_lon = clickd['latlng']['lng']
    St.append((marker_lat,marker_lon))
    out=''
    for i in range(len(St)):
        out = out + str(St[i][0]) + ', ' + str(St[i][1]) + '; '
    #markers = [dl.Marker(dl.Tooltip("double click on Marker after drag to update its position"), position=pos, id="marker{}".format(i), draggable=True) for i, pos in enumerate(St)]
    #Layer_group = dl.LayerGroup(markers, id="markers_group")
    #Layer_group = dl.LayerGroup(markers, id={"type": "markers_group", "index": 0})
    #markers = [dl.Marker(id={'type': 'marker', 'index': key}, position=data[key]) for key in data]
    markers = [dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon, id={'type': 'marker', 'index': i}) for i, pos in enumerate(St)]
    newMap = dl.Map([dl.TileLayer()] + markers,
                     center=center, zoom=12, id="map",
                     style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"})
    return [out,St,newMap]

@app.callback([Output("clickdata", "children",allow_duplicate=True),Output('internal-value','data',allow_duplicate=True),Output('map','children',allow_duplicate=True)],
              [State('internal-value','data')],
              [Input({"type": "marker", "index": ALL},"n_clicks")],
              prevent_initial_callbacks=True)
def remove_marker(St,*args):
    marker_id = callback_context.triggered[0]["prop_id"].split(".")[0].split(":")[1].split(",")[0]
    n_clicks = callback_context.triggered[0]["value"]
    if n_clicks ==2:
       del St[int(marker_id)]
       out=''
       for i in range(len(St)):
           out = out + str(St[i][0]) + ', ' + str(St[i][1]) + '; '
       markers = [dl.Marker(dl.Tooltip("Double click on Marker to remove it"), position=pos, icon=custom_icon, id={'type': 'marker', 'index': i}) for i, pos in enumerate(St)]
       newMap = dl.Map([dl.TileLayer()] + markers,
                     center=center, zoom=12, id="map",
                     style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"})
       return [out,St,newMap]


if __name__ == '__main__':
    #app.run_server(Debug=True)
    app.run_server(port=8058,Debug=True)