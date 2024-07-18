import dash
#import dash_html_components as html
#import dash_core_components as dcc
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import dash_loading_spinners as dls
import dash_leaflet as dl
import dash_leaflet.express as dlx

import sys
import os
from os import listdir

root_dir = 'C:/Users/gfotidellaf/repositories/UI_SCP/assets/'
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
im2 = root_dir +'images/DFG_logo.png'
im3 = root_dir +'images/MUBIL_logo.png'

#stops_file = "/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/GTFS_files_bus_stops_12_02_2024/all_stops_12_02_2024.csv"
#stops_file = "/home/beppe23/mysite/assets/all_stops_12_02_2024.csv"
#stops_file = "C:/Users/gfotidellaf/Desktop/CSL_Gipuzkoa/Accessibility/assets/all_stops_12_02_2024.csv"
stops_file = root_dir +'data/all_bus_stops.csv'


from PIL import Image
image1 = Image.open(im1)
image2 = Image.open(im2)
image3 = Image.open(im3)






app = dash.Dash()

input_group_Row = dbc.Row([ 
     dbc.Col([        
                html.P("Enter number", id="text"),
                dbc.Input(id='integer',placeholder="Enter int")
            ]),     
     dbc.Col([
        dbc.Button('New Tab', color='primary',id='new_Tab', n_clicks=0),
            ]) 
    ])

app.layout = html.Div([input_group_Row, html.Div(id='output-content')])

map_center = (43.26852347667122, -1.9741372404905988)
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 160,
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
    "margin-left": "30rem",
    "margin-right": "25rem",
}

INDICATORS_STYLE = {
    "background-color": "#f8f9fa",
    "position": "fixed",
    "top": 160,
    "right": 20,
    "bottom": 0,
    "width": "20rem"    
}
"""
    "position": "fixed",
    "top": 0,
    "right": 0,
    "bottom": 0,
    "width": "12rem",
    "padding": "2rem 1rem",
"""


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


sidebar =  html.Div(
       [
        html.P(['Import worker file'],id='import_text',style={"margin-top": "15px","font-weight": "bold"}),
        dcc.Upload(
             id='upload-data',
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
             multiple=True
        ),
        dcc.Store(id='worker_data', data=[]),
        html.P([ html.Br(),'Choose number of clusters'],id='cluster_num',style={"margin-top": "15px","font-weight": "bold"}),        
        dbc.Popover(
                  dbc.PopoverBody(mouse_over_mess_clusters), 
                  target="n_clusters",
                  body=True,
                  trigger="hover",style = {'font-size': 12, 'line-height':'2px'},
                  placement= 'right',
                  is_open=False),
        #dcc.Input(id="n_clusters", type="text", value='19'),
        dcc.Slider(1, 30, 1,
               value=19,
               id='n_clusters',
               marks=None,
               tooltip={"placement": "bottom", "always_visible": True}
            ) , 

        dbc.Button("Visualize workers", id="show_workers", n_clicks=0,style={"margin-top": "15px","font-weight": "bold"}),
        html.Br(),
        dbc.Button("Calculate baseline scenario", id="calc_baseline", n_clicks=0,style={"margin-top": "15px","font-weight": "bold"}),
        html.Br(),
        dbc.Button("Visualize scenarios", id="visualize_scenarios", n_clicks=0,style={"margin-top": "15px","font-weight": "bold"}),          
        html.P([ html.Br(),'Select type of interventions'],id='intervention_select',style={"margin-top": "15px", "font-weight": "bold"}),
        dcc.Dropdown(interventions, multi=False,style={"margin-top": "15px"}, id='choose_intervention'),
        html.P([ html.Br(),'Select action for markers'],id='action_select',style={"margin-top": "15px", "font-weight": "bold"}),
        dcc.Dropdown(stops_actions, multi=False,style={"margin-top": "15px"}, id='choose_stop_action'),           
        html.Div([
                 html.Div(id='outdata', style={"margin-top": "15px"}),
                 dcc.Store(id='internal-value_stops', data=[]),
                 dcc.Store(id='internal-value_coworking', data=[]),        
                 dcc.Store(id='internal-value_routes', data=[])
                 ],
                 id='sidebar_intervention', style={"margin-top": "15px"})
        ],
       id='sidebar',
       style=SIDEBAR_STYLE)

central_panel = html.Div(
       [
          html.Div([
             html.Img(src=image1,style={'width':'40%', "display": "inlineBlock", "verticalAlign": "top"}),
             html.Img(src=image2,style={'width':'25%',"display": "inlineBlock", "verticalAlign": "top"}),
             html.Img(src=image3,style={'width':'25%',"display": "inlineBlock", "verticalAlign": "top"})

             ],style= {'verticalAlign': 'top'}),
          dls.Clock(
                    children=[dl.Map([dl.TileLayer(),
                    dl.ScaleControl(position="topright")], center=map_center, 
                                     zoom=12,
                                     id="map",style={'width': '100%', 'height': '80vh', 'margin': "auto", "display": "block"})
                    ],
                    color="#435278",
                    speed_multiplier=1.5,
                    width=80,
                    show_initially=False
                    )
    ],
    style=CONTENT_STYLE)

indicators = html.Div(
        [           
          html.P([ html.Br(),'Liters of gasoline per kilometer (bus)'],id='gas_km',style={"margin-top": "15px","font-weight": "bold"}),
          #dcc.Input(id="choose_gas_km", type="text", value='1.12'),
          dcc.Slider(0, 10,0.05,
               value=1.12,
               id='choose_gas_km',
               marks=None,
               tooltip={"placement": "bottom", "always_visible": True}
          ) ,          
          html.P([ html.Br(),'CO2 Kg per lt'],id='CO2_lt',style={"margin-top": "15px","font-weight": "bold"}),
          #dcc.Input(id="choose_CO2_lt", type="text", value='2.3', style={"margin-bottom": "15px"}),             
          dcc.Slider(0, 10,0.05,
               value=2.3,
               id='choose_CO2_lt',
               marks=None,
               tooltip={"placement": "bottom", "always_visible": True}
          ),
          dbc.Row(
            [
                dbc.Col(
                    html.Div(html.P(['Choose trip time'],style={"font-weight": "bold"})),
                    style={"margin-top": "15px"},
                    width="auto"
                ),                
                dbc.Col(
                    html.Div(dcc.Dropdown(choose_transp_hour, multi=False, id='choose_transp_hour')),
                    style={"margin-top": "15px"},
                    width=4
                ),
                dbc.Col(
                    html.Div(dcc.Loading(html.Div(id="running_MCM"), id="loading-component_MCM")),
                    style={"margin-top": "15px"},
                    width="auto"
                ),
                dbc.Col(
                    html.Div(dbc.Button("Run Mode Choice", id="run_MCM", n_clicks=0, disabled=True)),
                    style={"margin-top": "15px"},
                    width="auto"
                )
            ]
          ),
          html.Div([
              dcc.Graph(
                figure={
                        'data': [{
                                'labels': [1, 2, 3], 
                                'values': [1, 2, 3], 
                                'type': 'pie',
                                }],
                        'layout': {
                            'title': 'Transport share'
                        }        
                }, id='graph', 
                style={'width':'60vh'})
            ], style={'width':'100%'})
        ],
        style=INDICATORS_STYLE)



"""
@app.callback(
    Output("output-content", "children"),
    [Input("load", "n_clicks")],
    [State("integer", "value")],
)
"""
#    [State("integer", "value")],
@app.callback(
    Output("output-content", "children"),
    [Input("new_Tab", "n_clicks")],
    [State("integer", "value")],
)
def render_tabs(click1, integ):
    output = ""
    ctx = dash.callback_context
    action = ctx.triggered[0]["prop_id"].split(".")[0]
    print(action)

    #if action == "load":
    if integ != None:
        output = int(integ)

        tabs = []
        for num in range(output):
            content = [
            html.H3(f'Tab {num + 1}'),  
            dbc.Container(
            [
            dbc.Row([
                dbc.Col(sidebar, width=2, className='bg-light'),
                dbc.Col(central_panel, width=7),
                dbc.Col(indicators, width=3)
                ],style={"height": "100vh"}
                ),
            ],
            fluid=True)  
            ]
            tabs.append(
                dcc.Tab(
                    label=f"Tab {num + 1}",
                    value=f"tab{num + 1}",
                    children=[html.Div(content)]
                )
            )

        return dcc.Tabs(
            id="tab",
            value="tab1",
            children=tabs,
        )

if __name__ == '__main__':
    app.run(debug=True,port=8050)
