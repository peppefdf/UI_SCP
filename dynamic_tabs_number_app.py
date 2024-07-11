import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State


app = dash.Dash()

input_group_Row = dbc.Row([ 
     dbc.Col([        
                html.P("Enter number", id="text"),
                dbc.Input(id='integer',placeholder="Enter int")
            ]),     
     dbc.Col([
        dbc.Button('Enter', color='primary',id='load', n_clicks=0),
            ]) 
    ])
app.layout = html.Div([input_group_Row, html.Div(id='output-content')])


#    [State("integer", "value")],
@app.callback(
    Output("output-content", "children"),
    [Input("load", "n_clicks")],
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
            dcc.Graph(
                figure={
                    'data': [{
                        'x': [1, 2, 3],
                        'y': [3, 1, 2 + num],
                        'type': 'bar'
                    }]
                }
            )
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