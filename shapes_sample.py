from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

app = Dash(__name__)


app.layout = html.Div([
    html.H4('Robot Lidar 2D simulation'),
    dcc.Graph(id="robo_sim"),
    # html.P("Change the position of the right-most data point:"),
    # html.Button("Move Up", n_clicks=0, 
    #             id='shapes-x-btn-up'),
    # html.Button("Move Down", n_clicks=0,
    #             id='shapes-x-btn-down'),
])

@app.callback(
    Output("robo_sim", "figure"))
    # Input("shapes-x-btn-up", "n_clicks"),
    # Input("shapes-x-btn-down", "n_clicks"))
def make_shape_taller(n_up, n_down):
    n = n_up-n_down
    fig = go.Figure(go.Scatter(
        x=[1, 0, 2, 1], y=[2, 0, n, 5], # replace with your own data source
        fill="toself"
    ))
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
