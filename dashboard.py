# dashboard.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

def create_dashboard(server):
    dash_app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

    # Load the dataset with clusters
    data = pd.read_csv('churn_with_clusters.csv')  # Ensure this file has the 'Cluster' column

    dash_app.layout = html.Div([
        html.H1('Bank Customer Churn Dashboard'),
        dcc.Tabs(id='tabs', value='tab-1', children=[
            dcc.Tab(label='Customer Data', value='tab-1'),
            dcc.Tab(label='Model Performance', value='tab-2'),
            dcc.Tab(label='Customer Segmentation', value='tab-3')
        ]),
        html.Div(id='tabs-content')
    ])

    @dash_app.callback(Output('tabs-content', 'children'),
                      [Input('tabs', 'value')])
    def render_content(tab):
        if tab == 'tab-1':
            fig = px.histogram(data, x='Age', title='Customer Age Distribution')
            return html.Div([dcc.Graph(figure=fig)])
        elif tab == 'tab-2':
            model_performance = pd.DataFrame({
                'Model': ['Random Forest', 'KNN', 'SVM', 'Logistic Regression'],
                'Accuracy': [0.86, 0.82, 0.85, 0.80]  # Placeholder values
            })
            fig = px.bar(model_performance, x='Model', y='Accuracy', title='Model Performance Comparison')
            return html.Div([dcc.Graph(figure=fig)])
        elif tab == 'tab-3':
            # Use 'Cluster' column directly
            fig = px.scatter(data, x='Age', y='Balance', color='Cluster', title='Customer Segmentation')
            return html.Div([dcc.Graph(figure=fig)])

    return dash_app
