import os
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro

import numpy as np
import pandas as pd
import scipy

# https://plotly.com/python/v3/normality-test/

def test_dist_norm(residuals):
    stat, p = shapiro(residuals)  # suitable for small samples

    # interpret
    alpha = 0.05
    if p > alpha:
        msg = 'Sample looks Gaussian (fail to reject H0)'
    else:
        msg = 'Sample does not look Gaussian (reject H0)'

    result_mat = [
        ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
        [len(residuals), stat, p, msg]
    ]

    swt_table = ff.create_table(result_mat)
    swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
    swt_table['layout']['height']=200
    swt_table['layout']['margin']['t']=50
    swt_table['layout']['margin']['b']=50

    swt_table.show()
    

def qqplotly(qqplot_data):
    fig = go.Figure()

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[0].get_xdata(),
        'y': qqplot_data[0].get_ydata(),
        'mode': 'markers',
        'marker': {
            'color': 'black',
            'size': 10
        }
    })

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[1].get_xdata(),
        'y': qqplot_data[1].get_ydata(),
        'mode': 'lines',
        'line': {
            'color': 'lightslategray'
        }

    })


    fig['layout'].update({
        'title': 'Quantile-Quantile Plot',
        'xaxis': {
            'title': 'Theoritical Quantities',
            'gridcolor': 'lightgrey'
        },
        'yaxis': {
            'title': 'Residual Quantities',
            'gridcolor': 'lightgray'
        },
        'showlegend': False,
        'width': 800,
        'height': 700,
        'plot_bgcolor': 'white',
        'font': {
            'size': 20,
            'color': 'black'
        }
    })


    fig.show()
    
    
def scatter_pred(actual, predicted):
    fig = go.Figure()

    fig.add_trace({
        'type': 'scatter',
        'x': predicted,
        'y': actual,
        'mode': 'markers',
        'marker': {
            'color': 'black',
            'size': 10
        }
    })

    fig.add_trace({
        'type': 'scatter',
        'x': actual,
        'y': actual,
        'mode': 'lines',
        'line': {
            'color': 'lightslategray',
        }

    })


    fig['layout'].update({
        'title': 'Predicted vs Actual',
        'xaxis': {
            'title': 'Predicted values',
            'gridcolor': 'lightgrey'
        },
        'yaxis': {
            'title': 'Actual values',
            'gridcolor': 'lightgray'
        },
        'showlegend': False,
        'width': 800,
        'height': 700,
        'plot_bgcolor': 'white',
        'font': {
            'size': 20,
            'color': 'black'
        }
    })
    fig.show()
    
    
    
def plot_resid(predicted, residuals):
    
    standarized_resid = (residuals - residuals.mean()) / residuals.std()
    
    fig = go.Figure()
    
    fig.add_trace({
        'type': 'scatter',
        'x': predicted,
        'y': standarized_resid,
        'mode': 'markers',
        'marker': {
            'color': 'black',
            'size': 10
        }
    })

    fig.add_trace({
        'type': 'scatter',
        'x': predicted,
        'y': [np.mean(standarized_resid)]*len(predicted),
        'mode': 'lines',
        'line': {
            'color': 'lightslategray',
            'dash': 'dash'
        }

    })


    fig['layout'].update({
        'title': 'Residuals',
        'xaxis': {
            'title': 'Predicted values',
            'gridcolor': 'lightgrey'
        },
        'yaxis': {
            'title': 'Standarized residuals',
            'gridcolor': 'lightgray'
        },
        'showlegend': False,
        'width': 800,
        'height': 700,
        'plot_bgcolor': 'white',
        'font': {
            'size': 20,
            'color': 'black'
        }
    })
    fig.show()