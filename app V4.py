import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import single_fea

#os.system('python passrate.py')
#os.system('python LR_prediction.py')
#os.system('python single_fea.py')
#single_fea.single_fea('RH Front Mod Datum[X]')         # RH Front Mod Datum[X] B008_TP_ABC[Y]

# Read csv files created from passrate.py and LR_prediction.py
df_passrate = pd.read_csv('./process/df_passrate.csv')
df_combined_2 = pd.read_csv('./process/df_combined_2.csv')
filter_tol = pd.read_csv('./process/filter_tol.csv')
df_measure_cleaned2 = pd.read_csv('./process/df_measure_cleaned2.csv')

# Read csv files created from single_fea.py
df_LR_single_now = pd.read_csv('./process/df_LR_single_now.csv')
df_LR_single_pred = pd.read_csv('./process/df_LR_single_pred.csv')
OneTol = pd.read_csv('./process/OneTol.csv')

# Create dropdown menu
def get_options(fea_list):
    dict_list = []
    for i in fea_list:
        dict_list.append({'label': i, 'value': i})
    return dict_list


app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children="Proactive Quality Prediction System",),
        html.P(children="Analyze the measurement dataset and plotting the result. Make prediction of quality status in the future.",),

        # Drop down menu for selecting feature names
        dcc.Dropdown(id='fea_selector', options=get_options(df_measure_cleaned2.columns[1:].unique()),
                     value=[df_measure_cleaned2.columns[1:].sort_values()[0]],
                     className='fea_selector',
                     style={"width": "50%", 'background-color': 'honeydew', 'color': 'black'}),

        # Plot for a single feature and prediction based on dropdown menu
        dcc.Graph(id='single_pred', figure={
            'data': [{'x': df_LR_single_now['X'], 'y': df_LR_single_now['Y'], 'type': 'lines', 'name': 'Current',
                      'line': {'color': 'midnightblue'}},
                     {'x': df_LR_single_pred['X_pred'], 'y': df_LR_single_pred['y_pred'], 'type': 'lines',
                      'name': 'Pred_Mean', 'line': {'color': 'indigo', 'dash': 'dot'}},
                     {'x': df_LR_single_pred['X_pred'], 'y': df_LR_single_pred['y_pred_low'], 'type': 'lines',
                      'name': 'Pred - 3sigma', 'line': {'color': 'magenta', 'dash': 'dot'}},
                     {'x': df_LR_single_pred['X_pred'], 'y': df_LR_single_pred['y_pred_hi'], 'type': 'lines',
                      'name': 'Pred + 3sigma', 'line': {'color': 'magenta', 'dash': 'dot'}},
                     ],
            'layout': {'title': 'Single Fea Trend and Prediction', 'xaxis': {'title': 'Feature Name'},
                       'yaxis': {'title': 'Deviation'},
                       'shapes': [{'type': 'line', 'x0': 0, 'x1': len(df_LR_single_pred['y_pred']),
                                   'y0': float(OneTol.iloc[0, 1]), 'y1': float(OneTol.iloc[0, 1]),
                                   'line': {'color': 'orange', 'dash': 'dot'}},
                                  {'type': 'line', 'x0': 0, 'x1': len(df_LR_single_pred['y_pred']),
                                   'y0': float(OneTol.iloc[1, 1]), 'y1': float(OneTol.iloc[1, 1]),
                                   'line': {'color': 'orange', 'dash': 'dot'}}
                                  ],
                       }, }
                  ),

        # Plot for showing overall pass rate and critical features pass rate
        dcc.Graph(figure={
                'data': [{'x':df_passrate.index, 'y':df_passrate['Pass Rate'], 'type':'lines', 'name':'Overall Pass Rate', 'line':{'color':'midnightblue'}},
                         {'x':df_passrate.index, 'y':df_passrate['Critical Pass Rate'], 'type':'lines', 'name':'Critical Features Pass Rate', 'line':{'color':'limegreen'}}],
                'layout': {'title':'Pass Rate (Overall and Critical Features)', 'xaxis':{'title': 'SN'}, 'yaxis':{'title': 'Pass Rate', 'range':[0,1]}},
                        },
                ),

        # Plot for current status of mean and prediction for 2000th and 3000th future products
        dcc.Graph(figure={
                'data': [{'x':df_combined_2['Fea'], 'y':df_combined_2['Mean'], 'type':'lines', 'name':'Mean', 'line':{'color':'midnightblue'}},
                         {'x':df_combined_2['Fea'], 'y':df_combined_2['2k_th'], 'type':'lines', 'name':'2000th', 'line':{'color':'indigo','dash':'dot'}},
                         {'x':df_combined_2['Fea'], 'y':df_combined_2['3k_th'], 'type':'lines', 'name':'3000th', 'line':{'color':'magenta','dash':'dot'}},
                         {'x':df_combined_2['Fea'], 'y':df_combined_2['LSL'], 'type':'lines', 'name':'LSL', 'line':{'color':'orange','dash':'dot'}},
                         {'x':df_combined_2['Fea'], 'y':df_combined_2['USL'], 'type':'lines', 'name':'USL', 'line':{'color':'orange','dash':'dot'}},
                         ],
                'layout': {'title':'Final Product Current Status vs. Prediction', 'xaxis':{'title':'SN'}, 'yaxis':{'title':'Deviation'}},
                        },
                ),
    ]
)

# Callback for dropdown menu
@app.callback(Output('single_pred', 'figure'),
              [Input('fea_selector', 'value')])
def update_graph(selected_dropdown_value):
    # Call single_fea.py to fetch data and create csv accordingly
    single_fea.single_fea(selected_dropdown_value)
    df_LR_single_now = pd.read_csv('./process/df_LR_single_now.csv')
    df_LR_single_pred = pd.read_csv('./process/df_LR_single_pred.csv')
    OneTol = pd.read_csv('./process/OneTol.csv')

    # Update figure
    figure = {'data': [{'x':df_LR_single_now['X'], 'y':df_LR_single_now['Y'], 'type':'lines', 'name':'Current', 'line':{'color':'midnightblue'}},
                       {'x':df_LR_single_pred['X_pred'], 'y':df_LR_single_pred['y_pred'], 'type':'lines', 'name':'Pred_Mean', 'line':{'color':'indigo','dash':'dot'}},
                       {'x':df_LR_single_pred['X_pred'], 'y':df_LR_single_pred['y_pred_low'], 'type':'lines', 'name':'Pred - 3sigma', 'line':{'color':'magenta','dash':'dot'}},
                       {'x': df_LR_single_pred['X_pred'], 'y': df_LR_single_pred['y_pred_hi'], 'type': 'lines', 'name': 'Pred + 3sigma', 'line':{'color':'magenta','dash':'dot'}},
                       ],
              'layout': {'title': 'Single Fea Trend and Prediction', 'xaxis': {'title': 'Feature Name'},
                            'yaxis': {'title': 'Deviation'},
                            'shapes': [{'type':'line', 'x0':0,'x1':len(df_LR_single_pred['y_pred']), 'y0':float(OneTol.iloc[0,1]), 'y1':float(OneTol.iloc[0,1]), 'line':{'color':'orange','dash':'dot'}},
                                   {'type':'line', 'x0':0,'x1':len(df_LR_single_pred['y_pred']), 'y0':float(OneTol.iloc[1,1]), 'y1':float(OneTol.iloc[1,1]), 'line':{'color':'orange','dash':'dot'}}
                                   ],
                         },
              }

    return figure

if __name__ == "__main__":
    app.run_server(debug=True)