#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pickle
# from dashapp import app as application
import pandas as pd
import plotly.graph_objs as go
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import apyori as ap
from apyori import apriori 
import mlxtend as ml
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import base64
import warnings
warnings.filterwarnings('ignore')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

global model1
global model2
global model3

global xx,yy,xx2,x3,yy2,y3
#loading preprocessed data
xx=pd.read_csv("xx.csv")
yy=pd.read_csv("yy.csv")
xx2=pd.read_csv("xx2.csv")
x3=pd.read_csv("x3.csv")
yy2=pd.read_csv("yy2.csv")
y3=pd.read_csv("y3.csv")

image_filename = './ipl.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.layout = html.Div(children=[html.Div([html.Div([html.H1(children='IPL Predictions',style={'fontFamily': 'Arial','color':'blue','width':'100%','textAlign': 'center','display': 'inline-block'})])]),
    html.Div([html.Div([html.H1(children='Score Predictor'),
    dcc.Dropdown(
        id = 'sp-algo',
        options=[
            {'label': 'Linear Regression', 'value': 'lr'},
            {'label': 'Random Forest Regressor', 'value': 'rf'},
            {'label': 'Neural Networks', 'value': 'nn'}
        ],
        value = 'lr'
    ),html.Div([html.Label('Model Params: '),
        html.Label('Max Depth: '),
            dcc.Input(
        id='md1',
        type='number',
        step=1,
        value=10
    ),html.Label(' Number of estimators: '),dcc.Input(
        id='ne1',
        type='number',
        step=1,
        value=100
    )
    ],id="rf1",style= {'display': 'none'}),
    html.Div([html.Label('Model Params: '),
        html.Label('Max Iterations: '),
            dcc.Input(
        id='mi1',
        type='number',
        step=1,
        value=1000
    ),html.Label('Layer 1 Neurons: '),dcc.Input(
        id='l11',
        type='number',
        step=1,
        value=10
    ),
    html.Label('Layer 2 Neurons: '),dcc.Input(
        id='l21',
        type='number',
        step=1,
        value=10
    ),
    html.Label('Layer 3 Neurons: '),dcc.Input(
        id='l31',
        type='number',
        step=1,
        value=10
    ),
    html.Label('Activation: '),dcc.Dropdown(
        id = 'act1',
        options=[
            {'label': 'Relu', 'value': 'relu'},
            {'label': 'logistic', 'value': 'logistic'},
            {'label': 'tanh', 'value': 'tanh'}
        ],
        value = 'relu'
    )
    ],id="nn1",style= {'display': 'none','color':'green','width':'100%'}),
    html.Div([
    html.Label('Runs at 5 Overs: '),
            dcc.Input(
        id='5ov1r',
        type='number',
        step=1,
        value=30
    ),html.Label(' Wickets at 5 Overs: '),dcc.Input(
        id='5ov1w',
        type='number',
        step=1,
        value=1
    ),html.Label(' Runs at 10 Overs: '),
    dcc.Input(
        id='10ov1r',
        type='number',
        step=1,
        value=60
    ),html.Label(' Wickets at 10 Overs: '),
    dcc.Input(
        id='10ov1w',
        type='number',
        step=1,
        value=3
    ),html.Label(' Runs at 15 Overs: '),
    dcc.Input(
        id='15ov1r',
        type='number',
        step=1,
        value=90
    ),html.Label(' Wickets at 15 Overs: '),
    dcc.Input(
        id='15ov1w',
        type='number',
        step=1,
        value=5
    ),html.Label(' inning: '),
    dcc.Input(
        id='inn',
        type='number',
        step=1,
        value=1
    ),
    html.Div([html.Div([html.H2(id='score1',children='Scores Pending',style={'color':'green','width':'100%'})])])]),
    html.Div([html.Div([html.H1(children='Win Prediction - Mid Way'),
    dcc.Dropdown(
        id = 'wp1-algo',
        options=[
            {'label': 'Logistic Regression', 'value': 'lr'},
            {'label': 'Random Forest Classifier', 'value': 'rf'},
            {'label': 'Neural Networks', 'value': 'nn'}
        ],
        value = 'lr'
    ),html.Div([html.Label('Model Params: '),
        html.Label('Max Depth: '),
            dcc.Input(
        id='md2',
        type='number',
        step=1,
        value=10
    ),html.Label(' Number of estimators: '),dcc.Input(
        id='ne2',
        type='number',
        step=1,
        value=100
    )
    ],id="rf2",style= {'display': 'none'}),
    html.Div([html.Label('Model Params: '),
        html.Label('Max Iterations: '),
            dcc.Input(
        id='mi2',
        type='number',
        step=1,
        value=1000
    ),html.Label('Layer 1 Neurons: '),dcc.Input(
        id='l12',
        type='number',
        step=1,
        value=10
    ),
    html.Label('Layer 2 Neurons: '),dcc.Input(
        id='l22',
        type='number',
        step=1,
        value=10
    ),
    html.Label('Layer 3 Neurons: '),dcc.Input(
        id='l32',
        type='number',
        step=1,
        value=10
    ),
    html.Label('Activation: '),dcc.Dropdown(
        id = 'act2',
        options=[
            {'label': 'Relu', 'value': 'relu'},
            {'label': 'logistic', 'value': 'logistic'},
            {'label': 'tanh', 'value': 'tanh'}
        ],
        value = 'relu'
    )
    ],id="nn2",style= {'display': 'none','color':'green','width':'100%'}),
    html.Div([html.Label('toss win(0/1): '),
            dcc.Input(
        id='toss2',
        type='number',
        step=1,
        value=1
    ),
    html.Label(' Runs at 5 Overs: '),
            dcc.Input(
        id='5ov2r',
        type='number',
        step=1,
        value=30
    ),html.Label(' Runs at 10 Overs: '),
    dcc.Input(
        id='10ov2r',
        type='number',
        step=1,
        value=60
    ),html.Label(' Runs at 15 Overs: '),
    dcc.Input(
        id='15ov2r',
        type='number',
        step=1,
        value=90
    ),html.Label(' Runs at 20 Overs: '),
    dcc.Input(
        id='20ov2r',
        type='number',
        step=1,
        value=120
    ),html.Label(' Wickets at 20 Overs: '),
    dcc.Input(
        id='20ov2w',
        type='number',
        step=1,
        value=5
    ),html.Label(' Average First Inning score on the ground: '),
    dcc.Input(
        id='avginn2',
        type='number',
        step=1,
        value=150
    ),html.Label(' Number of wins in last 2 matches: '),
    dcc.Input(
        id='wins2',
        type='number',
        step=1,
        value=1
    )])]),
    html.Div([html.Div([html.H2(id='score2',children='Result Pending',style={'color':'green','width':'100%'})])])]),
    html.Div([html.Div([html.H1(children='Win Prediction - 10 Overs Left'),
    dcc.Dropdown(
        id = 'wp2-algo',
        options=[
            {'label': 'Logistic Regression', 'value': 'lr'},
            {'label': 'Random Forest Classifier', 'value': 'rf'},
            {'label': 'Neural Networks', 'value': 'nn'}
        ],
        value = 'lr'
    ),html.Div([html.Label('Model Params: '),
        html.Label('Max Depth: '),
            dcc.Input(
        id='md3',
        type='number',
        step=1,
        value=10
    ),html.Label(' Number of estimators: '),dcc.Input(
        id='ne3',
        type='number',
        step=1,
        value=100
    )
    ],id="rf3",style= {'display': 'none'}),
    html.Div([html.Label('Model Params: '),
        html.Label('Max Iterations: '),
            dcc.Input(
        id='mi3',
        type='number',
        step=1,
        value=1000
    ),html.Label('Layer 1 Neurons: '),dcc.Input(
        id='l13',
        type='number',
        step=1,
        value=10
    ),
    html.Label('Layer 2 Neurons: '),dcc.Input(
        id='l23',
        type='number',
        step=1,
        value=10
    ),
    html.Label('Layer 3 Neurons: '),dcc.Input(
        id='l33',
        type='number',
        step=1,
        value=10
    ),
    html.Label('Activation: '),dcc.Dropdown(
        id = 'act3',
        options=[
            {'label': 'Relu', 'value': 'relu'},
            {'label': 'logistic', 'value': 'logistic'},
            {'label': 'tanh', 'value': 'tanh'}
        ],
        value = 'relu'
    )
    ],id="nn3",style= {'display': 'none','color':'green','width':'100%'}),
    html.Div([html.Div([html.Label('Innings 1: ')]),
    html.Label('toss win(0/1): '),
            dcc.Input(
        id='toss3',
        type='number',
        step=1,
        value=1
    ),
    html.Label(' Runs at 5 Overs: '),
            dcc.Input(
        id='5ov3r',
        type='number',
        step=1,
        value=30
    ),html.Label(' Runs at 10 Overs: '),
    dcc.Input(
        id='10ov3r',
        type='number',
        step=1,
        value=60
    ),html.Label(' Runs at 15 Overs: '),
    dcc.Input(
        id='15ov3r',
        type='number',
        step=1,
        value=90
    ),html.Label(' Runs at 20 Overs: '),
    dcc.Input(
        id='20ov3r',
        type='number',
        step=1,
        value=120
    ),html.Label(' Average First Inning score on the ground: '),
    dcc.Input(
        id='avginn3',
        type='number',
        step=1,
        value=150
    ),html.Label(' Number of wins in last 2 matches: '),
    dcc.Input(
        id='wins3',
        type='number',
        step=1,
        value=1
    ),html.Div([html.Label('Innings 2: ')]),
    html.Label('Runs at 5 Overs: '),
    dcc.Input(
        id='5ov32r',
        type='number',
        step=1,
        value=30
    ),html.Label(' Runs at 10 Overs: '),
    dcc.Input(
        id='10ov32r',
        type='number',
        step=1,
        value=60
    ),html.Label(' Wickets at 10 Overs: '),
    dcc.Input(
        id='10ov32w',
        type='number',
        step=1,
        value=1
    )
    ])]),
    html.Div([html.Div([html.H2(id='score3',children='Result Pending',style={'color':'green','width':'100%'})]),html.Div(id='hidden-div', style={'display':'none'})])])
])],style={'fontFamily': 'Arial','width':'100%'})])


@app.callback(
   Output(component_id='rf1', component_property='style'),
   [Input(component_id='sp-algo', component_property='value')])
def show_elem_rf1(val):
    if val=="rf":
        return {'display': 'block','color':'red','width':'100%'}
    else:
        return {'display': 'none','color':'red','width':'100%'}

@app.callback(
   Output(component_id='nn1', component_property='style'),
   [Input(component_id='sp-algo', component_property='value')])
def show_elem_nn1(val):
    if val=="nn":
        return {'display': 'block','color':'red','width':'100%'}
    else:
        return {'display': 'none','color':'red','width':'100%'}

@app.callback(
   Output(component_id='rf2', component_property='style'),
   [Input(component_id='wp1-algo', component_property='value')])
def show_elem_rf2(val):
    if val=="rf":
        return {'display': 'block','color':'red','width':'100%'}
    else:
        return {'display': 'none','color':'red','width':'100%'}

@app.callback(
   Output(component_id='nn2', component_property='style'),
   [Input(component_id='wp1-algo', component_property='value')])
def show_elem_nn2(val):
    if val=="nn":
        return {'display': 'block','color':'red','width':'100%'}
    else:
        return {'display': 'none','color':'red','width':'100%'}

@app.callback(
   Output(component_id='rf3', component_property='style'),
   [Input(component_id='wp2-algo', component_property='value')])
def show_elem_rf3(val):
    if val=="rf":
        return {'display': 'block','color':'red','width':'100%'}
    else:
        return {'display': 'none','color':'red','width':'100%'}

@app.callback(
   Output(component_id='nn3', component_property='style'),
   [Input(component_id='wp2-algo', component_property='value')])
def show_elem_nn3(val):
    if val=="nn":
        return {'display': 'block','color':'red','width':'100%'}
    else:
        return {'display': 'none','color':'red','width':'100%'}



@app.callback(
    Output('score1','children'),
    [Input('5ov1r', 'value'),Input('5ov1w','value'),Input('10ov1r','value'),Input('10ov1w', 'value'),Input('15ov1r','value'),Input('15ov1w','value'),Input('inn', 'value'),Input('sp-algo', 'value'),Input('md1','value'),Input('ne1','value'),Input('mi1','value'),Input('l11','value'),Input('l21','value'),Input('l31','value'),Input('act1','value')]
    )
def update_table(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15):
    data={}
    global model1
    for x,y in zip(["5_total_runs","5_player_dismissed","10_total_runs","10_player_dismissed","15_total_runs","15_player_dismissed",'inning'],[x1,x2,x3,x4,x5,x6,x7]):
        data[x]=y
    inp=pd.DataFrame([data])
    global model1,xx,yy
    if x8=="lr":
        model1=LinearRegression()
        model1.fit(xx,yy)
    elif x8=="rf":
        model1=RandomForestRegressor(max_depth=x9, random_state=0,n_estimators=x10)
        model1.fit(xx,yy)
    else:
        model1=MLPRegressor(max_iter=x11,hidden_layer_sizes=(x12,x13,x14),activation=x15)
        model1.fit(xx,yy)    
    ans=model1.predict(inp)[0]
    print(ans)
    return "Predicted Score: "+str(math.ceil(ans))

@app.callback(
    Output('score2','children'),
    [Input('toss2', 'value'),Input('5ov2r', 'value'),Input('10ov2r','value'),Input('15ov2r','value'),Input('20ov2r', 'value'),Input('20ov2w','value'),Input('avginn2','value'),Input('wins2', 'value'),Input('wp1-algo', 'value'),Input('md2','value'),Input('ne2','value'),Input('mi2','value'),Input('l12','value'),Input('l22','value'),Input('l32','value'),Input('act2','value')]
    )
def update_table1(y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16):
    data1={}
    for x,y in zip(["toss_win","5_total_runs","10_total_runs","15_total_runs","20_total_runs","20_player_dismissed","inn1_avg","roll_win"],[y1,y2,y3,y4,y5,y6,y7,y8]):
        data1[x]=y
    inp=pd.DataFrame([data1])
    print(data1)
    # model=pickle.load( open( "nn_model1.pkl", "rb" ) )
    global model2,xx2,yy2
    if y9=="lr":
        model2=LogisticRegression()
        model2.fit(xx2,yy2)
    elif y9=="rf":
        model2=RandomForestClassifier(max_depth=y10, random_state=0,n_estimators=y11)
        model2.fit(xx2,yy2)
    else:
        model2=MLPClassifier(max_iter=y12,hidden_layer_sizes=(y13,y14,y15),activation=y16)
        model2.fit(xx2,yy2)    
    ans=model2.predict_proba(inp)
    print(ans)
    return "Win Probability: "+str(round(ans[0][1]*100,2))+"%"

@app.callback(
    Output('score3','children'),
    [Input('toss3', 'value'),Input('5ov3r', 'value'),Input('10ov3r','value'),Input('15ov3r','value'),Input('20ov3r', 'value'),Input('5ov32r','value'),Input('10ov32r','value'),Input('10ov32w','value'),Input('avginn3','value'),Input('wins3', 'value'),Input('wp2-algo', 'value'),Input('md3','value'),Input('ne3','value'),Input('mi3','value'),Input('l13','value'),Input('l23','value'),Input('l33','value'),Input('act3','value')]
    )
def update_table2(z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18):
    data2={}
    for x,y in zip(["toss_win_x","5_total_runs_x","10_total_runs_x","15_total_runs_x","20_total_runs_x","5_total_runs_y","10_total_runs_y","10_player_dismissed_y","inn1_avg_x","roll_win"],[z1,z2,z3,z4,z5,z6,z7,z8,z9,z10]):
        data2[x]=y
    inp=pd.DataFrame([data2])
    print(data2)
    # model=pickle.load( open( "nn_model2.pkl", "rb" ) )
    global model3,x3,y3
    if z11=="lr":
        model3=LogisticRegression()
        model3.fit(x3,y3)
    elif z11=="rf":
        model3=RandomForestClassifier(max_depth=z12, random_state=0,n_estimators=z13)
        model3.fit(x3,y3)
    else:
        model3=MLPClassifier(max_iter=z14,hidden_layer_sizes=(z15,z16,z17),activation=z18)
        model3.fit(x3,y3)    
    ans=model3.predict_proba(inp)
    print(ans)
    return "Win Probability: "+str(round(ans[0][1]*100,2))+"%"



if __name__=='__main__':
    app.run_server(debug=True)


# In[ ]:




