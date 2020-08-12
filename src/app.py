import random
import numpy as np
import pandas as pd
import os

from torch import nn, optim
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from PIL import Image

import streamlit as st


random.seed(23)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

st.subheader('NBA SHOTS PREDICTION')

st.sidebar.header('CREATE STRATEGY')

column_list = pd.read_pickle('./data/column_list.pkl')

defender_list = pd.read_pickle('./data/defender_list.pkl')
player_list = pd.read_pickle('./data/player_list.pkl')


defteams = defender_list['OPPONENT'].drop_duplicates().sort_values()
offteams = player_list['TEAM'].drop_duplicates().sort_values()


# GATHER INPUT DATA FROM USER
def user_input_features():
    TEAM = st.sidebar.selectbox('OFFENSIVE TEAM', np.array(offteams))
    player_name = st.sidebar.selectbox('OFFENSIVE PLAYER NAME', np.array(player_list['player_name'][player_list['TEAM']==TEAM])) #TODO this should be a selection box
    OPPONENT = st.sidebar.selectbox('DEFENSIVE TEAM', np.array(defteams))
    CLOSEST_DEFENDER = st.sidebar.selectbox('CLOSEST DEFENDER NAME', np.array(defender_list['CLOSEST_DEFENDER'][defender_list['OPPONENT']==OPPONENT])) #TODO this should be a selection box
    LOCATION = st.sidebar.radio('OFFENSIVE TEAM LOCATION', ('Home','Away')) #TODO this should be a selection box
    PERIOD = st.sidebar.radio('PERIOD', (1,2,3,4))
    SHOT_NUMBER = st.sidebar.slider('SHOT NUMBER', 1, 50, 1)
    GAME_CLOCK = st.sidebar.slider('GAME CLOCK (SECONDS)', 0, 720, 720)
    SHOT_CLOCK = st.sidebar.slider('SHOT CLOCK (SECONDS)', 0, 24, 24)
    DRIBBLES = st.sidebar.slider('DRIBBLES BEFORE SHOT', 0, 50, 0)
    TOUCH_TIME = st.sidebar.slider('TOUCH TIME (SECONDS)', 0, 24, 24)
    SHOT_DIST = st.sidebar.slider('SHOT DISTANCE (FEET)', 0, 50, 0)
    PTS_TYPE = st.sidebar.radio('POINTS TYPE', ('Two', 'Three')) #TODO this should be a radio button
    CLOSE_DEF_DIST = st.sidebar.slider('DEFENDER DISTANCE (FEET)', 0, 50, 0)

    data = {
            'TEAM':TEAM,
            'player_name': player_name,
            'OPPONENT':OPPONENT,
            'CLOSEST_DEFENDER':CLOSEST_DEFENDER,
            'LOCATION': LOCATION,
            'PERIOD':PERIOD,
            'SHOT_NUMBER': SHOT_NUMBER,
            'GAME_CLOCK': GAME_CLOCK,
            'SHOT_CLOCK': SHOT_CLOCK,
            'DRIBBLES': DRIBBLES,           
            'TOUCH_TIME': TOUCH_TIME,           
            'SHOT_DIST': SHOT_DIST,           
            'PTS_TYPE': (2 if PTS_TYPE=='Two' else 3),           
            'CLOSE_DEF_DIST': CLOSE_DEF_DIST           
            }

    features = pd.DataFrame(data, index=[0])
    return features

banner = Image.open('img/banner.jpg')

st.image(banner, width=None)


df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

court = Image.open('img/court.png')

st.image(court, caption='NBA Court', width=None)


# PREDICT

# TODO DATA PRE-PROCESS PIPELINE
def data_process(df):

    data = pd.DataFrame(0, index=np.arange(1), columns=column_list)
    
    # identify location
    LOCATION = 'LOCATION_A' if df['LOCATION'][0]=='AWAY' else 'LOCATION_H'
    df['{}'.format(LOCATION)] = 1

    # identify offense team
    offteam = df['TEAM'][0]
    df['TEAM_{}'.format(offteam)] = 1

    # identify defense team
    defteam = df['OPPONENT'][0]
    df['OPPONENT_{}'.format(offteam)] = 1

    # identify offense player id
    offplayer_name = player_list['player_name'][0]
    offplayer_id = player_list['player_id'][player_list['player_name']==offplayer_name]
    offplayer_id = offplayer_id[0]
    df['player_id_{}p'.format(offplayer_id)] = 1

#    identify offense player id
    defplayer_name = defender_list['CLOSEST_DEFENDER'][0]
    defplayer_id = defender_list['CLOSEST_DEFENDER_PLAYER_ID'][defender_list['CLOSEST_DEFENDER']==defplayer_name]
    defplayer_id = defplayer_id[0]
    df['CLOSEST_DEFENDER_PLAYER_ID_{}d'.format(defplayer_id)] = 1

    df = df.drop(['TEAM', 'player_name', 'OPPONENT', 'CLOSEST_DEFENDER', 'LOCATION'], axis=1)
    
    for cols in df.columns:
        data[cols][0] = df[cols][0]


    data = data.fillna(0)
    return data

processed_data = data_process(df)

# LOAD DNN (PYTORCH) MODEL ARCHITECTURE
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(len(column_list), 200)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.3)
                
        self.fc2 = nn.Linear(200, 200)
        self.prelu = nn.PReLU(1)
        self.dout = nn.Dropout(0.3)
        
        self.fc6 = nn.Linear(200, 100)
        self.prelu = nn.PReLU(1)
        
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        dout = self.dout(h2)
        
        a6 = self.fc6(dout)
        h6 = self.prelu(a6)
        
        a7 = self.out(h6)
        
        y = self.out_act(a7)
        return y

# PREDICT THE OUTCOME
def predict_pytorch(data):
    dtype = torch.FloatTensor

    data_1 = torch.tensor(data.values).type(dtype)

    with torch.no_grad():
        y_pred = model.forward(data_1).cpu().numpy()
    
    return y_pred

model = Classifier()
model.load_state_dict(torch.load(r'./models/model_pytorch.pt'))
model.eval()

preds = predict_pytorch(processed_data)
st.subheader('Shot Prediction')
st.write('{}%'.format(round(preds[0][0]*100, 2))) # SHOULD BE CALCULATED FROM PREDICT FUNCTION
