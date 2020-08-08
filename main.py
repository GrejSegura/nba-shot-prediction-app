
import random
import numpy as np
import pandas as pd
import os
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# import lightgbm as lgb
# from sklearn import preprocessing
# import pickle
# import gc

from PIL import Image

import streamlit as st


random.seed(23)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

st.subheader('NBA SHOTS PREDICTION')

st.sidebar.header('User Input Parameters')

def user_input_features():
    player_name = st.sidebar.slider('Player Name', 4.3, 7.9, 5.4) #TODO this should be a selection box
    LOCATION = st.sidebar.slider('Home or Away', 1.0, 6.9, 1.3) #TODO this should be a selection box
    SHOT_NUMBER = st.sidebar.slider('Shot Number', 1.0, 6.9, 1.3)
    GAME_CLOCK = st.sidebar.slider('Game Clock', 0.1, 2.5, 0.2)
    SHOT_CLOCK = st.sidebar.slider('Shot Clock', 0.1, 2.5, 0.2)
    DRIBBLES = st.sidebar.slider('Dribbles', 0.1, 2.5, 0.2)
    TOUCH_TIME = st.sidebar.slider('Touch Time', 0.1, 2.5, 0.2)
    SHOT_DIST = st.sidebar.slider('Shot Distance', 0.1, 2.5, 0.2)
    PTS_TYPE = st.sidebar.slider('Points Type', 0.1, 2.5, 0.2) #TODO this should be a button
    CLOSEST_DEFENDER = st.sidebar.slider('Closest Defender (Name)', 0.1, 2.5, 0.2) #TODO this should be a selection box
    CLOSE_DEF_DIST = st.sidebar.slider('Defender Distance', 0.1, 2.5, 0.2)

    data = {'player_name': player_name,
            'LOCATION': LOCATION,
            'SHOT_NUMBER': SHOT_NUMBER,
            'GAME_CLOCK': GAME_CLOCK,
            'SHOT_CLOCK': SHOT_CLOCK,
            'DRIBBLES': DRIBBLES,           
            'TOUCH_TIME': TOUCH_TIME,           
            'SHOT_DIST': SHOT_DIST,           
            'PTS_TYPE': PTS_TYPE,           
            'CLOSEST_DEFENDER': CLOSEST_DEFENDER,           
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
# iris = datasets.load_iris()
# X = iris.data
# Y = iris.target

# clf = RandomForestClassifier()
# clf.fit(X, Y)

# prediction = clf.predict(df)
# prediction_proba = clf.predict_proba(df)

# st.subheader('Class labels and their corresponding index number')
# st.write(iris.target_names)

st.subheader('Shot Prediction')
st.write('90%')
# #st.write(prediction)

# st.subheader('Prediction Probability')
# st.write(prediction_proba)