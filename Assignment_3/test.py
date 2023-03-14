import score
import pickle
import numpy
import os
import requests
import subprocess
import time
import unittest
import mlflow
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Defining input values to test the score function on
message = "i am allergic to cat hate them"
threshold = 0.7

os.chdir('./Assignment_2/')
train=pd.read_csv(r"./Data/Training Data.csv")
val=pd.read_csv(r"./Data/Validation Data.csv")
test=pd.read_csv(r"./Data/Test Data.csv")

# Defining input values to test the score function on
spam = test.where(test.Label==1).Text[2]
nonspam = test.where(test.Label==0).Text[0]

# Importing Model
loaded_model = mlflow.sklearn.load_model('models:/multilayer-perceptron-model/None')

os.chdir('../')

# Defining Unit Tests

# Smoke Test: Function returns values without crashing
def test_smoke(text=message,model=loaded_model,threshold=threshold):
    label,propensity=score.score(text,model,threshold)
    assert label!= None
    assert propensity!= None

# Format Test: Check function input/output types 
def test_format(text=message,model=loaded_model,threshold=threshold):
    label,propensity=score.score(text,model,threshold)
    assert type(text) == str
    assert type(threshold) == float 
    assert type(label) == bool
    assert type(propensity) == float 

# Checking whether prediction value 0 or 1 
def test_pred(text=message,model=loaded_model,threshold=threshold):
    label,propensity=score.score(text,model,threshold)
    assert label == False or label == True

# Checking whether propensity value between 0 or 1 
def test_prop(text=message,model=loaded_model,threshold=threshold):
    label,propensity=score.score(text,model,threshold)
    assert propensity>=0 and propensity<=1

#When threshold is set to 0
def test_threshold_0(text=message,model=loaded_model,threshold=0):
    label,propensity=score.score(text,model,threshold)
    assert label == True

#When threshold is set to 1
def test_threshold_1(text=message,model=loaded_model,threshold=1):
    label,propensity=score.score(text,model,threshold)
    assert label == False

#Assertion on Spam SMS
def test_spam(text=spam,model=loaded_model,threshold=threshold):
    label,propensity=score.score(text,model,threshold)
    assert label == True

#Assertion on Non-spam SMS
def test_nonspam(text=nonspam,model=loaded_model,threshold=threshold):
    label,propensity=score.score(nonspam,model,threshold)
    assert label == False


def test_flask():
    # Launch the Flask app using os.system
    os.system('python Assignment_3/app.py &')

    # Wait for the app to start up
    time.sleep(1)

    # Make a request to the endpoint
    response = requests.get('http://127.0.0.1:5000/')
    print(response.status_code)

    # Assert that the response is what we expect
    assert response.status_code == 20
    assert type(response.text) == str

    # Shut down the Flask app using os.system
    os.system('kill $(lsof -t -i:5000)')

