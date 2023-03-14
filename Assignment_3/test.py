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

text = "i am allergic to cat hate them"
threshold = 0.7
# i = 0
# print(i+1,os.getcwd())
os.chdir('./Assignment_2/')
train=pd.read_csv(r"./Data/Training Data.csv")
val=pd.read_csv(r"./Data/Validation Data.csv")
test=pd.read_csv(r"./Data/Test Data.csv")

spam = test.where(test.Label==1).Text[2]
nonspam = test.where(test.Label==0).Text[0]

loaded_model = mlflow.sklearn.load_model('models:/multilayer-perceptron-model/None')

# os.chdir('../')
os.chdir('../')

label,propensity=score.score(text,loaded_model,threshold)

class TestFunction:
    def smoke_test(self):
        assert label!= None
        assert propensity!= None
    
    def test_format(self):
        assert type(text) == str
        assert type(threshold) == float 
        assert type(label) == numpy.int64
        assert type(propensity) == numpy.float64 

    # Checking whether prediction value 0 or 1 
    def test_pred(self):
        assert label == False or label == True

    # Checking whether propensity value between 0 or 1 
    def test_prop(self):
        assert propensity>=0 and propensity<=1

    #When threshold is set to 0
    def threshold_test_0(self):
        label,propensity=score.score(text,loaded_model,threshold=0)
        assert label == True

    #When threshold is set to 1
    def threshold_test_1(self):
        label,propensity=score.score(text,loaded_model,threshold=1)
        assert label == False

    #Assertion on Spam SMS
    def test_spam(self):
        label,propensity=score.score(spam,loaded_model,threshold)
        assert label == True

    #Assertion on Non-spam SMS
    def test_nonspam(self):
        label,propensity=score.score(nonspam,loaded_model,threshold)
        assert label == False


class TestFlask(unittest.TestCase):
    def test_flask(self):
        # Launch the Flask app using os.system
        os.system('python Assignment_3/app.py &')

        # Wait for the app to start up
        time.sleep(1)

        # Make a request to the endpoint
        response = requests.get('http://127.0.0.1:5000/')
        print(response.status_code)

        # Assert that the response is what we expect
        self.assertEqual(response.status_code, 200)
        print("OK")
        self.assertEqual(type(response.text), str)
        print("OKAY")

        # Shut down the Flask app using os.system
        os.system('kill $(lsof -t -i:5000)')


if __name__ == '__main__':
    unittest.main()

