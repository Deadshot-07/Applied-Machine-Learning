import score
import pickle, json
import os, time, requests
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Setting up directory path
os.chdir(os.path.abspath(os.path.dirname(__file__)))

# Defining input values to test the score function on
train=pd.read_csv(r"./Data/Training Data.csv")
val=pd.read_csv(r"./Data/Validation Data.csv")
test=pd.read_csv(r"./Data/Test Data.csv")

message = "i am allergic to cat hate them"
threshold = 0.7

spam = test.where(test.Label==1).Text[2]
nonspam = test.where(test.Label==0).Text[0]

# Importing Model
model_path = "./models/mlp_model.sav"
loaded_model = pickle.load(open(model_path, "rb"))

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
    os.system('start /b python Assignment_4/app.py &')

    # Wait for the app to start up
    time.sleep(1)

    # Make a request to the endpoint
    response = requests.get('http://127.0.0.1:5000/')
    print(response.status_code)

    # Assert that the response is what we expect
    assert response.status_code == 200
    assert type(response.text) == str

    # Make a post request to the endpoint score
    json_reply = requests.post('http://127.0.0.1:5000/score', {"sent": nonspam})

    # Assert that the response is what we expect
    assert json_reply.status_code == 200
    assert type(json_reply.text) == str

    # Assert it is a json as we intended
    load_json = json.loads(json_reply.text)

    assert type(load_json["Sentence"]) == str
    assert load_json["Prediction"] == "Spam" or load_json["Prediction"] == "Not Spam"

    prop_score = float(load_json["Propensity"])
    assert prop_score >= 0 and prop_score <= 1

    # Shut down the Flask app using os.system
    os.system('kill $(lsof -t -i:5000)')

def test_docker():
    # Build and run the Docker container
    os.system('docker build --network=host -t spam_class .')

    # Run Docker Container (and the app with it)
    os.system('docker run --shm-size=1G -p 5000:5000 --name spam-flask-app -it -d spam_class')

    time.sleep(10)
    # Run Test Flask again
    # Make a get request to the endpoint
    response = requests.get('http://127.0.0.1:5000/')
    print(response.status_code)

    # Assert that the response is what we expect
    assert response.status_code == 200
    assert type(response.text) == str

    # Make a post request to the endpoint score
    json_response = requests.post('http://127.0.0.1:5000/score', {"sent": nonspam})

    # Assert that the response is what we expect
    assert json_response.status_code == 200
    assert type(json_response.text) == str

    # Asserting to check whether its the intended json

    load_json = json.loads(json_response.text)

    assert type(load_json["Sentence"]) == str

    assert load_json["Prediction"] == "Spam" or load_json["Prediction"] == "Not Spam"

    prop_score = float(load_json["Propensity"])
    assert prop_score >= 0 and prop_score <= 1