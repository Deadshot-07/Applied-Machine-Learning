import os
import pickle, json
import score
from flask import Flask, request, render_template

# Local Imports
os.chdir(os.path.abspath(os.path.dirname(__file__)))

app = Flask(__name__, template_folder = './template')

# Importing Model
model_path = "./models/mlp_model.sav"
loaded_model = pickle.load(open(model_path, "rb"))

# Setting threshold value
threshold=0.5

@app.route('/') 
def home():
    return render_template('spam_prediction.html')

@app.route('/score', methods=['POST'])
def spam():
    sent = request.form['sent']
    prediction,propensity = score.score(sent,loaded_model,threshold)
    label="Spam" if prediction == True else "Not Spam"
    
    dictToReturn = {'Sentence':sent,'Prediction':label, 'Propensity':propensity}
    json_obj = json.dumps(dictToReturn, indent = 4) 
    
    return json_obj

if __name__ == '__main__': 
    app.run(host="0.0.0.0", port=5000, debug=True)