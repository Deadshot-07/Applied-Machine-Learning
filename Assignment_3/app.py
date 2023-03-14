from flask import Flask, request, render_template, url_for, redirect, jsonify
import score
import mlflow
import os

app = Flask(__name__)

os.chdir('./Assignment_2/')
loaded_model = mlflow.sklearn.load_model('models:/multilayer-perceptron-model/None')
os.chdir('../')

threshold=0.7

@app.route('/') 
def home():
    return render_template('spam_prediction.html')

@app.route('/spam', methods=['POST'])
def spam():
    sent = request.form['sent']
    prediction,propensity=score.score(sent,loaded_model,threshold)
    label="Spam" if prediction == True else "Not spam"
    ans1 = f"""The sentence : "{sent}" """
    ans2 = f"""Prediction : {label} """
    ans3 = f"""Propensity : {propensity}."""
    dictToReturn = {'Prediction':prediction, 'Propensity':propensity}
    # jsonify(dictToReturn)
    return render_template('result.html', ans1=ans1, ans2 = ans2, ans3=ans3)

if __name__ == '__main__':
    app.run(port=5000, debug=True)