import os
import mlflow
import sklearn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

os.chdir('./Assignment_2/')
train=pd.read_csv(r"./Data/Training Data.csv")
val=pd.read_csv(r"./Data/Validation Data.csv")
test=pd.read_csv(r"./Data/Test Data.csv")
os.chdir('../')

#splitting the datframe into X and y
y_train,X_train=train["Label"],train["Text"]
y_val,X_val=val["Label"],val["Text"]
y_test,X_test=test["Label"],test["Text"]

# replacing NAN entries by empty string
X_train = X_train.replace(np.nan, '', regex=True)
X_val = X_val.replace(np.nan, '', regex=True)
X_test = X_test.replace(np.nan, '', regex=True)

tfidf = TfidfVectorizer()
train_tfidf = tfidf.fit_transform(X_train)

def score(text:str, model, threshold:float) -> tuple:
    propensity = model.predict_proba(tfidf.transform([text]))[0]
    desired_predict = (model.predict_proba(tfidf.transform([text]))[:,1] >= threshold).astype(bool)
    return (bool(desired_predict[0]), float(propensity[1]))