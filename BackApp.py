from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import xgboost as xgb2

app = Flask(__name__)
model2 = xgb2.XGBClassifier()
model2.load_model("modelSaving.json")

@app.route('/')
def home():
    file = open("savingRes", "r")
    fileInfo = file.read()
    fileInfo = fileInfo.split("\n")
    file.close()
    return render_template('home.html', currMatchesInfo = fileInfo)

@app.route('/admin')
def admin():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    allData = request.form["matchToPredicting"]
    allData = allData.split(' \t')
    prData = []
    for el in allData:
        prData.append(float(el))
    df = pd.DataFrame([prData], columns=model2.feature_names_in_)
    predClass = model2.predict(df)
    predProbClass = model2.predict_proba(df)
    predProbClassRound = round(predProbClass.max(), 2)
    df = df[['second_id', 'first_id', 'tourney_year', 'tourney_month']]
    # Appending to file
    plId = []
    with open("PlayerID.csv", "r") as filePlId:
        temp = filePlId.read().split("\n")
        for el in temp:
            el1, el2 = el.split(',')
            plId.append([float(el1),el2])
        #print(plId)
    Pl1Name = df.iat[0, 0]
    Pl2Name = df.iat[0, 1]
    for x in plId:
        if x[0] == Pl2Name:
            Pl2Name = x[1]
        if Pl1Name == x[0]:
            Pl1Name = x[1]
    with open("savingRes", 'a') as file1:
        file1.write(f"\nPlayer{int(predClass[0])+1} WIN. Probability {int(predProbClass.max()*100)}%. {Pl1Name} vs {Pl2Name}. Date: Y{int(df.iat[0, 2])}M{int(df.iat[0, 3])+1}")

    return redirect('/')

if __name__ == "__main__":
    app.run()