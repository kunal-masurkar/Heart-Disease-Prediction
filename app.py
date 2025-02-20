from flask import Flask, request,render_template
import numpy as np
from DecisionTree import *
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    if(request.method == "POST"):
        data = pd.read_csv("./heart/heart.csv")
        X, y = data.values[:, 0:13], data.values[:, -1]
        clf = DecisionTree()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf.fit(X_train, y_train)
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = clf.predict(features)
        result = "Negative" if prediction == 0 else "Positive"
        return render_template("index.html", prediction_text = result)

if __name__ == "__main__":
    app.run(debug=True)