from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ✅ Load your trained model and scaler using pickle
model = pickle.load(open("Online_trained_model.sav", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict')
def predict():
    return render_template("submit.html")

@app.route('/submit', methods=["POST"])
def submit():
    try:
        quarter = int(request.form['quarter'])
        department = int(request.form['department'])
        day = int(request.form['day'])
        targeted_productivity = float(request.form['targeted_productivity'])
        smv = float(request.form['smv'])
        wip = float(request.form['wip'])
        over_time = float(request.form['over_time'])
        incentive = float(request.form['incentive'])
        idle_time = float(request.form['idle_time'])
        idle_men = float(request.form['idle_men'])
        no_of_style_change = float(request.form['no_of_style_change'])
        no_of_workers = float(request.form['no_of_workers'])

        input_data = np.array([[quarter, department, day, targeted_productivity, smv, wip,
                                over_time, incentive, idle_time, idle_men,
                                no_of_style_change, no_of_workers]])

        # ✅ Scale input
        scaled_data = scaler.transform(input_data)

        # ✅ Predict
        prediction = model.predict(scaled_data)[0]
        prediction = round(prediction, 4)

        return render_template("submit.html", prediction=prediction)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
