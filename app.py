from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("Online_trained_model.sav", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/submit', methods=["POST"])
def submit():
    try:
        # Collect form data using .get and convert to proper type
        data = [
            int(request.form.get("quarter", 0)),
            int(request.form.get("department", 0)),
            int(request.form.get("day", 0)),
            float(request.form.get("targeted_productivity", 0)),
            float(request.form.get("smv", 0)),
            float(request.form.get("wip", 0)),
            float(request.form.get("over_time", 0)),
            float(request.form.get("incentive", 0)),
            float(request.form.get("idle_time", 0)),
            float(request.form.get("idle_men", 0)),
            float(request.form.get("no_of_style_change", 0)),
            float(request.form.get("no_of_workers", 0))
        ]

        # Prepare data for prediction
        input_data = np.array([data])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]

        return render_template("submit.html", prediction=round(prediction, 4))

    except Exception as e:
        return f"<h3 style='color:red'>⚠️ Error: {str(e)}</h3><br><a href='/'>Back to Home</a>"

if __name__ == '__main__':
    app.run(debug=True)
