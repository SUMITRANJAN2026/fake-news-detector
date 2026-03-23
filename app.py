from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        news = request.form["news"]
        news_vector = vectorizer.transform([news])
        result = model.predict(news_vector)

        if result[0] == 1:
            prediction = "🟢 Real News ✅"
        else:
            prediction = "🔴 Fake News ❌"

    return render_template("index.html", prediction=prediction)

app.run(debug=True)