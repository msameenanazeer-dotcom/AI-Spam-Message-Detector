from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

messages = [
    "Win a free prize now",
    "Congratulations you won money",
    "Claim your free gift",
    "Call me when you reach home",
    "Can you send the notes",
    "Let's meet tomorrow",
    "Your free vacation is waiting",
    "You have won a lottery"
]

labels = [
    "spam",
    "spam",
    "spam",
    "ham",
    "ham",
    "ham",
    "spam",
    "spam"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

model = MultinomialNB()
model.fit(X, labels)

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":
        text = request.form["message"]
        prediction = model.predict(vectorizer.transform([text]))[0]

        result = prediction.upper()

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
