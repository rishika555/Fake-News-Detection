from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report

app = Flask(__name__)

# Assuming 'Fake.csv' and 'True.csv' are in the working directory
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

data_fake["class"] = 0
data_true["class"] = 1

# Filtering out the last 10 rows for manual testing (optional)
data_true_manual_testing = data_true.tail(10)
data_true = data_true.iloc[:-10]

data_fake_manual_testing = data_fake.tail(10)
data_fake = data_fake.iloc[:-10]

data_fake_manual_testing["class"] = 0
data_true_manual_testing["class"] = 1

data_merge = pd.concat([data_fake, data_true], axis=0)
data_merge.reset_index(drop=True, inplace=True)

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

data_merge['text'] = data_merge['text'].apply(wordopt)

x = data_merge['text']
y = data_merge['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

def train_model(model_type):
    if model_type == "LogisticRegression":
        model = LogisticRegression()
    elif model_type == "DecisionTreeClassifier":
        model = DecisionTreeClassifier()
    elif model_type == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(random_state=0)
    elif model_type == "RandomForestClassifier":
        model = RandomForestClassifier(random_state=0)
    else:
        raise ValueError("Invalid model type")

    model.fit(xv_train, y_train)  # Fit model with vectorized data
    return model

# Define models, but don't train them here
models = {"LogisticRegression": None,
          "DecisionTreeClassifier": None,
          "GradientBoostingClassifier": None,
          "RandomForestClassifier": None}

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"

def predict_news(news):
    predictions = {}
    testing_news = {"text": [news]}
    xv_news = vectorization.transform(testing_news['text'])  # Transform new data

    for model_name, model in models.items():
        prediction = model.predict(xv_news)[0]
        label = output_label(prediction)
        predictions[model_name] = label

    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    news = request.form['news']
    predictions = predict_news(news)
    return render_template('result.html', predictions=predictions)

if __name__ == '__main__':
    # Create and train all four models before running the Flask app
    for model_name in models:
        models[model_name] = train_model(model_name)
    app.run(debug=True, port=5000)
