from flask import Flask, render_template, request
from joblib import load
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

#data=pd.read_csv('IMDB Dataset.csv')
model=load("LinearSVC_grid.joblib")
vector=load("tfidvector.pkl")
#vector=TfidfVectorizer(stop_words='english',ngram_range=(1,2))
#X_train,X_test,y_train,y_test=train_test_split(data['review'],data['sentiment'],random_state=42)
#test=vector.fit_transform(X_train)

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict',methods=['POST'])
def predict():

    text=['blank']
    text.append(request.form['review'])
    tf=vector.transform(text)
    np.array(tf).reshape(1, -1)

    pred=model.predict(tf)
    if pred[1]==1:
        pred="positive"
    else:
        pred="negative"

    return render_template('form.html', prediction_text="The LinearSVC classifies this review as "+pred)


if __name__ == "__main__":
    
    app.run()
