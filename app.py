from flask import Flask, render_template, request
import pandas as pd
import joblib
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import joblib

app = Flask(__name__)

classifier_lr_off_from_joblib = joblib.load('classifier_lr_off.pkl')
classifier_lr_hate_from_joblib = joblib.load('classifier_lr_hate.pkl')
@app.route('/')
def mainf():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data=request.form['rtext']
    df=pd.read_csv("labeled_data.csv")
    df['tweet'][0]=data
    import re
    corpus = []
    for i in range(0, 24783):
        review = re.sub('[^a-zA-Z]', ' ', df['tweet'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    cv = CountVectorizer(max_features = 2000)
    X = cv.fit_transform(corpus).toarray()
    pred = classifier_lr_off_from_joblib.predict(X[:1])[0]+classifier_lr_hate_from_joblib.predict(X[:1])[0]
    return render_template('after.html',data=pred)

if __name__ == "__main__":
    app.run(debug=True)
