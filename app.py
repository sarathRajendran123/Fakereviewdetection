from flask import Flask, render_template, request, redirect, url_for
import os
import pickle

app = Flask(__name__)

# Define the text_process function (if required by your model)
def text_process(text):
    import re
    from nltk.corpus import stopwords
    stopwords_set = set(stopwords.words('english'))
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stopwords_set]
    return ' '.join(text)

# Load the trained pipeline
model_path = os.path.join(os.path.dirname(__file__), 'pipeline_model.pkl')
try:
    with open(model_path, 'rb') as file:
        pipeline = pickle.load(file)
except Exception as e:
    print(f"Error loading the pipeline model: {e}")
    pipeline = None

# Define a function to predict if the review is fake or not
def predict_review(review_text):
    if pipeline:
        prediction = pipeline.predict([review_text])
        return prediction[0]
    else:
        return "Model not loaded"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review_text = request.form.get('review_text')
        if review_text:
            prediction = predict_review(review_text)
            return render_template('result.html', prediction=prediction)
        else:
            return redirect(url_for('home'))
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
