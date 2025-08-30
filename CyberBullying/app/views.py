from django.shortcuts import render
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
with open('app/voting_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('app/vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

def predict(request):
    result = None
    text = None
    
    if request.method == 'POST':
        text = request.POST.get('text')
        if text:
            # Transform the input text
            X_new = vectorizer.transform([text])
            # Predict
            prediction = model.predict(X_new)
            result = 'Dangerous' if prediction[0] == 1 else 'Normal'
    
    return render(request, 'predict.html', {'text': text, 'result': result})
