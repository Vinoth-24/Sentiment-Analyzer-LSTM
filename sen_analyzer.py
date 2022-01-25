# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import re
from functions import *
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import numpy as np



# Load the Lstm model object from disk
model = load_model('lstm_model.h5')

def processing(review):
    voc_size = 10000
    final_review = normalize_and_lemmaize(review) # Cleaned text to predict
    print(final_review)
    onehot_=[one_hot(final_review,voc_size)] 
    sent_length=30
    embedded_docs=pad_sequences(onehot_,padding='pre',maxlen=sent_length)
    X_predi=np.array(embedded_docs)
    return X_predi

def prediction(X_predi):
    temp=model.predict(X_predi)
    Y_predi = (temp >= 0.040942967) #optimal threshold is 0.040942967
    Y_predi=1*Y_predi 
    return Y_predi
#     if Y_predi == 1:
#         result = "It is a good review" 
#     elif Y_predi == 0:
#         result ="It is a bad review"
#     return result
    



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	review = [message]
    	X_predi = processing(review)
    	my_prediction = prediction(X_predi)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
