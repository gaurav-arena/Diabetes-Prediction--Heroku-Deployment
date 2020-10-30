#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[3]:


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# In[ ]:


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [[x for x in request.form.values()]]
    final_features = np.asarray(features)
    prediction = model.predict(final_features)

    if(prediction[0]==1):
             output='Has Diabetes'
    else:
        output='Does not have diabetes'
                

        
    

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

