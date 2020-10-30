import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Pregnancies': 3, 'Glucose':78, 'BloodPressure':50, 'Insulin':88, 'BMI' :31, 'DiabetesPedigreeFunction': 0.248, 'Age': 26 ,'SkinThickness': 32})

print(r.json())