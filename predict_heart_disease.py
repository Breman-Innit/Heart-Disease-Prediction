import pandas as pd
import joblib

# Load the saved heart disease prediction model
model = joblib.load("heart_disease_model.pkl")

print("Heart disease prediction model loaded successfully!")

# Collect new patient information
print("\nEnter Patient Information:")

while True:
    age = float(input("Age: "))

    if 1 <= age <= 120:
        break
    else:
        print("Invalid age. Please enter an age between 1 and 120.")
while True:
    sex = float(input("Sex (1 = Male, 0 = Female): "))

    if sex in [0, 1]:
        break
    else:
        print("Invalid input. Enter 1 for Male or 0 for Female.")
while True:
    cp = float(input("Chest pain type (1-4): "))

    if cp in [1, 2, 3, 4]:
        break
    else:
        print("Invalid input. Enter a value between 1 and 4.")
while True:
    trestbps = float(input("Resting blood pressure: "))

    if trestbps > 0:
        break
    else:
        print("Invalid input. Blood pressure must be greater than 0.")
while True:
    chol = float(input("Cholesterol level: "))

    if chol > 0:
        break
    else:
        print("Invalid input. Cholesterol must be greater than 0.")
while True:
    fbs = float(input("Fasting blood sugar (1 = True, 0 = False): "))

    if fbs in [0, 1]:
        break
    else:
        print("Invalid input. Enter 1 or 0.")
while True:
    restecg = float(input("Resting ECG result (0-2): "))

    if restecg in [0, 1, 2]:
        break
    else:
        print("Invalid input. Enter 0, 1, or 2.")
while True:
    thalach = float(input("Maximum heart rate: "))

    if thalach > 0:
        break
    else:
        print("Invalid input. Maximum heart rate must be greater than 0.")
while True:
    exang = float(input("Exercise-induced angina (1 = Yes, 0 = No): "))

    if exang in [0, 1]:
        break
    else:
        print("Invalid input. Enter 1 for Yes or 0 for No.")
while True:
    oldpeak = float(input("Oldpeak: "))

    if oldpeak >= 0:
        break
    else:
        print("Invalid input. Oldpeak cannot be negative.")
while True:
    slope = float(input("ST segment slope (1-3): "))

    if slope in [1, 2, 3]:
        break
    else:
        print("Invalid input. Enter 1, 2, or 3.")
while True:
    ca = float(input("Number of major vessels (0-3): "))

    if ca in [0, 1, 2, 3]:
        break
    else:
        print("Invalid input. Enter 0, 1, 2, or 3.")
while True:
    thal = float(input("Thalassemia result (3, 6 or 7): "))

    if thal in [3, 6, 7]:
        break
    else:
        print("Invalid input. Enter 3, 6, or 7.")

new_patient = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])

# Make prediction
prediction = model.predict(new_patient)

# Calculate prediction probabilities
probabilities = model.predict_proba(new_patient)

print("\nNew Patient Prediction:")

if prediction[0] == 1:
    print("The model predicts heart disease.")
else:
    print("The model predicts no heart disease.")

# Display prediction probabilities
print("\nPrediction Probabilities:")
print(f"No Heart Disease: {probabilities[0][0] * 100:.2f}%")
print(f"Heart Disease: {probabilities[0][1] * 100:.2f}%")
