import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.exceptions import NotFittedError
import joblib

# Define the column names
columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

# Load the Cleveland heart disease dataset
df = pd.read_csv(
    "processed.cleveland.data",
    names=columns,
    na_values="?"
)

# Display the first five records
print(df.head())

# Display dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Display heart disease distribution
print("\nHeart Disease Distribution:")
print(df["num"].value_counts().sort_index())

# Convert the target into binary classification
# 0 = No heart disease
# 1 = Heart disease

df["target"] = (df["num"] > 0).astype(int)

# Display the new class distribution
print("\nBinary Heart Disease Distribution:")
print(df["target"].value_counts())

# Separate input features (X) from the target (y)
X = df.drop(columns=["num", "target"])
y = df["target"]

# Display the feature and target information
print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFirst Feature Record:")
print(X.iloc[0])

print("\nFirst Target:")
print(y.iloc[0])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the dataset split
print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

# Identify categorical and numerical features
categorical_features = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

numerical_features = [
    "age", "trestbps", "chol",
    "thalach", "oldpeak"
]

# Preprocess numerical features
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocess categorical features
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both preprocessing pipelines
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Create the machine learning pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# Save the trained model
joblib.dump(model, "heart_disease_model.pkl")

print("Model saved successfully!")
# Make predictions using the test data
y_pred = model.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# Display precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display the confusion matrix
# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Visualize the confusion matrix
plt.figure(figsize=(7, 5))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Heart Disease Prediction - Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["No Disease", "Heart Disease"])
plt.yticks([0, 1], ["No Disease", "Heart Disease"])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# Display values inside the graph
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# Visualize heart disease distribution
disease_counts = df["target"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(
    ["No Heart Disease", "Heart Disease"],
    disease_counts.values
)

plt.title("Heart Disease Distribution")
plt.xlabel("Patient Category")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("heart_disease_distribution.png", dpi=300)
plt.show()

# Perform 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print("\nCross-validation Scores:")
print(cv_scores)

print("\nMean Cross-validation Accuracy:")
print(f"{cv_scores.mean() * 100:.2f}%")

# Predict heart disease for a new patient

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

print("\nPrediction Probabilities:")
print(f"No Heart Disease: {probabilities[0][0] * 100:.2f}%")
print(f"Heart Disease: {probabilities[0][1] * 100:.2f}%")