import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning
import warnings
import joblib

# Load data
data = pd.read_csv('AI-Data.csv')  # Update path to your dataset

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if column != 'Class':  # Assuming 'Class' is the target variable
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Define features and target
X = data.drop('Class', axis=1)
y = data['Class'].apply(lambda x: {'L': 0, 'M': 1, 'H': 2}[x])  # Encoding class labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Train and save models
models = {
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=2000),  # Increased max_iter
    "MLPClassifier": MLPClassifier(max_iter=2000)  # Increased max_iter
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f'{model_name.lower()}_classifier.joblib')

# Save the label encoders
joblib.dump(label_encoders, 'label_encoders.joblib')
