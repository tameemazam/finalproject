from itertools import count
from django.shortcuts import render
import io
from django.http import HttpResponse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from .models import Student
from django.http import HttpResponse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from .models import Student
from django.db.models import Count
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from django.http import HttpResponse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Assuming data is loaded from a CSV or database into DataFrame
data = pd.read_csv('AI-Data.csv')

def marks_class_count_graph(request):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Class', data=data, order=['L', 'M', 'H'])
    plt.title('Marks Class Count Graph')
    return get_graph_response()

def marks_class_semester_graph(request):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Semester', hue='Class', data=data, hue_order=['L', 'M', 'H'])
    plt.title('Marks Class Semester-wise Graph')
    return get_graph_response()
def marks_class_nationality_graph(request):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='NationalITy', hue='Class', data=data, hue_order=['L', 'M', 'H'])
    plt.title('Marks Class Nationality-wise Graph')
    plt.xticks(rotation=45)  # Rotate labels to fit longer names
    return get_graph_response()

def marks_class_grade_graph(request):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='GradeID', hue='Class', data=data, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], hue_order=['L', 'M', 'H'])
    plt.title('Marks Class Grade-wise Graph')
    return get_graph_response()

def marks_class_section_graph(request):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='SectionID', hue='Class', data=data, hue_order=['L', 'M', 'H'])
    plt.title('Marks Class Section-wise Graph')
    return get_graph_response()

def marks_class_topic_graph(request):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Topic', hue='Class', data=data, hue_order=['L', 'M', 'H'])
    plt.title('Marks Class Topic-wise Graph')
    return get_graph_response()

def marks_class_stage_graph(request):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='StageID', hue='Class', data=data, hue_order=['L', 'M', 'H'])
    plt.title('Marks Class Stage-wise Graph')
    return get_graph_response()

def marks_class_absent_days_graph(request):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='StudentAbsenceDays', hue='Class', data=data, hue_order=['L', 'M', 'H'])
    plt.title('Marks Class Absent Days-wise Graph')
    return get_graph_response()

def marks_class_gender_graph(request):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='gender', hue='Class', data=data, order=['M', 'F'], hue_order=['L', 'M', 'H'])
    plt.title('Marks Class Gender-wise Graph')
    return get_graph_response()

# Add additional views for each graph requirement...

def get_graph_response():
    """Utility function to get the graph response."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    buf.close()
    return response

from django.http import HttpResponse

from django.shortcuts import render
from .models import Student
def home(request):
  students = Student.objects.all()[:5]  # Limit to first 10 students
  return render(request, 'frontend_ml/index.html', {'students': students})


import pandas as pd
from django.http import JsonResponse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from django.http import JsonResponse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from django.http import JsonResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from django.http import HttpResponse
from django.template import loader

# Load your data
data = pd.read_csv('AI-Data.csv')

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

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_and_evaluate_model(model, model_name):
    # Set up the pipeline for scaling and classification
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    # Generate the classification report as a dictionary
    report_dict = classification_report(y_test, predictions, output_dict=True)

    # Calculate and add accuracy to the report dictionary
    accuracy = accuracy_score(y_test, predictions)
    report_dict['accuracy'] = accuracy

    # Safely update 'f1-score' to 'f1_score' for all sub-dictionaries in the report
    for key, value in report_dict.items():
        if isinstance(value, dict) and 'f1-score' in value:
            value['f1_score'] = value.pop('f1-score')

    return report_dict

def render_html_response(context):
    template = loader.get_template('frontend_ml/model_performance_template.html')
    return HttpResponse(template.render(context))

def train_decision_tree(request):
    report = train_and_evaluate_model(DecisionTreeClassifier(), "Decision Tree")
    return render_html_response({"model_name": "Decision Tree", "report": report})

def train_random_forest(request):
    report = train_and_evaluate_model(RandomForestClassifier(), "Random Forest")
    return render_html_response({"model_name": "Random Forest", "report": report})

def train_perceptron(request):
    report = train_and_evaluate_model(Perceptron(), "Perceptron")
    return render_html_response({"model_name": "Perceptron", "report": report})

def train_logistic_regression(request):
    report = train_and_evaluate_model(LogisticRegression(), "Logistic Regression")
    return render_html_response({"model_name": "Logistic Regression", "report": report})

def train_mlp_classifier(request):
    report = train_and_evaluate_model(MLPClassifier(), "MLP Classifier")
    return render_html_response({"model_name": "MLP Classifier", "report": report})




from django.shortcuts import render
from .forms import StudentPredictionForm
from .utils import load_trained_model 

#model = load_trained_model('student_classifier.joblib')

from django.shortcuts import render
from .forms import StudentPredictionForm
from .utils import load_trained_model

def predict(request):
    if request.method == 'POST':
        form = StudentPredictionForm(request.POST)
        if form.is_valid():
            # Process the valid form data
            model = load_trained_model('student_classifier.joblib')
            data_for_prediction = [form.cleaned_data.get(field.name) for field in form]
            prediction = model.predict([data_for_prediction])[0]
            return render(request, 'result.html', {'prediction': prediction})
        else:
            # Log form errors to console or display them in the template
            print("Form errors:", form.errors)
    else:
        form = StudentPredictionForm()
    return render(request, 'frontend_ml/student_form.html', {'form': form})



from django.shortcuts import render
from .forms import StudentPredictionForm

from django.shortcuts import render, redirect
from .forms import StudentPredictionForm
from .utils import load_trained_model

from django.shortcuts import render
from .forms import StudentPredictionForm
import joblib
import os
from django.shortcuts import render
from .forms import StudentPredictionForm
import joblib
import os
from django.shortcuts import render
from .forms import StudentPredictionForm
import joblib
# Mapping dictionary for class labels
import os
import joblib
import pandas as pd
import os
import joblib

class_label_mapping = {0: 'L', 1: 'M', 2: 'H'}

def student_form(request):
    if request.method == 'POST':
        form = StudentPredictionForm(request.POST)
        if form.is_valid():
            current_directory = os.path.dirname(os.path.abspath(__file__))
            model_files = [f for f in os.listdir(current_directory) if f.endswith('.joblib')]
            predictions = {}

            # Load label encoders
            label_encoders_path = os.path.join(current_directory, 'label_encoders.joblib')
            label_encoders = joblib.load(label_encoders_path)

            # Load all models
            for model_file in model_files:
                if model_file != 'label_encoders.joblib':  # Skip label encoders file
                    model_name = model_file.split('_')[0].capitalize()  # Extract model name from file name
                    model_path = os.path.join(current_directory, model_file)
                    print("Loading model:", model_path)
                    model = joblib.load(model_path)
                    print("Model loaded successfully:", model)

                    data_for_prediction = [form.cleaned_data.get(field.name) for field in form]
                    encoded_data = []
                    for i, column in enumerate(X.columns):
                        if column in label_encoders:
                            encoded_data.append(label_encoders[column].transform([data_for_prediction[i]])[0])
                        else:
                            encoded_data.append(data_for_prediction[i])

                    # Make prediction
                    prediction = model.predict([encoded_data])[0]
                    if prediction in class_label_mapping:
                        predicted_class_label = class_label_mapping[prediction]
                    else:
                        predicted_class_label = "Unknown"
        
                    predictions[model_name] = predicted_class_label

            return render(request, 'frontend_ml/result.html', {'predictions': predictions})
        else:
            print("Form errors:", form.errors)
    else:
        form = StudentPredictionForm()
    return render(request, 'frontend_ml/student_form.html', {'form': form})
