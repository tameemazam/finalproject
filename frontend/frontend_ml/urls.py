from django.urls import path
from . import views
from django.urls import path
from .views import predict, student_form

from django.urls import path
from .views import predict, student_form

urlpatterns = [
    path('', views.home, name='home'),
    path('graphs/class-count/', views.marks_class_count_graph, name='graph_class_count'),
    path('graphs/semester/', views.marks_class_semester_graph, name='graph_semester'),
    path('graphs/gender/', views.marks_class_gender_graph, name='graph_gender'),
    path('graphs/nationality/', views.marks_class_nationality_graph, name='graph_nationality'),
    path('graphs/grade/', views.marks_class_grade_graph, name='graph_grade'),
    path('graphs/section/', views.marks_class_section_graph, name='graph_section'),
    path('graphs/topic/', views.marks_class_topic_graph, name='graph_topic'),
    path('graphs/stage/', views.marks_class_stage_graph, name='graph_stage'),
    path('graphs/absent-days/', views.marks_class_absent_days_graph, name='graph_absent_days'),
    path('train/decision-tree/', views.train_decision_tree, name='train_decision_tree'),
    path('train/random-forest/', views.train_random_forest, name='train_random_forest'),
    path('train/perceptron/', views.train_perceptron, name='train_perceptron'),
    path('train/logistic-regression/', views.train_logistic_regression, name='train_logistic_regression'),
    path('train/mlp-classifier/', views.train_mlp_classifier, name='train_mlp_classifier'),
    path('predict/', predict, name='predict'),
    path('form/', student_form, name='student_form'),
]


