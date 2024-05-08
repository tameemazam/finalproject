from django.db import models

from django.db import models
from django.db import models

class Student(models.Model):
    gender = models.CharField(max_length=1)
    NationalITy = models.CharField(max_length=50)  # Updated field name to match dataset
    PlaceofBirth = models.CharField(max_length=50)  # Updated field name to match dataset
    StageID = models.CharField(max_length=50)  # Updated field name to match dataset
    GradeID = models.CharField(max_length=10)  # Updated field name to match dataset
    SectionID = models.CharField(max_length=10)  # Updated field name to match dataset
    Topic = models.CharField(max_length=50)
    Semester = models.CharField(max_length=1)  # Updated field name to match dataset
    Relation = models.CharField(max_length=50)
    raisedhands = models.IntegerField()  # Updated field name to match dataset
    VisITedResources = models.IntegerField()  # Updated field name to match dataset
    AnnouncementsView = models.IntegerField()  # Updated field name to match dataset
    Discussion = models.IntegerField()
    ParentAnsweringSurvey = models.CharField(max_length=3)  # Updated field name to match dataset
    ParentschoolSatisfaction = models.CharField(max_length=4)  # Updated field name to match dataset
    StudentAbsenceDays = models.CharField(max_length=10)  # Updated field name to match dataset
    Class = models.CharField(max_length=1)  # Updated field name to match dataset

    def __str__(self):
        return f"{self.gender} - {self.NationalITy} - {self.Class}"

"""
import csv
from frontend_ml.models import Student

def populate_students_from_csv(csv_file_path):
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            student = Student(
                gender=row['gender'],
                NationalITy=row['NationalITy'],
                PlaceofBirth=row['PlaceofBirth'],
                StageID=row['StageID'],
                GradeID=row['GradeID'],
                SectionID=row['SectionID'],
                Topic=row['Topic'],
                Semester=row['Semester'],
                Relation=row['Relation'],
                raisedhands=int(row['raisedhands']),
                VisITedResources=int(row['VisITedResources']),
                AnnouncementsView=int(row['AnnouncementsView']),
                Discussion=int(row['Discussion']),
                ParentAnsweringSurvey=row['ParentAnsweringSurvey'],
                ParentschoolSatisfaction=row['ParentschoolSatisfaction'],
                StudentAbsenceDays=row['StudentAbsenceDays'],
                Class=row['Class']
            )
            student.save()

# Specify the path to your CSV file
csv_file_path = 'AI-Data.csv'

# Call the function to populate the database
populate_students_from_csv(csv_file_path)

print("Data imported successfully!")

"""