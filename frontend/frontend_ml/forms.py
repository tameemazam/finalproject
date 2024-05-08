from django import forms

NATIONALITIES_CHOICES = [
    ('KW', 'Kuwaiti'),
    ('lebanon', 'Lebanon'),
    ('Egypt', 'Egyptian'),
    ('SaudiArabia', 'Saudi Arabian'),
    ('USA', 'American'),
    ('Jordan', 'Jordanian'),
    ('Venezuela', 'Venezuela'),
    ('Iran', 'Iranian'),
    ('Tunis', 'Tunisian'),
    ('Morocco', 'Moroccan'),
    ('Syria', 'Syrian'),
    ('Iraq', 'Iraqi'),
    ('Palestine', 'Palestinian'),
    ('Libya', 'Libyan'),
]

PLACES_OF_BIRTH_CHOICES = [
    ('KuwaIT', 'Kuwait'),
    ('lebanon', 'Lebanon'),
    ('Egypt', 'Egypt'),
    ('SaudiArabia', 'SaudiArabia'),
    ('USA', 'USA'),
    ('Jordan', 'Jordan'),
    ('venzuela', 'venzuela'),
    ('Iran', 'Iran'),
    ('Tunis', 'Tunisa'),
    ('Morocco', 'Morocco'),
    ('Syria', 'Syria'),
    ('Iraq', 'Iraq'),
    ('Palestine', 'Palestine'),
    ('Lybia', 'Lybia'),
]


STAGE_CHOICES = [
    ('lowerlevel', 'lowerlevel'),
    ('HighSchool', 'HighSchool'),
    ('MiddleSchool', 'MiddleSchool'),
]

GRADES_CHOICES = [
    ('G-02', 'G-02'),
    ('G-04', 'G-04'),
    ('G-05', 'G-05'),
    ('G-06', 'G-06'),
    ('G-07', 'G-07'),
    ('G-08', 'G-08'),
    ('G-09', 'G-09'),
    ('G-10', 'G-10'),
    ('G-11', 'G-11'),
    ('G-12', 'G-12'),
]

SECTION_CHOICES = [
    ('A', 'A'),
    ('B', 'B'),
    ('C', 'C'),
]

TOPICS_CHOICES = [
    ('Arabic', 'Arabic'),
    ('Biology', 'Biology'),
    ('Chemistry', 'Chemistry'),
    ('English', 'English'),
    ('French', 'French'),
    ('Geology', 'Geology'),
    ('History', 'History'),
    ('IT', 'IT'),
    ('Math', 'Math'),
    ('Quran', 'Quran'),
    ('Science', 'Science'),
    ('Spanish', 'Spanish'),
]


class StudentPredictionForm(forms.Form):
    gender = forms.ChoiceField(choices=[('M', 'Male'), ('F', 'Female')])
    NationalITy = forms.ChoiceField(choices=NATIONALITIES_CHOICES)
    PlaceofBirth = forms.ChoiceField(choices=PLACES_OF_BIRTH_CHOICES)
    Stage = forms.ChoiceField(choices=STAGE_CHOICES)
    Grade = forms.ChoiceField(choices=GRADES_CHOICES)
    Section = forms.ChoiceField(choices=SECTION_CHOICES)
    Topic = forms.ChoiceField(choices=TOPICS_CHOICES)
    Semester = forms.ChoiceField(choices=[('F', 'First'), ('S', 'Second')])
    Relation = forms.ChoiceField(choices=[('Father', 'Father'), ('Mum', 'Mother')])
    raisedhands = forms.IntegerField(min_value=0, max_value=100)
    VisITedResources = forms.IntegerField(min_value=0, max_value=100)
    AnnouncementsView = forms.IntegerField(min_value=0, max_value=100)
    Discussion = forms.IntegerField(min_value=0, max_value=100)
    ParentAnsweringSurvey = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    ParentschoolSatisfaction = forms.ChoiceField(choices=[('Good', 'Good'), ('Bad', 'Bad')])
    StudentAbsenceDays = forms.ChoiceField(choices=[('Under-7', 'Under 7'), ('Above-7', 'Above 7')])