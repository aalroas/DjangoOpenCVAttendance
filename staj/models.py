from django.db import models

# Create your models here.
 
class Course(models.Model):
    id = models.CharField(primary_key='True', max_length=50)
    name = models.CharField(max_length=50)
    date = models.DateField(blank=True)
    time = models.TimeField(blank=True)
    def __str__(self):
         return self.name
# sorgulama = selcet id from courses  where date =  "biglisayardaki_tarih" and time = "biilgisayrdki_simmdi_timme"
# sorgulama loop ogreneri  liste scrip facce ,ogrenci cam onunndde gelirse - o ogrenci listesi bakacak eger tandik ise 
#  course id , student id  , date ve tarih, 1 sataus olarak boyle sql db e yazduracak.
 
class Student(models.Model):
    id = models.CharField(primary_key='True', max_length=100)
    name = models.CharField(max_length=200)
    def user_directory_path(instance, filename):
        return 'students/{0}/{1}'.format(instance.id,filename)
    image = models.ImageField(upload_to=user_directory_path)
    def __str__(self):
         return self.name

class Attendance(models.Model):
    course_id = models.ForeignKey(Course, on_delete=models.CASCADE)
    student_id = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField(blank=True)
    time = models.TimeField(blank=True)
    status = models.BooleanField(default='False')

