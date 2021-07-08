from django.contrib import admin
 
from .models import Student, Course, Attendance
from django.utils.html import format_html
# Register your models here.
 
class StudentAdmin(admin.ModelAdmin):
    list_display = ('image_tag','id', 'name')
    search_fields = ('name', 'id')
    ordering = ['id']
    def image_tag(self,obj):
        return format_html('<img src="{0}" style="width: 45px; height:45px;" />'.format(obj.image.url))

class CourseAdmin(admin.ModelAdmin):
    list_display = ('id', 'name','time','date')
    search_fields = ('id', 'name')
    ordering = ['id']

class AttendanceAdmin(admin.ModelAdmin):
    list_display = ('student','course','status','date','time')
    list_filter = ('status',)
    def student(self, obj):
        if obj.student_id_id:
           return Student.objects.get(id=obj.student_id_id).name
    def course(self, obj):
        if obj.course_id_id:
           return Course.objects.get(id=obj.course_id_id).name

admin.site.register(Student, StudentAdmin)
admin.site.register(Course, CourseAdmin)
admin.site.register(Attendance, AttendanceAdmin)
 