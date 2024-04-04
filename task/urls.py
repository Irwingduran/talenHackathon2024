from django.urls import path, include
from rest_framework import routers
from task import views

router = routers.DefaultRouter() 
router.register(r'task', views.TaskView, 'task' ) #or tasks 

urlpatterns = [
    path("api/v1/", include(router.urls)) ,
   
]

#ya no es necesario utilizar los métodos (GET, POST, PUT, DELETE) ya que lo genera por defecto 