from django.contrib import admin
from django.urls import path
from myapp import views 
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path ('', views.index, name='index'),
    path ('predictimage',views.predictimage,name='predictimage'),
    path ('final/<userid>/', views.final, name='final'),
]
