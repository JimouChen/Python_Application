from django.contrib import admin
from django.urls import path
from App import views

app_name = 'App'
urlpatterns = [
    path('', views.home, name='home'),
    path('/test', views.test, name='test'),
    path('/show', views.show_msg, name='show'),
    path('/handle_login', views.handle_login, name='login'),
]
