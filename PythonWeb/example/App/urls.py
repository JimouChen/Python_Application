from django.contrib import admin
from django.urls import path

from App import views

app_name = 'App'
urlpatterns = [
    path('home/', views.home, name='home'),
    # 增删改
    path('', views.handle_data, name='handle_data'),
    # 查询
    path('search/', views.find_data, name='search'),
]
