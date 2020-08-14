"""
# @Time    :  2020/8/14
# @Author  :  Jimou Chen
"""
from django.urls import path
from App import views

# 路由列表，名称一定是urlpatterns,路由的名字是home
urlpatterns = [
    path('home/', views.home, name='home')
]