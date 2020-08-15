"""
# @Time    :  2020/8/14
# @Author  :  Jimou Chen
"""
from django.urls import path
from App import views

# 路由列表，名称一定是urlpatterns,路由的名字是home
urlpatterns = [
    # path('home/', views.home, name='home')
    # 如果不加home/，那到浏览器也不用加home/
    path('', views.home, name='home'),
    path('phone/', views.get_phone, name='get_phone'),
    path('getName/<name>', views.get_name, name='get_name')
]