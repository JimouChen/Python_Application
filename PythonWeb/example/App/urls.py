from django.contrib import admin
from django.urls import path

from App import views

app_name = 'App'
urlpatterns = [
    path('', views.home, name='home'),
    # 增删改
    path('cud/', views.handle_data, name='handle_data'),
    # 查询
    path('search/', views.find_data, name='search'),
    # 使用原生sql
    path('rawsql/', views.raw_sql, name='raw_sql'),
    # 自定义管理器,看自己需要用不用
    path('manager/', views.my_manager, name='my_manager'),
    # 注册页
    path('register/', views.handle_register, name='register'),
    # 登录页
    path('login/', views.handle_login, name='login'),
    # 显示用户信息
    path('show/', views.show_msg, name='show'),
]
