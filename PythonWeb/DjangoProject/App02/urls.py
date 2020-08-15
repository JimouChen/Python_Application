"""
# @Time    :  2020/8/15
# @Author  :  Jimou Chen
"""
from django.urls import path
from App02 import views

app_name = 'App02'  # 应用的名空间
# 路由列表，名称一定是urlpatterns,路由的名字是home
urlpatterns = [
    path('home/', views.home, name='home'),
    # 过滤器
    path('filter/', views.handle_filter, name='filter'),
    # 内置标签测试
    path('tag/', views.handle_tag, name='tag'),
    # 渲染登陆页面
    path('', views.login_view, name='login_view'),
    # get方式登录
    path('login/', views.handle_login, name='login')

]
