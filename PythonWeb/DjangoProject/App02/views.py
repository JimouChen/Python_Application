from datetime import datetime

from django.http import HttpResponse
from django.shortcuts import render


# Create your views here.
def home(request):
    # return HttpResponse('App2 首页')
    users = [
        {'username': 'Park'},
        {'username': 'Tom'},
        {'username': 'Mary'}]
    print(locals())
    # locals() 局部变量字典
    return render(request, 'app2/index.html', context=locals())


def handle_filter(request):
    time = datetime.now()
    return render(request, 'app2/filter.html', locals())


def handle_tag(request):
    nums = [1, 2, 3, 4, 5]
    return render(request, 'app2/tag.html', locals())


# 显示登录界面
def login_view(request):
    return render(request, 'app2/login.html')


# 用GET的请求方式登录
def handle_login(request):
    username = request.GET.get('username')
    password = request.GET.get('password')
    if username == 'Jack' and password == '666':
        return HttpResponse('登陆成功')
    return HttpResponse('登陆失败')


def handle_post_login(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    if username == 'Jack' and password == '666':
        return HttpResponse('登陆成功')
    return HttpResponse('登陆失败')


def login2_view(request):
    return render(request, 'app2/login2.html')


def handle_url(request):
    return render(request, 'app2/menu.html')