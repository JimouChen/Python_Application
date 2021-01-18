from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect

# Create your views here.
from django.urls import reverse

from App.models import User


def home(request):
    # return HttpResponse('首页')
    # 查询数据库
    users = User.objects.all()
    show_content = {
        'title': 'Hello',
        'name': '世界',
        'users': users
    }
    return render(request, "index.html", context=show_content)


def get_phone(request):
    phone = '1232323'
    return HttpResponse(phone)


def get_name(request, name):
    return HttpResponse(name)


def handle_response(request):
    res = render(request, 'array.html')
    return res


def handle_redirect(request):
    # 重定向到home/下
    # return HttpResponseRedirect('/home/')
    # 一般用快捷方式，比较简短,redirect是HttpResponseRedirect的快捷方式
    # return redirect('/home/')

    # 也可以使用反向定位进行重定向
    return redirect(reverse('App:home'))

    # 可以重定向到其他网址
    # return redirect('https://www.baidu.com')
