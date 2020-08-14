from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
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
