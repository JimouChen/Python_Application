from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from App.models import User


def home(request):
    return HttpResponse('首页')


def handle_data(request):
    # 增加记录
    # user = User(username='Tom', password='123')
    # user.save()

    # user = User(username='Mary', password='123')
    # user.save()
    # user = User(username='Tee', password='1234')
    # user.save()
    # user = User(username='Peter', password='123')
    # user.save()

    # 另一种简便的方法
    user_dict = {'username': 'OP', 'password': '1333'}
    User.objects.create(**user_dict)

    # 修改,找到pk=1即uid=1的用户进行修改
    # user = User.objects.get(uid=1)  # 或者用pk=1
    # user.password = 3456
    # user.save()
    #
    # # 删除
    # try:
    #     user = User.objects.get(pk=4)  # 或者用uid=4
    #     if user:
    #         user.delete()
    # except Exception as e:
    #     print(e)

    return HttpResponse('update ok!')


def find_data(request):
    # 查询所有的
    users = User.objects.all()
    return render(request, 'search_list.html', locals())