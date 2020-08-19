from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from App.models import User


def home(request):
    return render(request, 'app01/home.html')


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
    # user_dict = {'username': 'Lay', 'password': '123'}
    # User.objects.create(**user_dict)

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

    # # 指定查询
    # user = User.objects.values('username')
    # print(user)

    return HttpResponse('update ok!')


# 查询
def find_data(request):
    # 查询所有的
    users = User.objects.all()
    # 条件查询，用filter,里面写查询条件
    u = User.objects.filter(uid=2)
    # 如果是uid>=2,
    u = User.objects.filter(uid__gt=2)
    # 如果是2<=uid<=8
    u = User.objects.filter(uid__gt=2).filter(uid__lt=8)

    '''像all和filter返回的都是QuerySet，可以遍历的集合'''

    return render(request, 'search_list.html', locals())


def raw_sql(request):
    username = ''
    sql = 'select * from user where uid >= 2'
    user = User.objects.raw(sql)
    print(list(user))

    return HttpResponse('raw sql update ok!')


def my_manager(request):
    return HttpResponse('自定义管理器')


def handle_register(request):
    m = request.method
    if m == 'GET':
        return render(request, 'app01/register.html')
    else:
        name = request.POST.get('username', '')
        psw = request.POST.get('password', '')

        if name and psw:
            user = User(username=name, password=psw)
            user.save()

            return HttpResponse('注册成功')
        else:
            return HttpResponse('数据不能为空，注册失败')


def handle_login(request):
    if request.method == 'GET':
        return render(request, 'app01/login.html')
    else:
        # 获取表单数据
        name = request.POST.get('username', '')
        psw = request.POST.get('password', '')

        # 判断用户名和密码在不在数据库里面
        c = User.objects.filter(username=name, password=psw).count()
        if c == 1:
            return HttpResponse('登陆成功')
        # 其他情况
        return HttpResponse('登录失败')


def show_msg(request):
    users = User.objects.all()

    return render(request, 'app01/show.html', locals())
