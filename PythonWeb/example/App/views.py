from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from App.models import User, Movie


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


# 也可以不写
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
            # return HttpResponse('登陆成功')
            return render(request, 'app01/login_success.html')
        # 其他情况
        return HttpResponse('登录失败')


def show_msg(request):
    users = User.objects.all()

    return render(request, 'app01/show.html', locals())


# 展示并且有分页功能
def page(num, size=20):
    # 接收当前页码数
    num = int(num)
    # 总记录数
    total_pages = Movie.objects.count()
    # 最大页码
    max_num = total_pages // size + 1
    # 判断页码越界
    if num < 1:
        num = 1
    if num > max_num:
        num = max_num
    # 计算出每页显示的记录
    page_show = Movie.objects.all()[size * (num - 1): num * size]

    return page_show, num


# 原生分页
def show_movie(request):
    # 接收请求参数num,获取不到就取值1返回
    num = request.GET.get('num', 1)
    # 处理分页
    movie, n = page(num)
    # 上一页和下一页
    last_page = n - 1
    next_page = n + 1

    # movie = Movie.objects.all()
    return render(request, 'index01.html', locals())


# Django分页
def show_movie_page(request):
    num = int(request.GET.get('num', 1))
    movie = Movie.objects.all()
    # 创建分页器对象
    pager = Paginator(movie, 20)
    # 获取当前页的数据
    try:
        page_data = pager.page(num)
    except PageNotAnInteger:
        # 返回第一页
        page_data = pager.page(1)
    except EmptyPage:
        # 返回最后一页
        page_data = pager.page(pager.num_pages)

    '''实现翻页功能'''
    # 每页开始的页码,以每5页作为半个区间,展示10页，num作为当前页
    begin_page = num - 5

    if begin_page < 1:
        begin_page = 1

    # 每页结束的页码
    end_page = num + 5
    if end_page > pager.num_pages:
        end_page = pager.num_pages

    if end_page <= 10:
        begin_page = 1
        end_page = num + 9
    else:
        begin_page = end_page - 9

    page_list = range(begin_page, end_page + 1)

    return render(request, 'index02.html', locals())
