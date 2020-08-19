from django.db import models


# Create your models here.
from django.db.models import Manager


class User(models.Model):
    # 表的字段信息
    uid = models.AutoField(primary_key=True)  # 主键，且自增,默认整型
    username = models.CharField(max_length=30, unique=True)  # CharField必须指明长度, 设了unique
    password = models.CharField(max_length=20)
    register_time = models.DateTimeField(auto_now_add=True)

    # 用这种格式打印出来
    def __str__(self):
        return self.username + ':' + str(self.uid)

    '''表的内部消息'''

    class Meta:
        db_table = 'user'  # 表名
        ordering = ['username']  # 按名字排序

# 扩展管理器功能
class MyManager(Manager):
    pass