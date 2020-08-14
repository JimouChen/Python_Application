from django.db import models


# Create your models here.
class User(models.Model):
    username = models.CharField(max_length=20)
    password = models.CharField(max_length=128)

    class Meta:
        db_table = 'user'  # 指定表名
