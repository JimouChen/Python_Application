# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey has `on_delete` set to the desired behavior.
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Car(models.Model):
    carid = models.AutoField(db_column='carId', primary_key=True)  # Field name made lowercase.
    name = models.CharField(max_length=45, blank=True, null=True)
    price = models.CharField(max_length=45, blank=True, null=True)
    color = models.CharField(max_length=20, blank=True, null=True)
    issold = models.IntegerField(db_column='isSold', blank=True, null=True)  # Field name made lowercase.
    comments = models.CharField(max_length=100, blank=True, null=True)
    istrue = models.IntegerField(db_column='isTrue', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'car'


class Cards(models.Model):
    id = models.ForeignKey('Users', models.DO_NOTHING, db_column='id', primary_key=True)
    holder = models.IntegerField(blank=True, null=True)
    money = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cards'


class Customer(models.Model):
    name = models.CharField(max_length=45, blank=True, null=True)
    email = models.CharField(max_length=45, blank=True, null=True)
    birth = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'customer'


class Member(models.Model):
    memberid = models.AutoField(db_column='memberId', primary_key=True)  # Field name made lowercase.
    name = models.CharField(max_length=45, blank=True, null=True)
    idcard = models.CharField(db_column='idCard', unique=True, max_length=45, blank=True, null=True)  # Field name made lowercase.
    phone = models.CharField(max_length=45, blank=True, null=True)
    credits = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'member'


class Notice(models.Model):
    noticeid = models.AutoField(db_column='noticeId', primary_key=True)  # Field name made lowercase.
    noticecontent = models.CharField(db_column='noticeContent', max_length=45, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'notice'


class Student(models.Model):
    stuid = models.AutoField(db_column='stuId', primary_key=True)  # Field name made lowercase.
    name = models.CharField(max_length=45, blank=True, null=True)
    idcard = models.CharField(db_column='idCard', unique=True, max_length=45, blank=True, null=True)  # Field name made lowercase.
    phone = models.CharField(max_length=45, blank=True, null=True)
    password = models.CharField(max_length=45, blank=True, null=True)
    schedule = models.CharField(max_length=45, blank=True, null=True)
    eventtime = models.CharField(db_column='eventTime', max_length=45, blank=True, null=True)  # Field name made lowercase.
    notice = models.CharField(max_length=45, blank=True, null=True)
    homework = models.CharField(max_length=45, blank=True, null=True)
    isfinished = models.CharField(db_column='isFinished', max_length=10, blank=True, null=True)  # Field name made lowercase.
    vianame = models.CharField(db_column='viaName', max_length=20, blank=True, null=True)  # Field name made lowercase.
    viaphone = models.CharField(db_column='viaPhone', max_length=45, blank=True, null=True)  # Field name made lowercase.
    spend = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    earn = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'student'


class User(models.Model):
    userid = models.AutoField(db_column='userId', primary_key=True)  # Field name made lowercase.
    name = models.CharField(max_length=45, blank=True, null=True)
    idcard = models.CharField(db_column='idCard', unique=True, max_length=20, blank=True, null=True)  # Field name made lowercase.
    phone = models.CharField(max_length=45, blank=True, null=True)
    right = models.IntegerField(blank=True, null=True)
    password = models.CharField(max_length=20, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'user'


class Users(models.Model):
    name = models.CharField(max_length=45, blank=True, null=True)
    password = models.CharField(max_length=45, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'users'
