"""
# @Time    :  2020/11/20
# @Author  :  Jimou Chen
"""
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from flask import Flask, jsonify, request, abort
from requests import Session
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///rest1.db')
Base = declarative_base()
app = Flask('test_app')


class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    password = Column(String)
    token = Column(String)


# 注册
@app.route("/register/", methods=('POST',))
def user_create():
    name = request.form.get('name')
    password = request.form.get('password')
    token = request.form.get('token')

    if name and password and token:
        session = Session()
        user = User(name=name, password=password, token=token)
        session.add(user)
        session.commit()
        return jsonify({
            'status': 201,
            'data': {
                'id': user.id,
                'name': user.name
            }
        })
    else:
        abort(401)


# 登录
@app.route('/login/', methods=('GET',))
def login():
    name = request.form.get('name')
    password = request.form.get('password')
    token = request.form.get('token')

    for i in Session().query(User):
        if name == i.name and password == i.password:
            # 更换token
            user = session.query(User).filter(User.name.like(name)).first()
            user.token = token
            session.commit()
            return jsonify({
                'status': 200,
                'data': {
                    'id': i.id,
                    'token': i.token
                }
            })


# 验证登录
@app.route("/judge/", methods=('GET',))
def judge():
    token = request.form.get('token')
    id_ = request.form.get('id')

    if token and id_:
        for i in Session().query(User):
            # 判断token和id是否一样
            if i.token == token and str(i.id) == id_:
                return jsonify({
                    'status': 200
                })

    # token或者id不对或者输入为空的情况登陆失败
    return jsonify({
        'status': 401
    })


# PUT登出
@app.route('/put/', methods=('PUT',))
def delete():
    token = request.form.get('token')
    name = request.form.get('name')

    if token and name:
        for i in Session().query(User):
            # 判断token和name是否一样,一样就把该token删除
            if i.token == token and i.name == name:
                user = session.query(User).filter(User.id.like(i.id)).first()
                # session.delete(user)
                user.token = ''
                session.commit()
                return jsonify({
                    'status': 204
                })


if __name__ == '__main__':
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    app.run()
