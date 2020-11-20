"""
# @Time    :  2020/11/20
# @Author  :  Jimou Chen
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from flask import Flask, jsonify, request, abort
from requests import Session

engine = create_engine('sqlite:///rest.db')
Base = declarative_base()
app = Flask('test_app')


class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    name = Column(String)


@app.route("/", methods=('POST',))
def user_create():
    name = request.form.get('name')
    if name:
        session = Session()
        user = User(name=name)
        session.add(user)
        session.commit()
        return jsonify({
            'id': user.id,
            'name': user.name
        })
    else:
        abort(400)


@app.route('/<string:name>/')
def hello(name: str = None):
    t = request.args.get('t')
    if t is not None and datetime.now().timestamp() - int(t) < 3600:
    # if t is not None:
        return jsonify([{
            'id': i.id,
            'name': i.name
        } for i in Session().query(User).filter(User.name == name)])
    else:
        abort(401)


if __name__ == '__main__':
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(name='Jack@{}'.format(datetime.now()))
    session.add(user)
    session.commit()

    for i in session.query(User).all():
        print(i.id, i.name)

    app.run()