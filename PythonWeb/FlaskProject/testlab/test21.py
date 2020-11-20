from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
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
            'id': user.id
        })
    else:
        abort(400)


if __name__ == '__main__':
    app.run()
