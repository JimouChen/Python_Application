"""
# @Time    :  2020/11/19
# @Author  :  Jimou Chen
"""
from flask import Flask, jsonify

app = Flask('test_app')


@app.route('/')
def hello():
    return jsonify({
        'hello': 'world'
    })


@app.route('/<string:name>/')
def test(name: str = None):
    local_ver = []
    for i in range(len(name)):
        local_ver.append(i * '*+')

    return jsonify({
        'hello ya': name,
        'local': local_ver
    })


if __name__ == '__main__':
    app.run()
