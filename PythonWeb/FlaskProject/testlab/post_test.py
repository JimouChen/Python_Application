"""
# @Time    :  2020/11/19
# @Author  :  Jimou Chen
"""
from flask import Flask, jsonify, request, abort

app = Flask('test_app')


@app.route("/", methods=('POST',))
def user_create():
    name = request.form.get('name')
    if name:
        return jsonify({
            'hello': name
        })
    else:
        abort(400)


if __name__ == '__main__':
    app.run()
