from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from flask import Flask, jsonify, request, abort
from requests import Session
from sqlalchemy.orm import sessionmaker
import pymysql

engine = create_engine('mysql+pymysql://root:jmchen@127.0.0.1:3306/db1')
# 'mysql://scott:tiger@localhost/foo'
Base = declarative_base()
app = Flask('chess_app')


class Chess(Base):
    __tablename__ = 'word_count'
    id = Column(Integer, primary_key=True)
    board = Column(String)
    move = Column(String)
    num = Column(String)


# 得到下一个新的状态
def get_new_board(old_board: str, move: str):
    for i in range(len(old_board) - 1):
        if old_board[i:i + 2] == move[0:2] and i % 2 == 0:
            return old_board[:i] + move[2:] + old_board[i + 2:]


# move
@app.route("/move/", methods=('GET',))
def get_next_statue():
    board = request.args.get('board')
    print(board)
    chess = session.query(Chess).filter(Chess.board.like(board)).first()
    # print(get_new_board(board, chess.move))
    if chess:
        session.commit()

        session.commit()
        return jsonify({
            "new_board": get_new_board(board, chess.move),
            "black_move": chess.move,
        })
    else:
        return jsonify({
            "statue": 404
        })


if __name__ == '__main__':
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    app.run()
