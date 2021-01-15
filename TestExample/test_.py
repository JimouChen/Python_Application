def get_new_board(old_board: str, move: str):
    for i in range(len(old_board) - 1):
        if old_board[i:i + 2] == move[0:2] and i % 2 == 0:
            return old_board[:i] + move[2:] + old_board[i + 2:]


s = '8979695949392919097717866646260600102030405060708012720323436383'
print(get_new_board(s, '7747'))
