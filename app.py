import chess
import chess.engine
import numpy as np
import tensorflow as tf
import os

stockfish_path = "/home/user/app/stockfish/stockfish-ubuntu-x86-64-avx512"
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

model_file = "chess_model.h5"
if os.path.exists(model_file):
    model = tf.keras.models.load_model(model_file)
    print("تم تحميل النموذج من الملف.")
else:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(64,), activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    print("تم إنشاء نموذج جديد.")

elo_file = "elo.txt"
if os.path.exists(elo_file):
    with open(elo_file, "r") as f:
        current_elo = int(f.read())
    print(f"تم تحميل Elo الحالي: {current_elo}")
else:
    current_elo = 2000  
    print(f"تم تعيين Elo الابتدائي إلى {current_elo}")

def update_elo(player_elo, opponent_elo, result):
    K = 32  
    expected_score = 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))
    return player_elo + K * (result - expected_score)

def board_to_input(board):
    board_fen = board.board_fen()
    board_array = np.array([
        1 if c == 'P' else
        -1 if c == 'p' else
        0 for c in board_fen.replace('/', '')
    ])
    if len(board_array) < 64:
        board_array = np.pad(board_array, (0, 64 - len(board_array)), mode='constant')
    return np.reshape(board_array, (1, 64))

def train_against_stockfish():
    global current_elo  
    stockfish_elo = 3500

    board = chess.Board()
    while not board.is_game_over():
        input_data = board_to_input(board)
        predictions = model.predict(input_data)
        move = np.argmax(predictions)  

        legal_moves = list(board.legal_moves)
        board.push(legal_moves[move % len(legal_moves)])

        result = engine.play(board, chess.engine.Limit(time=0.1))
        board.push(result.move)

    game_result = board.result()  
    if game_result == '1-0':  
        result = 1
    elif game_result == '0-1':  
        result = 0
    else:  
        result = 0.5

    current_elo = update_elo(current_elo, stockfish_elo, result)
    print(f"Elo الجديد: {current_elo}")

    with open(elo_file, "w") as f:
        f.write(str(current_elo))

    model.save(model_file)
    print(f"تم حفظ النموذج إلى {model_file} وElo إلى {elo_file}.")

    model.fit(input_data, np.array([result]), epochs=1)

game_number = 1
while True:
    print(f"المباراة رقم {game_number}")
    train_against_stockfish()
    game_number += 1
