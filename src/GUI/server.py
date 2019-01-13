
import time
import random
from flask import Flask, render_template
from flask_socketio import SocketIO

from GUI.poker import DoylesGame
from GUI.test_bot import TestBot
from GUI.logger import Logger
from Player.continual_resolving import ContinualResolving
from GUI.client import client as browser



# pystack = TestBot()
pystack = ContinualResolving()
logger = Logger('Data/logs.csv')
game = DoylesGame(bot=pystack, logger=logger)
GAME_IS_RUNNING = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@socketio.on('connect')
def test_connect():
    print('------USER CONNECTED------')


@socketio.on('start_game')
def start_game():
    print('------STARTING GAME------')
    avg_wins = logger.get_avg_wins()
    browser.change_stats(avg_wins=avg_wins)
    starting_player = 'player' if random.random() > 0.5 else 'bot'
    print('starting_player:', starting_player)
    time.sleep(2)
    game.start_round(starting_player)
    global GAME_IS_RUNNING
    if GAME_IS_RUNNING:
        for _ in range(100):
            print('ERROR: more then one user are trying to connect or multiple tabs opened!')
    GAME_IS_RUNNING = True
    return {'code':'success'}


@socketio.on('player_send_action')
def player_send_action(action, amount):
    print('---- PLAYER ACTION: {} {} ----'.format(action, amount))
    if game.current_player == 'player':
        success, action, amount = game.player_action(action, amount)
        return {'code':'success', 'action':action, 'amount':amount}
    else:
        return {'code':'not_your_turn'}

@socketio.on('player_received_end_game_msg')
def player_received_end_game_msg():
    print('------RESETING GAME------')
    global GAME_IS_RUNNING
    GAME_IS_RUNNING = False


def run_server():
    socketio.run(app, port=8000, log_output=False)
