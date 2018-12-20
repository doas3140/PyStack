
from flask_socketio import emit

class Client():
    def __init__(self):
        pass

    def change_cards(self, board, player_cards, bot_cards):
        emit('change_cards', {'player_cards':player_cards, 'board_cards':board, 'bot_cards':bot_cards})

    def change_chips(self, player_chips, bot_chips):
        emit('change_chips', {'player_chips':player_chips, 'opponent_chips':bot_chips})

    def it_is_players_turn(self):
        emit('players_turn')

    def notify_new_turn(self, player):
        emit('new_turn', {'player':player})

    def notify_winner(self, winner):
        emit('game_over', {'winner':winner})

    def change_stats(self, avg_wins):
        emit('change_stats', {'avg_wins':avg_wins})

    def show_error(self):
        emit('show_error')



client = Client()
