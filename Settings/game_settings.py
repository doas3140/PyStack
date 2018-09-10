'''
	Game constants which define the game played by DeepStack.
'''

class GameSettings():
    def __init__(self):
        # --- the number of card suits in the deck
        self.suit_count = 2
        # --- the number of card ranks in the deck
        self.rank_count = 3
        # --- the total number of cards in the deck
        self.card_count = self.suit_count * self.rank_count
        # --- the number of public cards dealt in the game (revealed after the first betting round)
        self.board_card_count = 1
        # --- the number of players in the game
        self.player_count = 2

game_settings = GameSettings()
