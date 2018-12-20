
import os
import numpy as np

class Logger():
    def __init__(self, filepath):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'a') as f:
                f.write('player_hand:bot_hand, [player:action:amount, player:action:amount, ...], winner:"winner":won_pot')


    def start_round(self, player_hand, bot_hand):
        p1_hand_string = ''.join(player_hand)
        p2_hand_string = ''.join(bot_hand)
        with open(self.filepath, 'a') as f:
            f.write( '\n{}:{},'.format(p1_hand_string, p2_hand_string) )


    def append_action(self, player, action, amount):
        with open(self.filepath, 'a') as f:
            f.write( '{}:{}:{},'.format(player, action, amount) )


    def append_winner(self, player, won_pot):
        with open(self.filepath, 'a') as f:
            f.write( '{}:{}:{}'.format(player,"winner",won_pot) )


    def parse_line(self, line):
        l = line.split(',')
        hands_str, actions_str, winner_str = l[0], l[1:-1], l[-1]
        hands = hands_str.split(':')
        actions = [ a.split(':') for a in actions_str ]
        # check if the round has finished (last split middle word is 'winner')
        # it is possible that last split can be action (if game hasnt ended)
        if len(winner_str.split(':')) == 3:
            winner, check, won_pot = winner_str.split(':')
            if check == 'winner':
                round_ended = True
            else:
                winner, won_pot, round_ended = None, None, False
        else:
            winner, won_pot, round_ended = None, None, False
        return hands, actions, round_ended, winner, won_pot


    def get_avg_wins(self):
        lines = open(self.filepath, 'r').readlines()
        wins = []
        for line in lines[1:]:
            line = line[:-1] if line[:-1] == '\n' else line # remove \n
            _, _, round_ended, winner, won_pot = self.parse_line(line)
            if round_ended:
                multiplier = 1 if winner == 'player' else -1
                wins.append(multiplier*int(won_pot))
        print('STATS:', wins)
        return int( np.mean(wins) ) if wins != [] else 0




#
