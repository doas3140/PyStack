'''
	Implements depth-limited re-solving at a node of the game tree.
	Internally uses @{cfrd_gadget|CFRDGadget} TODO SOLVER
'''

from ..Lookahead.lookahead import Lookahead
from ..Lookahead.cfrd_gadget import CFRDGadget
from ..Tree.tree_builder import PokerTreeBuilder
# from ..Tree.tree_visualiser import TreeVisualiser # dont need here?
from ..Settings.arguments import arguments
from ..Settings.constants import constants
from ..tools import tools
from ..Game.card_tools import card_tools

class Resolving():
	def __init__(self):
		self.tree_builder = PokerTreeBuilder()


	def _create_lookahead_tree(self, node):
		''' Builds a depth-limited public tree rooted at a given game node.
		@param: node the root of the tree
		'''
		pass


	def resolve_first_node(self, node, player_range, opponent_range):
		''' Re-solves a depth-limited lookahead using input ranges.
			Uses the input range for the opponent instead of a gadget range,
			so only appropriate for re-solving the root node of the game tree
			(where ranges are fixed).
		@param: node the public node at which to re-solve
		@param: player_range a range vector for the re-solving player
		@param: opponent_range a range vector for the opponent
		'''
		pass


	def resolve(self, node, player_range, opponent_cfvs):
		''' Re-solves a depth-limited lookahead using an input range for the player
			and the @{cfrd_gadget|CFRDGadget} to generate ranges for the opponent.
			@param: node the public node at which to re-solve
			@param: player_range a range vector for the re-solving player
			@param: opponent_cfvs a vector of cfvs achieved by the opponent
					before re-solving
		'''
		pass


	def _action_to_action_id(self, action):
		''' Gives the index of the given action at the node being re-solved.
			The node must first be re-solved with @{resolve} or @{resolve_first_node}.
		@param: action a legal action at the node
		@return the index of the action
		'''
		pass


	def get_possible_actions(self):
		''' Gives a list of possible actions at the node being re-solved.
 			The node must first be re-solved with @{resolve} or @{resolve_first_node}.
		@return a list of legal actions
		'''
		pass


	def get_root_cfv(self):
		''' Gives the average counterfactual values that the re-solve player
			received at the node during re-solving.
			The node must first be re-solved with @{resolve_first_node}.
		@return a vector of cfvs
		'''
		pass


	def get_root_cfv_both_players(self):
		''' Gives the average counterfactual values that each player received
			at the node during re-solving.
			Usefull for data generation for neural net training
			The node must first be re-solved with @{resolve_first_node}.
		@return a (2,K) tensor of cfvs, where K is the range size
		'''
		pass


	def get_action_cfv(self, action):
		''' Gives the average counterfactual values that the opponent received
			during re-solving after the re-solve player took a given action.
			Used during continual re-solving to track opponent cfvs. The node must
			first be re-solved with @{resolve} or @{resolve_first_node}.
		@param: action the action taken by the re-solve player
				at the node being re-solved
		@return a vector of cfvs
		'''
		pass


	def get_chance_action_cfv(self, action, board):
		''' Gives the average counterfactual values that the opponent received
			during re-solving after a chance event (the betting round changes and
			more cards are dealt).
			Used during continual re-solving to track opponent cfvs.
			The node must first be re-solved with @{resolve} or @{resolve_first_node}.
		@param: action the action taken by the re-solve player
				at the node being re-solved
		@param: board a vector of board cards
				which were updated by the chance event
		@return a vector of cfvs
		'''
		pass


	def get_action_strategy(self, action):
		''' Gives the probability that the re-solved strategy takes a given action.
			The node must first be re-solved with @{resolve} or @{resolve_first_node}.
		@param action a legal action at the re-solve node
		@return a vector giving the probability of taking the action
				with each private hand
		'''
		pass




#
