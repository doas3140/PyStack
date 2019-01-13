'''
	Caches first street/round outputs to diskself.
	Reads the disk, when same situation occurs.
'''
import os
import numpy as np
import pickle

class Cache():
	def __init__(self, dir_path):
		'''
		@param: str :path to caching directory
		'''
		self.dir_path = dir_path


	def get_element_dir_path(self, bets):
		'''
		@param: [2] :array of player bets
		@return str :path to current situation's directory
		'''
		return os.path.join(self.dir_path, '{}_{}'.format(int(bets[0]), int(bets[1])))


	def exists(self, bets):
		'''
		@param: [2]  :array of player bets
		@return bool :is this situation cached
		'''
		results_dir = self.get_element_dir_path(bets)
		return os.path.exists(results_dir)


	def get_resolve_results(self, bets):
		'''
		@param: [2]              :array of player bets
		@return LookaheadResults : results (without `next_street_cfvs`)
		'''
		results_dir = self.get_element_dir_path(bets)
		return pickle.load( open(os.path.join(results_dir, 'results.pkl'), 'rb') )


	def store_resolve_results(self, bets, results):
		'''
		@param: [2] :array of player bets
		@param: LookaheadResults :results to save
		'''
		results_dir = self.get_element_dir_path(bets)
		# create dir
		try:
			os.mkdir(results_dir)
		except FileExistsError:
			print('cache dir already exists, overriding previous results')
		# store everything in pickle file except next_street_cfvs store seperately
		np.save(os.path.join(results_dir, 'next_street_cfvs'), results.next_street_cfvs)
		results.next_street_cfvs = None
		pickle.dump(results, open(os.path.join(results_dir, 'results.pkl'), 'wb'))


	def get_next_street_cfvs(self, bets):
		'''
		@param: [2]       :array of player bets
		@return [b,B,P,I] :cfvs for all Boards for all Players (for each hand), (b=1)
		'''
		results_dir = self.get_element_dir_path(bets)
		return np.load(os.path.join(results_dir, 'next_street_cfvs.npy'))




#
