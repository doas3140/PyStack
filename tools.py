'''
	Assorted tools.
'''

class Tools():
	def __init__(self):
		self.C = {}
		self.max_choose = 55
		self._init_choose()


	def _init_choose(self):
		for i in range(0, self.max_choose+1):
			for j in range(0, self.max_choose+1):
				self.C[i*self.max_choose + j] = 0

		for i in range(0,self.max_choose+1):
			self.C[i*self.max_choose] = 1
			self.C[i*self.max_choose + i] = 1

		for i in range(1,self.max_choose+1):
			for j in range(1,i+1):
				self.C[i*self.max_choose + j] = self.C[(i-1)*self.max_choose + j-1] + self.C[(i-1)*self.max_choose + j]


	def choose(self, n, k):
		return self.C[n*self.max_choose + k]


	def max_number(self):
		return 999999




tools = Tools()
