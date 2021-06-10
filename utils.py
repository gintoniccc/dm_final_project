import torch
import numpy as np
import os
import time
from argparse import ArgumentParser
from sklearn.metrics import f1_score,roc_auc_score
def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)



def evaluate_perform(y_pred,y_true):
	f1 = f1_score(y_true, y_pred)
	auroc = roc_auc_score(y_true, y_pred)
	print(f"f1:{f1} | auroc:{auroc}")
	return f1,auroc



class Timer(object):
	""" A quick tic-toc timer
	Credit: http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
	"""

	def __init__(self, name=None, verbose=True):
		self.name = name
		self.verbose = verbose
		self.elapsed = None

	def __enter__(self):
		self.tstart = time.time()
		return self

	def __exit__(self, type, value, traceback):
		self.elapsed = time.time() - self.tstart
		if self.verbose:
			if self.name:
				print('[%s]' % self.name,)
			print('Elapsed: %s' % self.elapsed)