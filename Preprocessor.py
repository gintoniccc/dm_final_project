import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
class Preprocessor(object):
	"""docstring for Preprocessor"""
	def __init__(self, args):
		super(Preprocessor, self).__init__()
		self.args = args
		self.chats,self.delet = self.preprocess()
		# Todo
		# Maybe can add labelencoder to add some feature from other column
		

	def load_file(self):
		chat_df = pd.read_csv(self.args.chat_df_path,na_values='', keep_default_na=False)
		#ban_df  = pd.read_csv(self.args.ban_df_path, usecols=['channelId', 'originVideoId'],na_values='', keep_default_na=False)

		delet = pd.read_csv(self.args.delete_df_path,
							usecols=['id', 'retracted'])

		delet = delet[delet['retracted'] == 0]

		delet['banned'] = True
		
		#ban_df['banned'] = 1
		return chat_df,delet

	def preprocess(self):
		chats,delet = self.load_file()
		chats = pd.merge(chats, delet[['id', 'banned']], how='left')
		chats['banned'].fillna(False, inplace=True)
		# chats = pd.merge(chats, delet, on=['channelId', 'originVideoId'], how='left')
		# chats['banned'].fillna(0, inplace=True)
		
		return chats,delet

	def sample(self,sample_rate = None):
		if not sample_rate:
			sample_rate = self.args.sample_rate

		banned_sample = self.chats[self.chats['banned']==True]
		normal_sample = self.chats[self.chats['banned']==False].sample((sample_rate*len(banned_sample)))
		sample_chats = normal_sample.append(banned_sample, ignore_index=True)

		dataset = Vtuber_Dataset(sample_chats)
		trainsize = int(self.args.trainsize_ratio*len(dataset))
		validsize = len(dataset) - trainsize
		trainset, valset = torch.utils.data.random_split(dataset, [trainsize, validsize])
		train_loader,valid_loader = DataLoader(trainset,self.args.batch_size,drop_last = True,shuffle = True),DataLoader(valset,self.args.batch_size)
		return train_loader,valid_loader


class Vtuber_Dataset(Dataset):
	"""docstring for Vtuber_Dataset"""
	def __init__(self,df):
		super(Vtuber_Dataset, self).__init__()
		self.df = df
	def __getitem__(self,idx):
		body = self.df.loc[idx,"body"]
		target = torch.tensor(self.df.loc[idx,"banned"],dtype = torch.long)
		return body,target

	def __len__(self):
		return len(self.df)
		