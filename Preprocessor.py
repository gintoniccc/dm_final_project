import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
class Preprocessor(object):
	"""docstring for Preprocessor"""
	def __init__(self, args):
		super(Preprocessor, self).__init__()
		self.args = args
		self.chats,self.ban = self.preprocess()
		# Todo
		# Maybe can add labelencoder to add some feature from other column
		

	def load_file(self):
		chat_df = pd.read_csv(self.args.chat_df_path,na_values='', keep_default_na=False)
		ban_df  = pd.read_csv(self.args.ban_df_path, usecols=['channelId', 'originVideoId'],na_values='', keep_default_na=False)
		ban_df['banned'] = 1
		return chat_df,ban_df

	def preprocess(self):
		chats,ban = self.load_file()
		chats = pd.merge(chats, ban, on=['channelId', 'originVideoId'], how='left')
		chats['banned'].fillna(0, inplace=True)
		
		return chats,ban

	def sample(self,tokenizer,sample_rate = None):
		if not sample_rate:
			sample_rate = self.args.sample_rate

		banned_sample = self.chats[self.chats['banned']==True]
		normal_sample = self.chats[self.chats['banned']==False].sample((sample_rate*len(banned_sample)))
		sample_chats = normal_sample.append(banned_sample, ignore_index=True)

		dataset = Vtuber_Dataset(sample_chats,tokenizer,self.args.max_length)
		trainsize = int(self.args.trainsize_ratio*len(dataset))
		validsize = len(dataset) - trainsize
		trainset, valset = torch.utils.data.random_split(dataset, [trainsize, validsize])
		train_loader,valid_loader = DataLoader(trainset,self.args.batch_size,drop_last = True),DataLoader(valset,self.args.batch_size)
		return train_loader,valid_loader


class Vtuber_Dataset(Dataset):
	"""docstring for Vtuber_Dataset"""
	def __init__(self,df,tokenizer,max_length = 512):
		super(Vtuber_Dataset, self).__init__()
		self.df = df
		self.tokenizer = tokenizer
		self.max_length = max_length
	def __getitem__(self,idx):
		body = self.df.loc[idx,"body"][:self.max_length]
		target = torch.tensor(self.df.loc[idx,"banned"],dtype = torch.long)
		res = self.tokenizer(body, return_tensors="pt",padding = "max_length")
		input_ids = res["input_ids"].squeeze(0)
		att_mask  = res["attention_mask"].squeeze(0)
		try:
			assert input_ids.shape[0] == self.max_length
		except:
			print(f"error found,input_ids_shape:{input_ids.shape[0]}")
			body = self.df.loc[idx+1,"body"]
			target = torch.tensor(self.df.loc[idx+1,"banned"])
			res = self.tokenizer(body, return_tensors="pt",padding = "max_length")
			input_ids = res["input_ids"].squeeze(0)
			att_mask  = res["attention_mask"].squeeze(0)
			
		return input_ids,att_mask,target


	def __len__(self):
		return len(self.df)
		