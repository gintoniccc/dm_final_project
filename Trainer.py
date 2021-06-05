import os
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Vtu_model(nn.Module):
	"""docstring for Vtu_model"""
	def __init__(self,backbone,args):
		super(Vtu_model, self).__init__()
		self.backbone = backbone
		self.cls = nn.Sequential(
			nn.Linear(768,64),
			nn.LeakyReLU(),
			nn.Linear(64,2)
			)
		self.args = args
	def forward(self,**inputs):
		pool_featrue = self.backbone(**inputs)[1]
		pred = self.cls(pool_featrue)
		return pred


class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, backbone,tokenizer,train_loader,valid_loader, args):
		super(Trainer, self).__init__()
		self.model = Vtu_model(backbone,args).to(device)
		self.tokenizer = tokenizer
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.args = args

		self.optimizer = torch.optim.Adam(self.model.parameters(),lr = self.args.lr)
		self.critique = nn.CrossEntropyLoss()#Here maybe can design class weight

		self.best_eval_loss = None
	def train(self):
		for epo in range(self.args.epoch_num):
			avg_loss = 0
			for body,label in tqdm(self.train_loader):
				inputs = self.tokenizer(body,padding=True, truncation=True,max_length = self.args.max_length,return_tensors="pt",verbose=False)
				for k,v in inputs.items():
					inputs[k] = v.to(device)
				label = label.to(device)
				self.optimizer.zero_grad()
				pred = self.model(**inputs)
				loss = self.critique(pred,label)
				loss.backward()
				avg_loss += loss.item()
				self.optimizer.step()
			print(f"epo:{epo} | avg_loss:{avg_loss/len(self.train_loader)}")
			logging.info('epo:%s | avg_loss:%s', epo, avg_loss/len(self.train_loader))
			if epo % self.args.eval_step == 0:
				eval_loss = self.eval()
				print(f"epo:{epo} | eval_loss:{eval_loss/len(self.valid_loader)}")
				logging.info('epo:%s | avg_loss:%s', epo, eval_loss/len(self.valid_loader))				
				if self.best_eval_loss == None or eval_loss < self.best_eval_loss:
					self.save()
					self.best_eval_loss = eval_loss
	def eval(self):
		eval_loss = 0
		with torch.no_grad():
			for body,label in tqdm(self.valid_loader):
				inputs = self.tokenizer(body,padding=True, truncation=True,max_length = self.args.max_length,return_tensors="pt",verbose=False)
				for k,v in inputs.items():
					inputs[k] = v.to(device)
				label = label.to(device)
				pred = self.model(**inputs)
				loss = self.critique(pred,label)
				eval_loss += loss.item()
		return eval_loss

	def save(self,save_path = None,file_name = "Vtu_model.pt"):
		if save_path == None:
			save_path = self.args.ckpt_path
			if not os.path.exists(args.ckpt_path):
				os.makedirs(args.ckpt_path)
		print('save model to', save_path+file_name)
		torch.save(self.model.state_dict(), save_path + file_name)
	def load(self,load_path = None,file_name = "Vtu_model.pt"):
		if load_path == None:
			load_path = self.args.ckpt_path
		print('load model from', load_path+file_name)
		self.model.load_state_dict(torch.load(
				load_path + file_name), map_location=lambda storage, loc: storage)
		self.model.eval()