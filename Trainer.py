import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5
class Vtu_model(nn.Module):
	"""docstring for Vtu_model"""
	def __init__(self,backbone,args):
		super(Vtu_model, self).__init__()
		self.backbone = backbone
		for p in self.backbone.parameters():
			p.requires_grad = False
		self.cls = nn.Sequential(
			nn.Linear(768,64),
			nn.LeakyReLU(),
			nn.Linear(64,2)
			)
		self.args = args
	def forward(self,**inputs):
		pool_featrue = self.backbone(**inputs).pooler_output
		pred = self.cls(pool_featrue)
		#output = self.backbone(**inputs)
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

		self.optimizer = torch.optim.AdamW(self.model.cls.parameters(),lr = self.args.lr)
		self.critique = nn.CrossEntropyLoss(weight = torch.tensor([1,self.args.sample_rate],dtype = torch.float).to(device))#Here maybe can design class weight
		self.best_eval_loss = None
	def train(self):
		for epo in range(self.args.epoch_num):
			avg_loss = 0
			for body,label in (self.train_loader):
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
			#print(f"epo:{epo} | avg_loss:{avg_loss/len(self.train_loader)}")
			logging.info('epo:%s | avg_loss:%s', epo, avg_loss/len(self.train_loader))
			if epo % self.args.eval_step == 0:
				eval_loss = self.eval()
				#print(f"epo:{epo} | eval_loss:{eval_loss/len(self.valid_loader)}")
				logging.info('epo:%s | eval_loss:%s', epo, eval_loss/len(self.valid_loader))				
				if self.best_eval_loss == None or eval_loss < self.best_eval_loss:
					self.save()
					self.best_eval_loss = eval_loss
	def eval(self):
		eval_loss = 0
		with torch.no_grad():
			for body,label in (self.valid_loader):
				inputs = self.tokenizer(body,padding=True, truncation=True,max_length = self.args.max_length,return_tensors="pt",verbose=False)
				for k,v in inputs.items():
					inputs[k] = v.to(device)
				label = label.to(device)
				
				pred = self.model(**inputs)
				loss = self.critique(pred,label)
				eval_loss += loss.item()
		return eval_loss
	def test(self,test_loader,load_path = None, file_name = None):
		if load_path != None or file_name != None:
			self.load(load_path,file_name)
		else:
			self.load()

		true_y,pred_y = np.array([]),np.array([])
		with torch.no_grad():
			test_loss = 0
			for body,label in (test_loader):
				inputs = self.tokenizer(body,padding=True, truncation=True,max_length = self.args.max_length,return_tensors="pt",verbose=False)
				for k,v in inputs.items():
					inputs[k] = v.to(device)
				label = label.to(device)
				
				pred = self.model(**inputs)
			
				pred_logits = torch.argmax(pred,dim = 1).cpu().numpy()
				pred_y = np.concatenate((pred_y,pred_logits),axis = 0)
				true_y = np.concatenate((true_y,label.cpu().numpy()),axis = 0)
			return pred_y,true_y


	def save(self,save_path = None,file_name = None):
		if save_path == None:
			save_path = self.args.ckpt_path
			if not os.path.exists(self.args.ckpt_path):
				os.makedirs(self.args.ckpt_path)
		if file_name == None:
			file_name = f"Vtu_model+{self.args.model}+.pt"
		print('save model to', save_path+file_name)
		torch.save(self.model.state_dict(), save_path + file_name)
	def load(self,load_path = None,file_name = None):
		if load_path == None:
			load_path = self.args.ckpt_path
		if file_name == None:
			file_name = f"Vtu_model+{self.args.model}+.pt"
		print('load model from', load_path+file_name)
		self.model.load_state_dict(torch.load(
				load_path + file_name,map_location=lambda storage, loc: storage))
		self.model.eval()