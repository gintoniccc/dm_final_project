import os
import sys
import logging
import warnings
import pandas as pd
from transformers import XLMRobertaModel, XLMRobertaTokenizer,XLMRobertaForSequenceClassification
from transformers import BertTokenizer, BertModel
from Preprocessor import Preprocessor
from Trainer import Trainer
from utils import set_seed,Timer,evaluate_perform
from argparse import ArgumentParser

warnings.filterwarnings("ignore") 
SEED = 0
def main(args):
	set_seed(SEED)
	logging.basicConfig(filename=args.log_file, level=logging.INFO)
	logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
	if args.model.find("xlm") != -1:
		tokenizer = XLMRobertaTokenizer.from_pretrained(args.model)
		backbone = XLMRobertaModel.from_pretrained(args.model)
	else:
		tokenizer = BertTokenizer.from_pretrained(args.model)
		backbone = BertModel.from_pretrained(args.model)
	logging.info('start data preprocessing...')
	with Timer("parse data and sampling.."):
		P = Preprocessor(args)
		if  args.do_preprocess:
			P.preprocess()
			##Todo resample for several batch ?
			df = P.sample(args.sample_rate)
		else:
			try:
				df = pd.read_csv(args.sample_chat_path)
			except:
				P.preprocess()
				##Todo resample for several batch ?
				df = P.sample(args.sample_rate)
				
		train_loader,valid_loader,test_loader = P.create_dataloader(df) #may change to train,valid,test
	logging.info('finish data preprocessing')
	logging.info('loading model...')
	trainer = Trainer(backbone,tokenizer,train_loader,valid_loader, args)

	if not args.test:
		trainer.train()
		logging.info('finish training')
	else:
		logging.info("start testing")
		y_pred,y_true = trainer.test(test_loader)
		f1,auroc = evaluate_perform(y_pred,y_true)


	#Todo:
	#body_emb = trainer.generate_emb()
	#plot some result ,eg : tsne

	return

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--chat_df_path',default='./data/chats_2021-05.csv')
	parser.add_argument('--delete_df_path',default='./data/deletion_events.csv')
	parser.add_argument('--sample_chat_path',default="./data/sample_chat.csv")
	parser.add_argument('--model',default='bert-base-multilingual-uncased')
	parser.add_argument('--batch_size',default=32,type=int)
	parser.add_argument('--lr',default = 5*1e-5)
	parser.add_argument('--epoch_num',default=200,type=int)
	parser.add_argument('--sample_rate',default=5,type=int)
	parser.add_argument('--trainsize_ratio',default = 0.7)
	parser.add_argument('--testsize_ratio',default = 0.2)
	parser.add_argument('--eval_step',default = 1,type=int)
	parser.add_argument('--log_file',default='experiment.log')
	parser.add_argument('--ckpt_path',default='ckpt/')
	parser.add_argument('--test', action="store_true")
	parser.add_argument('--max_length',default = 128, help="body text max length")
	parser.add_argument('--do_preprocess',action="store_true")
	args = parser.parse_args()
	main(args)