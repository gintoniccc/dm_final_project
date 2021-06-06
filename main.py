import os
import sys
import logging
import warnings
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from Preprocessor import Preprocessor
from Trainer import Trainer
from utils import set_seed,Timer
from argparse import ArgumentParser

warnings.filterwarnings("ignore") 
SEED = 0
def main(args):
	set_seed(SEED)
	logging.basicConfig(filename=args.log_file, level=logging.INFO)
	logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
	logging.info('loading model...')

	tokenizer = XLMRobertaTokenizer.from_pretrained(args.model)
	backbone = XLMRobertaModel.from_pretrained(args.model)

	logging.info('start data preprocessing...')
	with Timer("parse data and sampling.."):
		P = Preprocessor(args)
		##Todo resample for several batch ?
		train_loader,valid_loader = P.sample(args.sample_rate) #may change to train,valid,test
	logging.info('finish data preprocessing')
	trainer = Trainer(backbone,tokenizer,train_loader,valid_loader, args)
	trainer.train()
	logging.info('finish training')

	#Todo:
	#body_emb = trainer.generate_emb()
	#plot some result ,eg : tsne

	return

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--chat_df_path',default='./data/chats_2021-05.csv')
	parser.add_argument('--delete_df_path',default='./data/deletion_events.csv')
	parser.add_argument('--model',default='xlm-roberta-base')
	parser.add_argument('--batch_size',default=64,type=int)
	parser.add_argument('--lr',default = 1e-3)
	parser.add_argument('--epoch_num',default=8,type=int)
	parser.add_argument('--sample_rate',default=2,type=int)
	parser.add_argument('--trainsize_ratio',default = 0.9)
	parser.add_argument('--eval_step',default = 1,type=int)
	parser.add_argument('--log_file',default='experiment.log')
	parser.add_argument('--ckpt_path',default='ckpt/')
	parser.add_argument('--test', action="store_true")
	parser.add_argument('--max_length',default = 64, help="body text max length")
	args = parser.parse_args()
	main(args)