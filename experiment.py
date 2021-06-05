import os
import sys
import logging

log_file = 'experiment.log'
logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.DEBUG, 
    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

experiment = "python main.py --sample_rate=1 --ckpt_path='./ckpt/ckpt_sample_rate_1/' --log_file=%s" % (log_file)
logging.info('[Experiment]')
logging.info(experiment)
os.system(experiment)

experiment = "python main.py --sample_rate=2 --ckpt_path='./ckpt/ckpt_sample_rate_2/' --log_file=%s" % (log_file)
logging.info('[Experiment]')
logging.info(experiment)
os.system(experiment)

experiment = "python main.py --sample_rate=5 --ckpt_path='./ckpt/ckpt_sample_rate_5/' --log_file=%s" % (log_file)
logging.info('[Experiment]')
logging.info(experiment)
os.system(experiment)

experiment = "python main.py --sample_rate=10 --ckpt_path='./ckpt/ckpt_sample_rate_10/' --log_file=%s" % (log_file)
logging.info('[Experiment]')
logging.info(experiment)
os.system(experiment)
