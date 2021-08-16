
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("__file__"))))

import nets
import utils
import time
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

config_path = '../yml/train.yml'
config = utils.config.parse(config_path)

# make model output dir

os.makedirs(os.path.dirname(config['training']['callbacks']['model_checkpoint']['filepath']), exist_ok=True)
start_time = str(int(time.time()))
config['start_time'] = start_time
trainer = nets.train.Trainer(config=config)
trainer.train()
