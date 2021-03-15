
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("__file__"))))

import nets
import utils

os.environ['CUDA_VISIBLE_DEVICES']='1'
# os.environ['CUDA_VISIBLE_DEVICES']='0'
config_path = '../yml/solaris_train.yml'
config = utils.config.parse(config_path)

# print('Config:')
# for key, value in config.items():
#     print('{} : {}', key value)





# make model output dir
# os.makedirs(os.path.dirname(config['training']['model_dest_path']), exist_ok=True)
os.makedirs(os.path.dirname(config['training']['callbacks']['model_checkpoint']['filepath']), exist_ok=True)

trainer = nets.train.Trainer(config=config)
trainer.train()
