
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("__file__"))))

import nets
import utils

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
config_path = '../yml/solaris_infer.yml'
config = utils.config.parse(config_path)
# print('Config:')
# print(config)

# make infernce output dir
# os.makedirs(os.path.dirname(config['inference']['output_dir']), exist_ok=True)

inferer = nets.infer.Inferer(config)
inferer()
