import os
import sys
import torch
import skimage.io
import numpy as np
from warnings import warn
from .model_io import get_model
from .transform import process_aug_dict
from .datagen import InferenceTiler as InferenceTiler
from utils.core import get_data_paths
import torch.nn.functional as F

class Inferer(object):
    """Object for training `solaris` models using PyTorch or Keras."""

    def __init__(self, config, custom_model_dict=None):
        self.config = config
        self.batch_size = self.config['batch_size']
        self.framework = self.config['nn_framework']
        self.model_name = self.config['model_name']
        self.aoi = self.config["get_aoi"]
        self.date = self.config["training_date"]
        self.boundary = self.config["boundary"]
        self.weight_file = self.config['weight_file']

        # check if the model was trained as part of the same pipeline; if so,
        # use the output from that. If not, use the pre-trained model directly.
        if self.config['train']:
            warn('Because the configuration specifies both training and '
                 'inference, solaris is switching the model weights path '
                 'to the training output path.')
            self.model_path = self.config['training']['model_dest_path']
            if custom_model_dict is not None:
                custom_model_dict['weight_path'] = self.config[
                    'training']['model_dest_path']
        else:
    
            if len(self.model_name.split('_'))==2:
                self.model_path = self.config.get('model_path', None) + self.aoi + '_' +self.model_name.split('_')[0]+ '_' + self.model_name.split('_')[1]+ '_'+ self.date + '/' + self.weight_file
            else : 
                self.model_path = self.config.get('model_path', None) + self.aoi + '_' +self.model_name.split('_')[0]+ '_' + self.date + '/' + self.weight_file
        self.infer_mode = self.config['infer']
        if self.infer_mode :
            self.mode = 'Infer'
        
        self.model = get_model(self.model_name, self.framework, self.mode,
                               self.model_path, pretrained=True,  custom_model_dict=custom_model_dict)
        self.window_step_x = self.config['inference'].get('window_step_size_x',
                                                          None)
        self.window_step_y = self.config['inference'].get('window_step_size_y',
                                                          None)
        if self.window_step_x is None:
            self.window_step_x = self.config['data_specs']['width']
        if self.window_step_y is None:
            self.window_step_y = self.config['data_specs']['height']
        self.stitching_method = self.config['inference'].get(
            'stitching_method', 'average')
        self.output_dir = self.config['inference']['output_dir'] + self.aoi + '_' + self.date + '/'

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        if self.framework in ['torch', 'pytorch']:
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.gpu_count = torch.cuda.device_count()
            else:
                self.gpu_count = 0
    def __call__(self, infer_df=None):

        with torch.no_grad():
            print(self.model_path)
            if infer_df is None:
                infer_df = get_infer_df(self.config)

            inf_tiler = InferenceTiler(
                self.framework,
                width=self.config['data_specs']['width'],
                height=self.config['data_specs']['height'],
                x_step=self.window_step_x,
                y_step=self.window_step_y,
                augmentations=process_aug_dict(
                    self.config['inference_augmentation']))
            for idx, im_path in enumerate(infer_df['image']):
                leng=len(infer_df['image'])
                print(idx,'/',leng, '  (%0.2f%%)' % float(100*idx/leng))

                inf_input, idx_refs, (
                    src_im_height, src_im_width) = inf_tiler(im_path)

                if self.framework in ['torch', 'pytorch']:

                    with torch.no_grad():
                        self.model.eval()

                    if torch.cuda.is_available():
                        device = torch.device('cuda')
                        self.model = self.model.cuda()
                    else:
                        device = torch.device('cpu')

                    inf_input = torch.from_numpy(inf_input).float().to(device)

                    # add additional input data, if applicable
                    if self.config['data_specs'].get('additional_inputs',
                                                     None) is not None:
                        inf_input = [inf_input]
                        for i in self.config['data_specs']['additional_inputs']:
                            inf_input.append(
                                infer_df[i].iloc[idx].to(device))
                    
    
                    
                    subarr_preds = self.model(inf_input)


                    subarr_preds = subarr_preds.cpu().data.numpy()
                    subarr_preds = subarr_preds[:, :, :src_im_height,:src_im_width]

                    
                       
                skimage.io.imsave(os.path.join(self.output_dir,os.path.split(im_path)[1]), subarr_preds)


def get_infer_df(config):

    infer_df = get_data_paths(config['inference_data_csv']+config['get_aoi']+'_Test_df.csv' , infer=True)
    return infer_df
