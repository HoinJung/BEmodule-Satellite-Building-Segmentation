import os
# os.environ['CUDA_VISIBLE_DEVICES']='1,2,3'
import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("__file__"))))
import torch
import gdal
import numpy as np
from warnings import warn
from .model_io import get_model
from .transform import process_aug_dict
from .datagen import InferenceTiler as InferenceTiler
from utils.core import get_data_paths
from raster.image import stitch_images, create_multiband_geotiff

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
#         self.weight_path = self.config['model_path']['callbacks']['model_checkpoint']['filepath']+self.aoi+'_'+ self.start_time +'/' 
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
            self.model_path = self.config.get('model_path', None) + self.aoi + '_' + self.date + '/' + 'final.pth'
#             self.model_path = self.config.get('model_path', None) + self.aoi + '_' + self.date + '/' + 'best_epoch73_0.127.pth'
        self.model = get_model(self.model_name, self.framework,
                               self.model_path, pretrained=True,
                               custom_model_dict=custom_model_dict)
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
        self.boundary_dir = self.output_dir + 'boundary/'
        self.sum_dir = self.output_dir + 'check_mask_presigmoid/'
#         self.mask_dir = self.output_dir + 'mask/'
        self.final_dir = self.output_dir + 'binary/'
        
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.boundary_dir):
            os.makedirs(self.boundary_dir)
        if not os.path.isdir(self.sum_dir):
            os.makedirs(self.sum_dir)
#         if not os.path.isdir(self.mask_dir):
#             os.makedirs(self.mask_dir)
        if not os.path.isdir(self.final_dir):
            os.makedirs(self.final_dir)
#         if self.framework in ['torch', 'pytorch']:
#             self.gpu_available = torch.cuda.is_available()
#             if self.gpu_available:
#                 self.gpu_count = torch.cuda.device_count()
#             else:
#                 self.gpu_count = 0
    def __call__(self, infer_df=None):
        """Run inference.
        Arguments
        ---------
        infer_df : :class:`pandas.DataFrame` or `str`
            A :class:`pandas.DataFrame` with a column, ``'image'``, specifying
            paths to images for inference. Alternatively, `infer_df` can be a
            path to a CSV file containing the same information.  Defaults to
            ``None``, in which case the file path specified in the Inferer's
            configuration dict is used.
        """
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
                temp_im = gdal.Open(im_path)
                proj = temp_im.GetProjection()
                gt = temp_im.GetGeoTransform()

                leng=len(infer_df['image'])
                print(idx+1,'/',leng, '  (%0.2f%%)' % float(100*idx/leng))

                inf_input, idx_refs, (
                    src_im_height, src_im_width) = inf_tiler(im_path)

                if self.framework == 'keras':
                    subarr_preds = self.model.predict(inf_input,
                                                      batch_size=self.batch_size)


                elif self.framework in ['torch', 'pytorch']:
                    #self.model.eval()
                    with torch.no_grad():
                        self.model.eval()

                    #self.model.eval()    
                    if torch.cuda.is_available():

                        device = torch.device('cuda')
                        self.model = self.model.cuda()

    #                     device = torch.device('cpu')



                        print(self.date)
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
    ##UNET
                    if self.boundary :
                        d1,d2,d3,d4,d5,boundary, subarr_preds ,combi, binary= self.model(inf_input)
                        ## edgefeature
#                         d1 = d1.cpu().data.numpy()
#                         d1 = d1[:, :, :src_im_height,:src_im_width]
#                         d2 = d2.cpu().data.numpy()
#                         d2 = d2[:, :, :src_im_height,:src_im_width]
#                         d3 = d3.cpu().data.numpy()
#                         d3 = d3[:, :, :src_im_height,:src_im_width]
#                         d4 = d4.cpu().data.numpy()
#                         d4 = d4[:, :, :src_im_height,:src_im_width]
#                         d5 = d5.cpu().data.numpy()
#                         d5 = d5[:, :, :src_im_height,:src_im_width]
                        
                        #mask
                        subarr_preds = subarr_preds.cpu().data.numpy()
                        subarr_preds = subarr_preds[:, :, :src_im_height,:src_im_width]
                        #boundary
                        boundary = boundary.cpu().data.numpy()
                        boundary = boundary[:, :, :src_im_height,:src_im_width]
                        
                        sum_output = combi.cpu().data.numpy()
                        sum_output = sum_output[:, :, :src_im_height,:src_im_width]
                        #binary
                        binary = binary.cpu().data.numpy()
                        binary = binary[:, :, :src_im_height,:src_im_width]
                        
                    else :
                        
                        subarr_preds = self.model(inf_input)
                        subarr_preds = subarr_preds.cpu().data.numpy()
                        subarr_preds = subarr_preds[:, :, :src_im_height,:src_im_width]                    #1228

                       
                if self.boundary :
#                     d1_result = stitch_images(d1,
#                                                     idx_refs=idx_refs,
#                                                     out_width=src_im_width,
#                                                     out_height=src_im_height,
#                                                     method=self.stitching_method)
#                     d1_result = np.swapaxes(d1_result, 1, 0)
#                     d1_result = np.swapaxes(d1_result, 2, 0)
#                     d2_result = stitch_images(d2,
#                                                     idx_refs=idx_refs,
#                                                     out_width=src_im_width,
#                                                     out_height=src_im_height,
#                                                     method=self.stitching_method)
#                     d2_result = np.swapaxes(d2_result, 1, 0)
#                     d2_result = np.swapaxes(d2_result, 2, 0)
#                     d3_result = stitch_images(d3,
#                                                     idx_refs=idx_refs,
#                                                     out_width=src_im_width,
#                                                     out_height=src_im_height,
#                                                     method=self.stitching_method)
#                     d3_result = np.swapaxes(d3_result, 1, 0)
#                     d3_result = np.swapaxes(d3_result, 2, 0)
#                     d4_result = stitch_images(d4,
#                                                     idx_refs=idx_refs,
#                                                     out_width=src_im_width,
#                                                     out_height=src_im_height,
#                                                     method=self.stitching_method)
#                     d4_result = np.swapaxes(d4_result, 1, 0)
#                     d4_result = np.swapaxes(d4_result, 2, 0)
#                     d5_result = stitch_images(d5,
#                                                     idx_refs=idx_refs,
#                                                     out_width=src_im_width,
#                                                     out_height=src_im_height,
#                                                     method=self.stitching_method)
#                     d5_result = np.swapaxes(d5_result, 1, 0)
#                     d5_result = np.swapaxes(d5_result, 2, 0)
                    
                    stitched_result = stitch_images(subarr_preds,
                                                    idx_refs=idx_refs,
                                                    out_width=src_im_width,
                                                    out_height=src_im_height,
                                                    method=self.stitching_method)
                    stitched_result = np.swapaxes(stitched_result, 1, 0)
                    stitched_result = np.swapaxes(stitched_result, 2, 0)
                    #boundary
                    stitched_boundary = stitch_images(boundary,
                                                    idx_refs=idx_refs,
                                                    out_width=src_im_width,
                                                    out_height=src_im_height,
                                                    method=self.stitching_method)
                    stitched_boundary = np.swapaxes(stitched_boundary, 1, 0)       
                    stitched_boundary = np.swapaxes(stitched_boundary, 2, 0)
                    #sum
                    stitched_sum = stitch_images(sum_output,
                                                    idx_refs=idx_refs,
                                                    out_width=src_im_width,
                                                    out_height=src_im_height,
                                                    method=self.stitching_method)            
                    stitched_sum = np.swapaxes(stitched_sum, 1, 0)
                    stitched_sum = np.swapaxes(stitched_sum, 2, 0)
                    #binary
                    stitched_final = stitch_images(binary,
                                                    idx_refs=idx_refs,
                                                    out_width=src_im_width,
                                                    out_height=src_im_height,
                                                    method=self.stitching_method)            
                    stitched_final = np.swapaxes(stitched_final, 1, 0)
                    stitched_final = np.swapaxes(stitched_final, 2, 0)

                    
                    
#                     create_multiband_geotiff(d1_result,
#                                              os.path.join(self.boundary_dir,'d1',
#                                                           os.path.split(im_path)[1]).replace('RGB-PanSharpen','d1'),
#                                              proj=proj, geo=gt, nodata=np.nan,
#                                              out_format=gdal.GDT_Float32)
#                     create_multiband_geotiff(d2_result,
#                                              os.path.join(self.boundary_dir,'d2',
#                                                           os.path.split(im_path)[1]).replace('RGB-PanSharpen','d2'),
#                                              proj=proj, geo=gt, nodata=np.nan,
#                                              out_format=gdal.GDT_Float32)
#                     create_multiband_geotiff(d3_result,
#                                              os.path.join(self.boundary_dir,'d3',
#                                                           os.path.split(im_path)[1]).replace('RGB-PanSharpen','d3'),
#                                              proj=proj, geo=gt, nodata=np.nan,
#                                              out_format=gdal.GDT_Float32)
#                     create_multiband_geotiff(d4_result,
#                                              os.path.join(self.boundary_dir,'d4',
#                                                           os.path.split(im_path)[1]).replace('RGB-PanSharpen','d4'),
#                                              proj=proj, geo=gt, nodata=np.nan,
#                                              out_format=gdal.GDT_Float32)
#                     create_multiband_geotiff(d5_result,
#                                              os.path.join(self.boundary_dir,'d5',
#                                                           os.path.split(im_path)[1]).replace('RGB-PanSharpen','d5'),
#                                              proj=proj, geo=gt, nodata=np.nan,
#                                              out_format=gdal.GDT_Float32)
                    
                    
                    #mask
                    create_multiband_geotiff(stitched_result,
                                             os.path.join(self.output_dir,
                                                          os.path.split(im_path)[1]).replace('RGB-PanSharpen','mask'),
                                             proj=proj, geo=gt, nodata=np.nan,
                                             out_format=gdal.GDT_Float32)
                    #boundary
                    create_multiband_geotiff(stitched_boundary,
                                             os.path.join(self.boundary_dir,
                                                          os.path.split(im_path)[1]).replace('RGB-PanSharpen','Boundary'),
                                             proj=proj, geo=gt, nodata=np.nan,
                                             out_format=gdal.GDT_Float32)
        #             #sum
                    create_multiband_geotiff(stitched_sum,
                                             os.path.join(self.sum_dir,
                                                          os.path.split(im_path)[1]).replace('RGB-PanSharpen','sum_infer'),
                                             proj=proj, geo=gt, nodata=np.nan,
                                             out_format=gdal.GDT_Float32)
                    #binary
                    create_multiband_geotiff(stitched_final,
                                             os.path.join(self.final_dir,
                                                          os.path.split(im_path)[1]).replace('RGB-PanSharpen','binary'),
                                             proj=proj, geo=gt, nodata=np.nan,
                                             out_format=gdal.GDT_Float32)
                else : 
    #     # # ###BE
                    stitched_result = stitch_images(subarr_preds,
                                                    idx_refs=idx_refs,
                                                    out_width=src_im_width,
                                                    out_height=src_im_height,
                                                    method=self.stitching_method)
                    stitched_result = np.swapaxes(stitched_result, 1, 0)
                    stitched_result = np.swapaxes(stitched_result, 2, 0)

            

                    create_multiband_geotiff(stitched_result,
                                             os.path.join(self.output_dir,
                                                          os.path.split(im_path)[1]).replace('RGB-PanSharpen','mask'),
                                             proj=proj, geo=gt, nodata=np.nan,
                                             out_format=gdal.GDT_Float32)


                    # mask
                    
def get_infer_df(config):
    """Get the inference df based on the contents of ``config`` .
    This function uses the logic described in the documentation for the config
    file to determine where to find images to be used for inference.
    See the docs and the comments in solaris/data/config_skeleton.yml for
    details.
    Arguments
    ---------
    config : dict
        The loaded configuration dict for model training and/or inference.
    Returns
    -------
    infer_df : :class:`dict`
        :class:`dict` containing at least one column: ``'image'`` . The values
        in this column correspond to the path to filenames to perform inference
        on.
    """

    infer_df = get_data_paths(config['inference_data_csv']+config['get_aoi']+'_Test_df.csv' , infer=True)
    return infer_df
