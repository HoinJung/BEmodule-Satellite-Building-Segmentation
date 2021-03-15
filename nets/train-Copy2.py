"""Training code for `solaris` models."""

import numpy as np
import pandas as pd
from .model_io import get_model, reset_weights
from .datagen import make_data_generator
from .losses import get_loss
from .optimizers import get_optimizer
from .callbacks import get_callbacks
from .torch_callbacks import TorchEarlyStopping, TorchTerminateOnNaN
from .torch_callbacks import TorchModelCheckpoint
from .metrics import get_metrics
import torch
from torch.optim.lr_scheduler import _LRScheduler
import tensorflow as tf
import time
import skimage
from .assembly_block import assembly_block
now = time.localtime()


class Trainer(object):
    """Object for training `solaris` models using PyTorch or Keras. """

    def __init__(self, config, custom_model_dict=None, custom_losses=None):
        self.config = config
        self.pretrained = self.config['pretrained']
        self.batch_size = self.config['batch_size']
        self.framework = self.config['nn_framework']
        self.model_name = self.config['model_name']
        self.boundary = self.config['boundary']
        self.model_path = self.config.get('model_path', None)
        self.start_time = str(now.tm_mon) + str(now.tm_mday) +  str(now.tm_hour) + str(now.tm_min)
        
        ##aoi add part
        self.aoi = self.config["get_aoi"]
#         self.weight_path = self.config['training']['callbacks']['model_checkpoint']['filepath']+self.aoi+'_'+ self.start_time +'/' 
        self.weight_path = self.config['training']['callbacks']['model_checkpoint']['filepath']+self.aoi+'_'+ self.start_time  +'/' 
        
        try:
            self.num_classes = self.config['data_specs']['num_classes']
        except KeyError:
            self.num_classes = 1
        self.model = get_model(self.model_name, self.framework,
                               self.model_path, self.pretrained,
                               custom_model_dict, self.num_classes)

        self.train_df, self.val_df = get_train_val_dfs(self.config)
        self.train_datagen = make_data_generator(self.framework, self.config,
                                                 self.train_df, stage='train')
        self.val_datagen = make_data_generator(self.framework, self.config,
                                               self.val_df, stage='validate')
        self.epochs = self.config['training']['epochs']
        self.optimizer = get_optimizer(self.framework, self.config)
        self.lr = self.config['training']['lr']
        self.custom_losses = custom_losses
        self.loss = get_loss(self.framework,
                             self.config['training'].get('loss'),
                             self.config['training'].get('loss_weights'),
                             self.custom_losses)
#         self.loss_mask = get_loss(self.framework,
#                              self.config['training'].get('loss_mask'),
#                              self.config['training'].get('loss_mask_weights'),
#                              self.custom_losses)
#         self.loss_boundary = get_loss(self.framework,
#                              self.config['training'].get('loss_boundary'),
#                              self.config['training'].get('loss_boundary_weights'),
#                              self.custom_losses)
        self.checkpoint_frequency = self.config['training'].get('checkpoint_'
                                                                + 'frequency')
        self.callbacks = get_callbacks(self.framework, self.config)
        self.metrics = get_metrics(self.framework, self.config)
        self.verbose = self.config['training']['verbose']
        if self.framework in ['torch', 'pytorch']:
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.gpu_count = torch.cuda.device_count()
            else:
                self.gpu_count = 0
        elif self.framework == 'keras':
            self.gpu_available = tf.test.is_gpu_available()

        self.is_initialized = False
        self.stop = False

        self.initialize_model()

    def initialize_model(self):
        """Load in and create all model training elements."""
        if not self.pretrained:
            self.model = reset_weights(self.model, self.framework)

        if self.framework == 'keras':
            self.model = self.model.compile(optimizer=self.optimizer,
                                            loss=self.loss,
                                            metrics=self.metrics['train'])

        elif self.framework == 'torch':
            if self.gpu_available:
                self.model = self.model.cuda()
                if self.gpu_count > 1:
                    self.model = torch.nn.DataParallel(self.model)
            # create optimizer
            if self.config['training']['opt_args'] is not None:
                self.optimizer = self.optimizer(
                    self.model.parameters(), lr=self.lr,
                    **self.config['training']['opt_args']
                )
            else:
                self.optimizer = self.optimizer(
                    self.model.parameters(), lr=self.lr
                )
            # wrap in lr_scheduler if one was created
            for cb in self.callbacks:
                if isinstance(cb, _LRScheduler):
                    self.optimizer = cb(
                        self.optimizer,
                        **self.config['training']['callbacks'][
                            'lr_schedule'].get(['schedule_dict'], {})
                        )
                    # drop the LRScheduler callback from the list
                    self.callbacks = [i for i in self.callbacks if i != cb]

        self.is_initialized = True

    def train(self):
        """Run training on the model."""
        if not self.is_initialized:
            self.initialize_model()

        if self.framework == 'keras':
            self.model.fit_generator(self.train_datagen,
                                     validation_data=self.val_datagen,
                                     epochs=self.epochs,
                                     callbacks=self.callbacks)

        elif self.framework == 'torch':
#            tf_sess = tf.Session()
            val_loss_set = []
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                  milestones=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350],
                                   gamma=0.5)
            print()
            print("=======================")
            print("Trainging Start")
            print("aoi :", self.aoi)
            print("model : ", self.model_name)
            print("batch size : ",self.batch_size)
            print("epoch : ", self.epochs)
            print("starting time : %04d/%02d/%02d %02d:%02d:%02d"% (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
            print("=======================")
            print()
            
            for epoch in range(self.epochs):
                if self.verbose:
                    print('Beginning training epoch {}'.format(epoch))
                # TRAINING
                self.model.train()
                for batch_idx, batch in enumerate(self.train_datagen):
                    if torch.cuda.is_available():
                        if self.config['data_specs'].get('additional_inputs',
                                                         None) is not None:
                            data = []
                            for i in ['image'] + self.config[
                                    'data_specs']['additional_inputs']:
                                data.append(torch.Tensor(batch[i]).cuda())
                        else:
                            data = batch['image'].cuda()
                        ############################################################ original                            
#                         target = batch['mask'].cuda().float()
                        ##################################################### boundary added
                        if self.boundary : 
                            target_m = batch['mask'].cuda().float()
#                             print(target_m.shape)
                            target_b = batch['boundary'].cuda().float()
#                             print(target_b.shape)
                        else :
                            target = batch['mask'].cuda().float()
                    else:
                        if self.config['data_specs'].get('additional_inputs',
                                                         None) is not None:
                            data = []
                            for i in ['image'] + self.config[
                                    'data_specs']['additional_inputs']:
                                data.append(torch.Tensor(batch[i]))
                        else:
                            data = batch['image']
                        target = batch['mask'].float()
                    self.optimizer.zero_grad()
                    ############################################################ original
#                     output = self.model(data)
#                     loss = self.loss(output, target)
                    ############################################################ boundary added
                    if self.boundary :
#                         output_m = self.model(data)
#                         tmp_m = output_m
#                         arr_m_output = tmp_m.cpu().detach().numpy()
#                         output_arr = skimage.segmentation.find_boundaries(arr_m_output, mode='inner', background=0).astype(np.float32)
#                         output_b = torch.from_numpy(output_arr).cuda().float()
#                         loss_m = self.loss(output_m, target_m)
#                         loss_b = self.loss(output_b, target_b)
#                         loss = loss_m + loss_b
                        
                        # 1223 modify 
#                         conv_m, output_m = self.model(data)
#                         output_b, output_m = assembly_block(conv_m, output_m)
#                         loss_m = self.loss(output_m, target_m)
#                         loss_b = self.loss(output_b, target_b)
#                         loss = (loss_m/4) + (loss_b/4)*3
                        
                        # 1224 modify 
#                         output_m, output_b = self.model(data)
#                         loss_m = self.loss_mask(output_m, target_m)
#                         loss_b = self.loss_boundary(output_b, target_b)
# #                         loss = (loss_m/4) + (loss_b/4)*3
#                         loss = loss_m + loss_b
                        
                        # 1225 modify
#                         output_m, output_b = self.model(data)
#                         loss_m = self.loss(output_m, target_m)
#                         loss_b = self.loss(output_b, target_b)
#                         loss = (loss_m/4) + (loss_b/4)*3
# #                         loss = loss_m + loss_b
                        
                        # boundary only
#                         output_m, output_b = self.model(data)
#                         loss = self.loss(output_b, target_b)
                        
#                         # 1227 HEM
#                         o1, o2, o3, o4, o5, o6 = self.model(data)
#                         loss1 = self.loss(o1, target_b)
#                         loss2 = self.loss(o2, target_b)
#                         loss3 = self.loss(o3, target_b)
#                         loss4 = self.loss(o4, target_b)
#                         loss5 = self.loss(o5, target_b)
#                         loss6 = self.loss(o6, target_b)
#                         loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                        
# #                         # 1228 HED UNET
#                         o1, o2, o3, o4, o5, o6, mask,sum_output = self.model(data)
#                         loss1 = self.loss(o1, target_b)
#                         loss2 = self.loss(o2, target_b)
#                         loss3 = self.loss(o3, target_b)
#                         loss4 = self.loss(o4, target_b)
#                         loss5 = self.loss(o5, target_b)
#                         loss6 = self.loss(o6, target_b)
#                         loss7 = self.loss(mask, target_m)
                        
#                         loss = (loss1 + loss2 + loss3 + loss4 + loss5)/10+ (loss6/4) + (loss7/4)
# #                         loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6)*3/7 + (loss7/7)

                        # 1229 Edge UNET
                      
#                         o1, o2, o3, o4, o5, o6, mask,binary = self.model(data)
#                         loss1 = self.loss(o1, target_b)
#                         loss2 = self.loss(o2, target_b)
#                         loss3 = self.loss(o3, target_b)
#                         loss4 = self.loss(o4, target_b)
#                         loss5 = self.loss(o5, target_b)
#                         loss6 = self.loss(o6, target_b)
#                         loss7 = self.loss(mask, target_m)
                        
#                         loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5 + (loss6+2*loss7)
                        
#                          1229 BE UNET (main)

                        o1, o2, o3, o4, o5, boundary,out,   binary = self.model(data)
#                         o1, o2, o3, o4, o5, boundary,out,  mask,  binary = self.model(data)
                        loss1 = self.loss(o1, target_b)
                        loss2 = self.loss(o2, target_b)
                        loss3 = self.loss(o3, target_b)
                        loss4 = self.loss(o4, target_b)
                        loss5 = self.loss(o5, target_b)
                        loss6 = self.loss(boundary, target_b)
                        loss7 = self.loss(out, target_m)
#                         loss8 = self.loss(mask, target_m)


                        
                        loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5 + (loss6+loss7)
#                         loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5 + (loss6 + 2*loss7)
#                         loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5 + (loss6 + 5*loss7)
                        
    
    
    
#                         ## no HED loss
#                         boundary,mask,   binary = self.model(data)
#                         loss1 = self.loss(boundary, target_b)
#                         loss2 = self.loss(mask, target_m)
#                         loss = loss1+loss2
                        
                        ##   1230 Residual BE UNET 
#                         o1, o2, o3, o4, o5, boundary, mask,binary = self.model(data)
#                         loss1 = self.loss(o1, target_b)
#                         loss2 = self.loss(o2, target_b)
#                         loss3 = self.loss(o3, target_b)
#                         loss4 = self.loss(o4, target_b)
#                         loss5 = self.loss(o5, target_b)
#                         loss6 = self.loss(boundary, target_b)
#                         loss7 = self.loss(mask, target_m)
# #                         loss = (loss1 + loss2 + loss3 + loss4 + loss5)/10 + (loss6 + loss7)
#                         beta = loss1 + loss2 + loss3 + loss4 + loss5
#                         loss = (beta)/10 + (loss6) + loss7/beta
                        
#                         ##   1230 Combination
#                         o1, o2, o3, o4, o5, boundary, mask, combination, binary = self.model(data)
#                         loss1 = self.loss(o1, target_b)
#                         loss2 = self.loss(o2, target_b)
#                         loss3 = self.loss(o3, target_b)
#                         loss4 = self.loss(o4, target_b)
#                         loss5 = self.loss(o5, target_b)
#                         loss6 = self.loss(boundary, target_b)
#                         loss7 = self.loss(mask, target_m)
#                         loss_f = self.loss(combination, target_m)
#                         loss = (loss1 + loss2 + loss3 + loss4 + loss5+loss6)/6 + loss7 + loss_f
                        
                        
                    else : 
                        output = self.model(data)
                        loss = self.loss(output, target)
                    loss.backward()
                    self.optimizer.step()

                    if self.verbose and batch_idx % 10 == 0:

                        print('    loss at batch {}: {}'.format(
                            batch_idx, loss), flush=True)

                # VALIDATION
                with torch.no_grad():
                    self.model.eval()
                    torch.cuda.empty_cache()
                    val_loss = []
                    for batch_idx, batch in enumerate(self.val_datagen):
                        if torch.cuda.is_available():
                            if self.config['data_specs'].get(
                                    'additional_inputs', None) is not None:
                                data = []
                                for i in ['image'] + self.config[
                                        'data_specs']['additional_inputs']:
                                    data.append(torch.Tensor(batch[i]).cuda())
                            else:
                                data = batch['image'].cuda()
                            if self.boundary : 
                                target_m = batch['mask'].cuda().float()

                                target_b = batch['boundary'].cuda().float()

                            else :
                                target = batch['mask'].cuda().float()
                            
                        else:
                            if self.config['data_specs'].get(
                                    'additional_inputs', None) is not None:
                                data = []
                                for i in ['image'] + self.config[
                                        'data_specs']['additional_inputs']:
                                    data.append(torch.Tensor(batch[i]))
                            else:
                                data = batch['image']
                            target = batch['mask'].float()
                            
                    if self.boundary :
#                         a, val_output_m = self.model(data)
#                         val_tmp_m = val_output_m
#                         val_arr_m_output = val_output_m.cpu().detach().numpy()
#                         val_output_arr = skimage.segmentation.find_boundaries(val_arr_m_output, mode='inner', background=0).astype(np.float32)
#                         val_output_b = torch.from_numpy(val_output_arr).cuda().float()

#                         val_loss_m = self.loss(val_output_m, target_m)
#                         val_loss_b = self.loss(val_output_b, target_b)
#                         val_loss.append(val_loss_m + val_loss_b)
                        
                        # 1224 modify
#                         val_output_m, val_output_b = self.model(data)
#                         val_loss_m = self.loss_mask(val_output_m, target_m)
#                         val_loss_b = self.loss_boundary(val_output_b, target_b)
#                         val_loss.append(val_loss_m + val_loss_b)
                        
                        # 1225 modify
#                         val_output_m, val_output_b = self.model(data)
#                         val_loss_m = self.loss(val_output_m, target_m)
#                         val_loss_b = self.loss(val_output_b, target_b)
#                         val_loss.append(val_loss_m/4 + val_loss_b*3/4)
                        
#                         # boundary only
#                         val_output_m, val_output_b = self.model(data)
#                         val_loss.append(self.loss(val_output_b, target_b))
                
                        #1227 HED
#                         v1,v2,v3,v4,v5,v6 = self.model(data)
#                         loss_val_1 = self.loss(v1, target_b)
#                         loss_val_2 = self.loss(v2, target_b)
#                         loss_val_3 = self.loss(v3, target_b)
#                         loss_val_4 = self.loss(v4, target_b)
#                         loss_val_5 = self.loss(v5, target_b)
#                         loss_val_6 = self.loss(v6, target_b)
#                         loss = loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4 + loss_val_5 + loss_val_6 
#                         val_loss.append(loss)
                        
#                         #1228 HED Unet
#                         v1,v2,v3,v4,v5,v6,v_mask,v_sum_output = self.model(data)
#                         loss_val_1 = self.loss(v1, target_b)
#                         loss_val_2 = self.loss(v2, target_b)
#                         loss_val_3 = self.loss(v3, target_b)
#                         loss_val_4 = self.loss(v4, target_b)
#                         loss_val_5 = self.loss(v5, target_b)
#                         loss_val_6 = self.loss(v6, target_b)
#                         loss_val_7 = self.loss(v_mask, target_m)
#                         loss = (loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4 + loss_val_5 + loss_val_6 ) * 3/7 + (loss_val_7 /7)
#                         val_loss.append(loss)

# #                          #1229 Edge Unet
#                         v1,v2,v3,v4,v5,v6,v_mask,v_binary = self.model(data)
#                         loss_val_1 = self.loss(v1, target_b)
#                         loss_val_2 = self.loss(v2, target_b)
#                         loss_val_3 = self.loss(v3, target_b)
#                         loss_val_4 = self.loss(v4, target_b)
#                         loss_val_5 = self.loss(v5, target_b)
#                         loss_val_6 = self.loss(v6, target_b)
#                         loss_val_7 = self.loss(v_mask, target_m)
                        
# #                       
#                         loss = (loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4 + loss_val_5)/5 + (loss_val_6+2*loss_val_7)
#                         val_loss.append(loss)
                        
                        
#                          #1229 BE Unet
#                         v1,v2,v3,v4,v5,v_boundary,v_mask,v_combi , v_binary = self.model(data)
                        v1,v2,v3,v4,v5,v_boundary,v_mask, v_binary = self.model(data)
                        loss_val_1 = self.loss(v1, target_b)
                        loss_val_2 = self.loss(v2, target_b)
                        loss_val_3 = self.loss(v3, target_b)
                        loss_val_4 = self.loss(v4, target_b)
                        loss_val_5 = self.loss(v5, target_b)
                        loss_val_6 = self.loss(v_boundary, target_b)
                        loss_val_7 = self.loss(v_mask, target_m)
#                         loss_val_8 = self.loss(v_combi, target_m)
#                         loss = (loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4 + loss_val_5)/5 + (loss_val_6 + loss_val_7+loss_val_8)
#                         loss = (loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4 + loss_val_5)/5 + (loss_val_6 + 2*loss_val_7)
#                         loss = (loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4 + loss_val_5)/5 + (loss_val_6 + 5*loss_val_7)
                        loss = (loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4 + loss_val_5)/5 + (loss_val_6 + loss_val_7)
                        val_loss.append(loss)
                        
#                         ## no HED loss
#                         v_boundary, v_mask, v_binary = self.model(data)
#                         loss_val_1 = self.loss(v_boundary, target_b)
#                         loss_val_2 = self.loss(v_mask, target_m)
#                         loss = loss_val_1 + loss_val_2
#                         val_loss.append(loss)
#                         #1230 Residual BE Unet
#                         v1,v2,v3,v4,v5,v_boundary,v_mask,v_binary = self.model(data)
#                         loss_val_1 = self.loss(v1, target_b)
#                         loss_val_2 = self.loss(v2, target_b)
#                         loss_val_3 = self.loss(v3, target_b)
#                         loss_val_4 = self.loss(v4, target_b)
#                         loss_val_5 = self.loss(v5, target_b)
#                         loss_val_6 = self.loss(v_boundary, target_b)
#                         loss_val_7 = self.loss(v_mask, target_m)
                        
# #                         loss = (loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4 + loss_val_5 + loss_val_6 )/10 + (loss_val_6 + loss_val_7)
#                         alpha = (loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4 + loss_val_5 + loss_val_6 )
#                         loss = alpha/10 + loss_val_6 + loss_val_7/alpha
        
        
# #                         1230 combination
#                         v1,v2,v3,v4,v5,v_boundary,v_mask,v_combination, v_binary = self.model(data)
#                         loss_val_1 = self.loss(v1, target_b)
#                         loss_val_2 = self.loss(v2, target_b)
#                         loss_val_3 = self.loss(v3, target_b)
#                         loss_val_4 = self.loss(v4, target_b)
#                         loss_val_5 = self.loss(v5, target_b)
#                         loss_val_6 = self.loss(v_boundary, target_b)
#                         loss_val_7 = self.loss(v_mask, target_m)
#                         loss_val_f = self.loss(v_combination, target_m)
                        
#                         loss = (loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4 + loss_val_5 + loss_val_6 )/6 + loss_val_7 + loss_val_f
                        
                        
                        val_loss.append(loss)
                        
                    else :                                                        
                        val_output = self.model(data)
                        val_loss.append(self.loss(val_output, target))
                        
                        
                        
                    val_loss = torch.mean(torch.stack(val_loss))
                if self.verbose:
                    print()
                    print('    Validation loss at epoch {}: {}'.format(
                        epoch, val_loss))
                    print()

                check_continue = self._run_torch_callbacks(
                    loss.detach().cpu().numpy(),
                    val_loss.detach().cpu().numpy())
                if not check_continue:
                    break
                val_loss_set.append(val_loss)
            self.save_model()
            print("model saved : ",end='')
            print(self.weight_path)
            print("saved time : ",end='')
            print(now.tm_mon, end='')
            print('/', end='')
            print(now.tm_mday)
            temp_file_name = self.weight_path+"loss_val_save.csv"
            np.savetxt(temp_file_name, val_loss_set, delimiter=",")

    def _run_torch_callbacks(self, loss, val_loss):
        for cb in self.callbacks:
            if isinstance(cb, TorchEarlyStopping):
                cb(val_loss)
                if cb.stop:
                    if self.verbose:
                        print('Early stopping triggered - '
                              'ending training')
                    return False

            elif isinstance(cb, TorchTerminateOnNaN):
                cb(val_loss)
                if cb.stop:
                    if self.verbose:
                        print('Early stopping triggered - '
                              'ending training')
                    return False

            elif isinstance(cb, TorchModelCheckpoint):
                # set minimum num of epochs btwn checkpoints (not periodic)
                # or
                # frequency of model saving (periodic)
                # cb.period = self.checkpoint_frequency

                if cb.monitor == 'loss':
                    cb(self.model, loss_value=loss)
                elif cb.monitor == 'val_loss':
                    cb(self.model, loss_value=val_loss)
                elif cb.monitor == 'periodic':
                    # no loss_value specification needed; defaults to `loss`
                    # cb(self.model, loss_value=loss)
                    cb(self.model)

        return True

    def save_model(self):
        """Save the final model output."""
        if self.framework == 'keras':
            self.model.save(self.config['training']['model_dest_path'])
        elif self.framework == 'torch':
            if isinstance(self.model, torch.nn.DataParallel):
                torch.save(self.model.module.state_dict(), self.weight_path+'final.pth')
#                            self.config['training']['callbacks']['model_checkpoint']['filepath']+self.aoi+'_'+ self.start_time +'/' +'final.pth')
#                            self.config['training']['model_dest_path'])
            else:
                torch.save(self.model.state_dict(), self.weight_path+'final.pth')
#                            self.config['training']['callbacks']['model_checkpoint']['filepath']+self.aoi+'_'+ self.start_time+'/' + 'final.pth')
#                            self.config['training']['model_dest_path'])


def get_train_val_dfs(config):
    """Get the training and validation dfs based on the contents of ``config``.

    This function uses the logic described in the documentation for the config
    files to determine where to find training and validation dataset files.
    See the docs and the comments in solaris/data/config_skeleton.yml for
    details.

    Arguments
    ---------
    config : dict
        The loaded configuration dict for model training and/or inference.

    Returns
    -------
    train_df, val_df : :class:`tuple` of :class:`dict` s
        :class:`dict` s containing two columns: ``'image'`` and ``'label'``.
        Each column corresponds to paths to find matching image and label files
        for training.
    """
    #aoi add part

    aoi = config["get_aoi"]
    train_dir = config["train_csv_dir"]
    train_df = pd.read_csv(train_dir+aoi+'_train_df.csv')
        
#     train_df = pd.read_csv(config['training_data_csv'])

    if config['data_specs']['val_holdout_frac'] is None:
        if config['validation_data_csv'] is None:
            raise ValueError(
                "If val_holdout_frac isn't specified in config,"
                " validation_data_csv must be.")
        val_df = pd.read_csv(config['validation_data_csv'])

    else:
        val_frac = config['data_specs']['val_holdout_frac']
        val_subset = np.random.choice(train_df.index,
                                      int(len(train_df)*val_frac),
                                      replace=False)
        val_df = train_df.loc[val_subset]
        # remove the validation samples from the training df
        train_df = train_df.drop(index=val_subset)

    return train_df, val_df
