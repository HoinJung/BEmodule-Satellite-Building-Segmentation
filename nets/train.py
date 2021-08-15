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
import torch
from torch.optim.lr_scheduler import _LRScheduler
import time
import skimage
from .assembly_block import assembly_block
now = time.localtime()


class Trainer(object):
    

    def __init__(self, config, custom_model_dict=None, custom_losses=None):
        self.config = config
        self.pretrained = self.config['pretrained']
        self.batch_size = self.config['batch_size']
        self.framework = self.config['nn_framework']
        self.model_name = self.config['model_name']
        self.boundary = self.config['boundary']
        self.num_stage = self.config['num_stage']
        self.model_path = self.config.get('model_path', None)
        self.start_time = self.config['start_time']
        self.aoi = self.config["get_aoi"]
        self.weight_path = self.config['training']['callbacks']['model_checkpoint']['filepath']+self.aoi+'_'+self.model_name+'_'+ self.start_time  +'/' 
        
        try:
            self.num_classes = self.config['data_specs']['num_classes']
        except KeyError:
            self.num_classes = 1
        self.train_mode = self.config['train']
        if self.train_mode :
            self.mode = 'Train'
        self.model = get_model(self.model_name, self.framework, self.mode, 
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
        self.loss_mask = get_loss(self.framework,
                             self.config['training'].get('loss_mask'),
                             self.config['training'].get('loss_mask_weights'),
                             self.custom_losses)
        self.loss_boundary = get_loss(self.framework,
                             self.config['training'].get('loss_boundary'),
                             self.config['training'].get('loss_boundary_weights'),
                             self.custom_losses)
        self.checkpoint_frequency = self.config['training'].get('checkpoint_'
                                                                + 'frequency')
        self.callbacks = get_callbacks(self.framework, self.config)
        
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


        if self.framework == 'torch':
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

            val_loss_set = []
            

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
                check_lr = get_lr(self.optimizer)
                
                print("check learning rate : %.8f" % check_lr)

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

                        if self.boundary : 
                            target_m = batch['mask'].cuda().float()

                            target_b = batch['boundary'].cuda().float()

                            
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

                    if self.boundary :
                        if self.num_stage==6 : 
                            o1, o2, o3, o4, o5, o6, boundary,out = self.model(data)
                        elif self.num_stage==4:
                            o1, o2, o3, o4, boundary,out = self.model(data)
                        elif self.num_stage==5 : 
                            data, target_m, target_b = data.cuda(), target_m.cuda(), target_b.cuda()
                            o1, o2, o3, o4, o5, boundary,out = self.model(data)
                        loss1 = self.loss_boundary(o1, target_b)
                        loss2 = self.loss_boundary(o2, target_b)
                        loss3 = self.loss_boundary(o3, target_b)
                        loss4 = self.loss_boundary(o4, target_b)
                        loss6 = self.loss_boundary(boundary, target_b)
                        loss7 = self.loss(out, target_m)
                        loss7_s = 1 - self.loss_mask(out, target_m)
                        if self.num_stage ==6 :
                            loss8 = self.loss_boundary(o6, target_b)
                            loss5 = self.loss_boundary(o5, target_b)
                            if epoch < 2 :
                                loss_boundary = (loss1 + loss2 + loss3 + loss4 + loss5+loss8) + (loss6)
                                loss_mask = (loss7 + loss7_s)/2
                                loss = 5*(loss_boundary + loss_mask)
                            else : 
                                loss_boundary = (loss1 + loss2 + loss3 + loss4 + loss5+loss8)/6 + 5*(loss6)
                                loss_mask = (loss7 + loss7_s)
                                loss = loss_boundary + loss_mask
                        elif self.num_stage ==5 : 
                            loss5 = self.loss_boundary(o5, target_b)
                            if epoch < 2 :
                                loss_boundary = (loss1 + loss2 + loss3 + loss4 + loss5) + (loss6)
                                loss_mask = (loss7 + loss7_s)/2
                                loss = 5*(loss_boundary + loss_mask)
                            else : 
                                loss_boundary = (loss1 + loss2 + loss3 + loss4 + loss5)/5 + 5*(loss6)
                                loss_mask = (loss7 + loss7_s)
                                loss = loss_boundary + loss_mask
                        elif self.num_stage==4:
                            if epoch < 2 :
                                loss_boundary = (loss1 + loss2 + loss3 + loss4) + (loss6)
                                loss_mask = (loss7 + loss7_s)/2
                                loss = 5*(loss_boundary + loss_mask)
                            else : 
                                loss_boundary = (loss1 + loss2 + loss3 + loss4)/4 + 5*(loss6)
                                loss_mask = (loss7 + loss7_s)
                                loss = loss_boundary + loss_mask
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
                        
                        if self.num_stage==6 : 
                            v1,v2,v3,v4,v5,v6,v_boundary,v_mask = self.model(data)
                        elif self.num_stage == 4:
                            v1,v2,v3,v4,v_boundary,v_mask = self.model(data)
                        else :     
                            v1,v2,v3,v4,v5,v_boundary,v_mask = self.model(data)

                        loss_val_1 = self.loss_boundary(v1, target_b)
                        loss_val_2 = self.loss_boundary(v2, target_b)
                        loss_val_3 = self.loss_boundary(v3, target_b)
                        loss_val_4 = self.loss_boundary(v4, target_b)
                        loss_val_6 = self.loss_boundary(v_boundary, target_b)
                        loss_val_7 = self.loss(v_mask, target_m)
                        loss_val_7_s = 1 - self.loss_mask(v_mask, target_m)
                        if self.num_stage==6 : 
                            loss_val_5 = self.loss_boundary(v5, target_b)
                            loss_val_8 = self.loss_boundary(v6, target_b)
                            if epoch < 2 :
                                loss_boundary = (loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4 + loss_val_5+loss_val_8) + (loss_val_6)
                                loss_mask = (loss_val_7 + loss_val_7_s)/2
                                loss = 5*(loss_boundary + loss_mask)
                            else : 
                                loss_boundary = (loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4 + loss_val_5+loss_val_8)/6 + 5*(loss_val_6)
                                loss_mask = (loss_val_7 + loss_val_7_s)
                                loss = loss_boundary + loss_mask
                                
                        elif self.num_stage==5 : 
                            loss_val_5 = self.loss_boundary(v5, target_b)
                            if epoch < 2 :
                                loss_boundary = (loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4 + loss_val_5) + (loss_val_6)
                                loss_mask = (loss_val_7 + loss_val_7_s)/2
                                loss = 5*(loss_boundary + loss_mask)
                            else : 
                                loss_boundary = (loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4 + loss_val_5)/5 + 5*(loss_val_6)
                                loss_mask = (loss_val_7 + loss_val_7_s)
                                loss = loss_boundary + loss_mask
                        elif self.num_stage == 4:
                            if epoch < 2 :
                                loss_boundary = (loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4) + (loss_val_6)
                                loss_mask = (loss_val_7 + loss_val_7_s)/2
                                loss = 5*(loss_boundary + loss_mask)
                            else : 
                                loss_boundary = (loss_val_1 + loss_val_2 + loss_val_3 + loss_val_4)/4 + 5*(loss_val_6)
                                loss_mask = (loss_val_7 + loss_val_7_s)
                                loss = loss_boundary + loss_mask
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
                    cb(self.model,self.weight_path,  loss_value=loss)
                elif cb.monitor == 'val_loss':
                    cb(self.model, self.weight_path,loss_value=val_loss)
                elif cb.monitor == 'periodic':
                    # no loss_value specification needed; defaults to `loss`
                    # cb(self.model, loss_value=loss)
                    cb(self.model,self.weight_path)

        return True

    def save_model(self):
        """Save the final model output."""
        if self.framework == 'keras':
            self.model.save(self.config['training']['model_dest_path'])
        elif self.framework == 'torch':
            if isinstance(self.model, torch.nn.DataParallel):
                torch.save(self.model.module.state_dict(), self.weight_path+'final.pth')
            else:
                torch.save(self.model.state_dict(), self.weight_path+'final.pth')


def get_train_val_dfs(config):
    aoi = config["get_aoi"]
    train_dir = config["train_csv_dir"]
    train_df = pd.read_csv(train_dir+aoi+'_Train_df.csv')
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
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
