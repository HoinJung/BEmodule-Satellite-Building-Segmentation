
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
from .transform import _check_augs, process_aug_dict
from utils.core import _check_df_load
from utils.io import imread, _check_channel_order
import skimage
import skimage.segmentation
def make_data_generator(framework, config, df, stage='train'):

    if framework.lower() not in ['pytorch', 'torch']:
        raise ValueError('{} is not an accepted value for `framework`'.format(
            framework))

    # make sure the df is loaded
    df = _check_df_load(df)

    if stage == 'train':
        augs = config['training_augmentation']
        shuffle = config['training_augmentation']['shuffle']
    elif stage == 'validate':
        augs = config['validation_augmentation']
        shuffle = False
    try:
        num_classes = config['data_specs']['num_classes']
    except KeyError:
        num_classes = 1

   
    if framework in ['torch', 'pytorch']:
        dataset = TorchDataset(
            df,
            augs=augs,
            batch_size=config['batch_size'],
            label_type=config['data_specs']['label_type'],
            is_categorical=config['data_specs']['is_categorical'],
            num_classes=num_classes,
            dtype=config['data_specs']['dtype'])
        # set up workers for DataLoader for pytorch
        data_workers = config['data_specs'].get('data_workers')
        if data_workers == 1 or data_workers is None:
            data_workers = 0  # for DataLoader to run in main process
        data_gen = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=config['training_augmentation']['shuffle'],
            num_workers=data_workers,
            drop_last=True)
        
    return data_gen



class TorchDataset(Dataset):

    def __init__(self, df, augs, batch_size, label_type='mask',
                 is_categorical=False, num_classes=1, dtype=None):

        super().__init__()

        self.df = df
        self.batch_size = batch_size
        self.n_batches = int(np.floor(len(self.df)/self.batch_size))
        self.aug = _check_augs(augs)
        self.is_categorical = is_categorical
        self.num_classes = num_classes

        if dtype is None:
            self.dtype = np.float32  # default
        # if it's a string, get the appropriate object
        elif isinstance(dtype, str):
            try:
                self.dtype = getattr(np, dtype)
            except AttributeError:
                raise ValueError(
                    'The data type {} is not supported'.format(dtype))
        # lastly, check if it's already defined in the right format for use
        elif issubclass(dtype, np.number) or isinstance(dtype, np.dtype):
            self.dtype = dtype

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get one image, mask pair"""
        # Generate indexes of the batch
        image = imread(self.df['image'].iloc[idx])
        mask = imread(self.df['label'].iloc[idx])
        boundary = mask

        if not self.is_categorical:
            mask[mask != 0] = 1
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        if len(boundary.shape) == 2:
            boundary = boundary[:, :, np.newaxis]
        
        sample = {'image': image, 'mask': mask, 'boundary' : boundary}

        if self.aug:
            sample = self.aug(**sample)
        


        sample['image'] = _check_channel_order(sample['image'],
                                               'torch').astype(self.dtype)
        sample['mask'] = _check_channel_order(sample['mask'],
                                              'torch').astype(np.float32)

        sample['boundary'] = _check_channel_order(skimage.segmentation.find_boundaries(sample['mask'], mode='inner', background=0),
                                              'torch').astype(np.float32)

        return sample


class InferenceTiler(object):


    def __init__(self, framework, width, height, x_step=None, y_step=None,
                 augmentations=None):

        self.framework = framework
        self.width = width
        self.height = height
        if x_step is None:
            self.x_step = self.width
        else:
            self.x_step = x_step
        if y_step is None:
            self.y_step = self.height
        else:
            self.y_step = y_step
        self.aug = _check_augs(augmentations)

    def __call__(self, im):
 
        # read in the image if it's a path
        if isinstance(im, str):
            im = imread(im)

        # determine how many samples will be generated with the sliding window
        src_im_height = im.shape[0]
        src_im_width = im.shape[1]
        

        
        y_steps = int(1+np.ceil((src_im_height-self.height)/self.y_step))
        x_steps = int(1+np.ceil((src_im_width-self.width)/self.x_step))
        if len(im.shape) == 2:  # if there's no channel axis
            im = im[:, :, np.newaxis]  # create one - will be needed for model
        top_left_corner_idxs = []
        output_arr = []
        for y in range(y_steps):
            if self.y_step*y + self.height > im.shape[0]:
                y_min = im.shape[0] - self.height
            else:
                y_min = self.y_step*y

            for x in range(x_steps):
                if self.x_step*x + self.width > im.shape[1]:
                    x_min = im.shape[1] - self.width
                else:
                    x_min = self.x_step*x

                subarr = im[y_min:y_min + self.height,
                            x_min:x_min + self.width,
                            :]
                if self.aug is not None:
                    subarr = self.aug(image=subarr)['image']
                output_arr.append(subarr)
                top_left_corner_idxs.append((y_min, x_min))
        output_arr = np.stack(output_arr).astype(np.float32)
        if self.framework in ['torch', 'pytorch']:
            output_arr = np.moveaxis(output_arr, 3, 1)
        

        return output_arr, top_left_corner_idxs, (src_im_height, src_im_width)
