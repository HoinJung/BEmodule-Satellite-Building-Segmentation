import yaml
from nets import zoo


def parse(path):

    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        f.close()
    
    if not config['train'] and not config['infer']:
        raise ValueError('"train", "infer", or both must be true.')
    if config['train'] and config['train_csv_dir'] is None:
        raise ValueError('"train_csv_dir" must be provided if training.')
    if config['infer'] and config['inference_data_csv'] is None:
        raise ValueError('"inference_csv_dir" must be provided if "infer".')
    
    train_aoi = config['aoi']
    
    """ Custom AOI """
    if train_aoi == 2:
        aoi = 'AOI_2_Vegas'
    elif train_aoi == 3:
        aoi = 'AOI_3_Paris'
    elif train_aoi == 4:
        aoi = 'AOI_4_Shanghai'
    elif train_aoi == 5:
        aoi = 'AOI_5_Khartoum'
    elif train_aoi == 6:
        aoi = 'Urban3D'
    elif train_aoi == 7:
        aoi = 'WHU'
    elif train_aoi == 8:
        aoi = 'mass'
    elif train_aoi == 9:
        aoi = 'WHU_asia'        
    config['get_aoi'] = aoi

    if config['training']['lr'] is not None:
        config['training']['lr'] = float(config['training']['lr'])


    if config['validation_augmentation'] is not None \
            and config['inference_augmentation'] is None:
        config['inference_augmentation'] = config['validation_augmentation']

    return config
