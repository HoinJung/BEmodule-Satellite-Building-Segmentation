import yaml
from nets import zoo


def parse(path):
    """Parse a config file for running a model.

    Arguments
    ---------
    path : str
        Path to the YAML-formatted config file to parse.

    Returns
    -------
    config : dict
        A `dict` containing the information from the config file at `path`.

    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        f.close()
    ##aoi 선택을 위해 아래 내용 삭제(변경)함 .. 추후 변경 필요. train.py line 280에 aoi 관련 내용 삽입
    if not config['train'] and not config['infer']:
        raise ValueError('"train", "infer", or both must be true.')
    if config['train'] and config['train_csv_dir'] is None:
        raise ValueError('"train_csv_dir" must be provided if training.')
    if config['infer'] and config['inference_data_csv'] is None:
        raise ValueError('"inference_csv_dir" must be provided if "infer".')
    
    train_aoi = config['aoi']
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
    config['get_aoi'] = aoi
    config['training']['callbacks']['model_checkpoint']['path_aoi'] = aoi
        
#Learning Rate 변경
#원본
#    if config['training']['lr'] is not None:
#        config['training']['lr'] = float(config['training']['lr'])

#for DHDN
    if config['training']['lr'] is not None:
        config['training']['lr'] = float(config['training']['lr'])

    # TODO: IMPLEMENT UPDATING VALUES BASED ON EMPTY ELEMENTS HERE!

    if config['validation_augmentation'] is not None \
            and config['inference_augmentation'] is None:
        config['inference_augmentation'] = config['validation_augmentation']

    return config
