from cubercnn.data import simple_register, get_filter_settings_from_cfg

def register_rhino_datasets(cfg):
    filter_settings = get_filter_settings_from_cfg(cfg)
    for dataset_name in ['RHINO_train', 'RHINO_val', 'RHINO_test']:
        simple_register(dataset_name, filter_settings, filter_empty=True)