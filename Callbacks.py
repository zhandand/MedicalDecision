from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def loadCallbacks(config):
    callbacks = {}
    for callback in config:
        if callback == 'ModelCheckpoint':
            callbacks[callback] = load_ModelCheckpoint(config[callback])
        elif callback == 'EarlyStopping':
            callbacks[callback] = load_EarlyStopping(config[callback])
        else:
            raise NameError(callback + " is not defined")
    return callbacks

def load_ModelCheckpoint(config):
    return ModelCheckpoint(
                dirpath = config['dirpath'],
                monitor = config['monitor'],
                mode = config['mode'],
            );

def load_EarlyStopping(config):
    return EarlyStopping(
                monitor = config['monitor'],
                min_delta = config['min_delta'],
                patience = config['patience'],
                verbose = config['verbose'],
                mode = config['mode'],
            )