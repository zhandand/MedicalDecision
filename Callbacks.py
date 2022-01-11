from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def loadCallbacks(config):
    callbacks = {}
    for callback in config:
        if callback == 'ModelCheckpoint':
            callbacks[callback] = ModelCheckpoint(** config[callback])
        elif callback == 'EarlyStopping':
            callbacks[callback] = EarlyStopping(** config[callback])
        else:
            raise NameError(callback + " is not defined")
    return callbacks
