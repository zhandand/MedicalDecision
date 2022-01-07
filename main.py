import argparse
import os

import mlflow
import pytorch_lightning as pl

from pipeine import train_valid_pipeline,test_pipeline, train_pipeline
from utils import mk_rundir, read_config, save_config

pipeline_dict = {
    'train' : train_pipeline,
    'test' : test_pipeline,
    'train_valid' : train_valid_pipeline
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./config/example.yaml",
                        help="config path for experiment")
    args = parser.parse_args()
    config = read_config(args.config_path)

    config['run_path'] = mk_rundir(config['output'] +config['mlflow']['experiment'],config['mlflow']['run_name'])
    config['Callbacks']['ModelCheckpoint']['dirpath'] = config['run_path']
    config['data']['seed'] = config['seed']
    print(config)

    mlflow.set_experiment(config['mlflow']["experiment"])

    for pipeline in config['pipeline']:
        with mlflow.start_run():
            mlflow.set_tag("Run Name", config['mlflow']["run_name"])
            mlflow.set_tag("pipeline", pipeline)
            pl.seed_everything(config['seed'], workers=True)
            pipeline_dict[pipeline](**config)
            mlflow.log_param("path", config['run_path'])
    save_config(os.path.join(config['run_path'] , 'config.yaml'),config)
