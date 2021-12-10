import mlflow
import pytorch_lightning as pl
import torch
from torch.optim import SGD, Adam
from utils import to_device

class MedDec(pl.LightningModule):
    def __init__(self, *args):
        super().__init__()
        self.kwargs = args[0]
        self.resource = args[1]
        self.model_name = self.kwargs['model']['name']
        exec('from model.' + self.model_name + " import "+self.model_name)
        self.model = eval(self.model_name)(self.kwargs['model'],self.resource)
        self.automatic_optimization = self.kwargs['automatic_optimization']
        print("init model done...")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """during every batch

        Args:
            batch ([type]): [description]
            batch_idx ([type]): [description]
        """
        x, y = batch
        y_hat = self.model(x)
        if self.kwargs['model']['criterion'] == 'CE' or self.kwargs['model']['criterion'] == 'MSE':
            loss = self.criterion()(y_hat, y)
        elif self.kwargs['model']['criterion'] == 'InfoNCE':
            loss = self.criterion()(y_hat, self.kwargs['model']['t'],y,self.resource['pmd'][x,y])

        return {"loss": loss,
                "pred": y_hat,
                'label': y}

    def training_epoch_end(self, outputs):
        """during every epoch

        Args:
            losses ([type]): [description]
        """
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        preds = torch.stack([x['pred'] for x in outputs]
                            ).squeeze().data.cpu().numpy()
        labels = torch.stack([x['label'] for x in outputs]
                             ).squeeze().data.cpu().numpy()

        mlflow.log_metric(key="train_loss", value=avg_loss,
                          step=self.current_epoch)
        self.log("train_loss", avg_loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        # self.calMetric(preds,labels)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        if self.kwargs['model']['criterion'] == 'CE' or self.kwargs['model']['criterion'] == 'MSE':
            loss = self.criterion()(y_hat, y)
        elif self.kwargs['model']['criterion'] == 'InfoNCE':
            loss = self.criterion()(y_hat, self.kwargs['model']['t'],y,self.resource['pmd'][x,y]) 
        # self.log("val_loss", loss)
        return {"loss": loss,
                "pred": y_hat,
                'label': y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        preds = torch.stack([x['pred'] for x in outputs]
                            ).squeeze().data.cpu().numpy()
        labels = torch.stack([x['label'] for x in outputs]
                             ).squeeze().data.cpu().numpy()
        # mlflow.log_metric(key="val_loss", value=avg_loss,
        #                   step=self.current_epoch)
        metrics = self.calMetric(preds, labels)
        metrics['val_loss'] = avg_loss

        self.log_metrics("val_",** metrics)
        if 'ModelCheckpoint' in self.kwargs['Callbacks']:
            monitor = self.kwargs['Callbacks']['ModelCheckpoint']['monitor'] 
            if monitor[:3] == 'val':
                self.log(monitor, metrics[monitor], on_step=False,
                    on_epoch=True, prog_bar=True, logger=True)
        # self.log("val_loss", avg_loss, on_step=False,
        #          on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion()(y_hat, y)
        return {"loss": loss,
                "pred": y_hat,
                'label': y}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        preds = torch.stack([x['pred'] for x in outputs]
                            ).squeeze().data.cpu().numpy()
        labels = torch.stack([x['label'] for x in outputs]
                             ).squeeze().data.cpu().numpy()

        mlflow.log_metric(key="test_loss", value=avg_loss)
        metrics = self.calMetric(preds, labels)
        self.log_metrics("test_", **metrics)

    def criterion(self):
        if self.kwargs['model']['criterion'] == 'CE':
            from torch.nn.functional import cross_entropy
            return cross_entropy
        elif self.kwargs['model']['criterion'] == 'MSE':
            from torch.nn.functional import mse_loss
            return mse_loss
        elif self.kwargs['model']['criterion'] == 'InfoNCE':
            from loss import InfoNCE
            return InfoNCE
        else :
            raise NotImplementedError('This loss not defined')

    def calMetric(self, preds, labels):
        metrics = {}
        for metric in self.kwargs['model']['metric']:
            if metric == 'f1':
                from metric import average_f1
                metrics[metric] = average_f1(labels, preds)
            if metric == 'jaccard':
                from metric import jaccard
                metrics[metric] = jaccard(labels, preds)
            if metric == 'PRAUC':
                from metric import precision_auc
                metrics[metric] = precision_auc(labels, preds)
        return metrics
        # mlflow.log_metrics(metrics)

    def configure_optimizers(self):
        self.opimizer_config = self.kwargs['optimizer']
        if 'Adam' in self.opimizer_config.keys() :
            # self.lr = self.opimizer_config['lr']
            # self.weight_decay = self.opimizer_config['weight_decay']
            return Adam(self.model.parameters(), **self.kwargs['optimizer']['Adam'])
        elif 'SGD' in self.opimizer_config['type'] :
            # self.lr = self.opimizer_config['lr']
            # self.weight_decay = self.opimizer_config['weight_decay']
            # self.momentum = self.opimizer_config['momentum']
            return SGD(self.model.parameters(),**self.kwargs['optimizer']['SGD'])

    def log_metrics(self, prefix:str,**metrics:dict):
        for metric in metrics.keys():
            mlflow.log_metric(key=prefix + metric,
                              value=metrics[metric], step=self.current_epoch)
