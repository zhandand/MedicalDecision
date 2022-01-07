import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import SGD, Adam


class MedDec(pl.LightningModule):
    def __init__(self, *args):
        super().__init__()
        self.kwargs = args[0]
        self.resource = args[1]
        self.model_name = self.kwargs['model']['name']
        exec('from model.' + self.model_name + " import "+self.model_name)
        self.model = eval(self.model_name)(self.kwargs['model'], self.resource)
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
        y_hat, y = self.train_in_model(batch, batch_idx)
        if self.kwargs['model']['criterion'] == 'CE' or self.kwargs['model']['criterion'] == 'MSE':
            # loss = self.criterion()(y_hat, y.float())
            loss = self.criterion()(y_hat, y)
        # elif self.kwargs['model']['criterion'] == 'InfoNCE':
        #     loss = self.criterion()(y_hat, self.kwargs['model']['t'],y,self.resource['pmd'][x,y])
        elif self.kwargs['model']['criterion'] == 'NCESoftmaxLoss':
            loss = self.criterion()(y_hat, y)
        return {"loss": loss,
                "pred": y_hat.detach(),
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
        y_hat, y = self.train_in_model(batch, batch_idx)
        if self.kwargs['model']['criterion'] == 'CE' or self.kwargs['model']['criterion'] == 'MSE':
            # loss = self.criterion()(y_hat, y.float())
            loss = self.criterion()(y_hat, y)
        # elif self.kwargs['model']['criterion'] == 'InfoNCE':
        #     loss = self.criterion()(y_hat, self.kwargs['model']['t'],y,self.resource['pmd'][x,y])
        elif self.kwargs['model']['criterion'] == 'NCESoftmaxLoss':
            loss = self.criterion()(y_hat, y)
        # self.log("val_loss", loss)
        return {"loss": loss,
                "pred": y_hat.detach(),
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

        self.log_metrics("val_", ** metrics)
        if 'ModelCheckpoint' in self.kwargs['Callbacks']:
            monitor = self.kwargs['Callbacks']['ModelCheckpoint']['monitor']
            if 'train' not in monitor:
                self.log(monitor, metrics[monitor], on_step=False,
                         on_epoch=True, prog_bar=True, logger=True)
        # self.log("val_loss", avg_loss, on_step=False,
        #          on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion()(y_hat, y.float())
        return {"loss": loss,
                "pred": y_hat,
                'label': y}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        preds = torch.stack([x['pred'] for x in outputs]
                            ).squeeze().data.cpu().numpy()
        labels = torch.stack([x['label'] for x in outputs]
                             ).squeeze().data.cpu().numpy()

        self.save_preds(preds, labels)
        mlflow.log_metric(key="test_loss", value=avg_loss)
        metrics = self.calMetric(preds, labels)
        self.log_metrics("", **metrics)

    def criterion(self):
        if self.kwargs['model']['criterion'] == 'CE':
            from torch.nn.functional import binary_cross_entropy
            return binary_cross_entropy
        elif self.kwargs['model']['criterion'] == 'MSE':
            from torch.nn.functional import mse_loss
            return mse_loss
        elif self.kwargs['model']['criterion'] == 'InfoNCE':
            from loss import InfoNCE
            return InfoNCE
        elif self.kwargs['model']['criterion'] == 'NCESoftmaxLoss':
            from loss import NCESoftmaxLoss
            return NCESoftmaxLoss
        else:
            raise NotImplementedError('This loss not defined')

    def calMetric(self, preds, labels):
        metrics = {}
        for metric in self.kwargs['model']['metric']:
            if metric == 'f1':
                from metric import average_f1
                metrics[metric] = average_f1(
                    labels, np.rint(preds))    # 计算f1需要threshold取整
            if metric == 'jaccard':
                from metric import jaccard
                metrics[metric] = jaccard(labels, np.rint(
                    preds))       # 计算jaccard需要threshold取整
            if metric == 'PRAUC':
                from metric import precision_auc
                metrics[metric] = precision_auc(labels, preds)
        return metrics
        # mlflow.log_metrics(metrics)

    def configure_optimizers(self):
        self.opimizer_config = self.kwargs['optimizer']
        if 'Adam' in self.opimizer_config.keys():
            # self.lr = self.opimizer_config['lr']
            # self.weight_decay = self.opimizer_config['weight_decay']
            return Adam(self.model.parameters(), **self.kwargs['optimizer']['Adam'])
        elif 'SGD' in self.opimizer_config['type']:
            # self.lr = self.opimizer_config['lr']
            # self.weight_decay = self.opimizer_config['weight_decay']
            # self.momentum = self.opimizer_config['momentum']
            return SGD(self.model.parameters(), **self.kwargs['optimizer']['SGD'])

    def log_metrics(self, prefix: str, **metrics: dict):
        for metric in metrics.keys():
            mlflow.log_metric(key=prefix + metric,
                              value=metrics[metric], step=self.current_epoch)

    def round(self, data, threshold):
        """计算 f1, jaccard等指标时需要取整

        Args:
            data ([type]): 需要取整的数据
            threshold ([type]): 阈值

        Returns:
            [type]: 取整后的数据
        """
        data[data >= threshold] = 1
        data[data < threshold] = 0
        return data

    def save_preds(self, preds, labels):
        np.savez(self.kwargs['run_path']+'PREDS', preds=preds, labels=labels)

    def train_in_model(self, batch, batch_idx):
        # 由于不同的模型的batch和输入不同，为了统一接口，在此函数中进行逻辑处理
        if self.model_name == "GCC":
            graph_q, graph_k = batch
            feat_q = self.model(graph_q, graph_q.ndata['feats'])
            feat_k = self.model(graph_k, graph_k.ndata['feats'])
            return feat_q, feat_k
        else:
            x, y = batch
            y_hat = self.model(x)
            return y_hat, y
