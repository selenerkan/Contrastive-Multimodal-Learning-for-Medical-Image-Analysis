import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
import torchmetrics
from torch.nn import Softmax
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torchvision
import pandas as pd


class TabularModel(LightningModule):
    '''
    Tabular Model Class including the training, validation and testing steps
    '''

    def __init__(self, learning_rate=0.013, weight_decay=0.01):

        super().__init__()
        self.register_buffer('class_weights', torch.tensor([1.5565749235474007,
                                                           1.0,
                                                           0.47304832713754646,
                                                           4.426086956521739,
                                                           0.4614687216681777,
                                                           0.0783197414986921,
                                                           3.584507042253521]))

        self.save_hyperparameters()

        self.lr = learning_rate
        self.wd = weight_decay
        self.num_classes = 7

        # IMAGE DATA
        # fc layer for tabular data
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.num_classes)

        # track AUC
        self.train_auc = torchmetrics.AUROC(
            task="multiclass", num_classes=self.num_classes)
        self.val_auc = torchmetrics.AUROC(
            task="multiclass", num_classes=self.num_classes)
        # track precision and recall
        self.train_precision = torchmetrics.Precision(
            task="multiclass", average='macro', num_classes=self.num_classes, top_k=1)
        self.val_precision = torchmetrics.Precision(
            task="multiclass", average='macro', num_classes=self.num_classes, top_k=1)

        self.train_recall = torchmetrics.Recall(
            task="multiclass", average='macro', num_classes=self.num_classes, top_k=1)
        self.val_recall = torchmetrics.Recall(
            task="multiclass", average='macro', num_classes=self.num_classes, top_k=1)
        # track F1 score
        self.train_F1 = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, top_k=1)
        self.val_F1 = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, top_k=1)
        # track accuracy
        self.train_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=self.num_classes, top_k=1)
        self.val_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=self.num_classes, top_k=1)

        self.train_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=self.num_classes, top_k=1)
        self.val_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=self.num_classes, top_k=1)

        self.train_acc = 0
        self.total_samples = 0

        self.softmax = Softmax(dim=1)

        # self.register_buffer('records_df',pd.DataFrame(columns=['prediction','label']))

    def forward(self, x):
        """
        x is the input tabular data
        """

        x = x.to(torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd)
        # scheduler = MultiStepLR(optimizer,
        #                         # List of epoch indices
        #                         milestones=[30, 60],
        #                         gamma=0.1)  # Multiplicative factor of learning rate decay

        # return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):

        img, tab, y = batch

        y_pred = self(tab)

        loss_func = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_func(y_pred, y)
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)

        # record auc
        train_auc = self.train_auc(y_pred, y)
        self.log('train_auc', train_auc,
                 on_epoch=True, on_step=False)
        # calculate acc
        # take softmax
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
        y_pred_softmax = self.softmax(y_pred)

        # get the index of max value
        pred_label = torch.argmax(y_pred_softmax, dim=1)

        # calculate and log accuracy
        train_acc = self.train_macro_accuracy(pred_label, y)
        self.log('train_macro_acc', train_acc, on_epoch=True, on_step=False)

        # calculate and log accuracy
        train_micro_acc = self.train_micro_accuracy(pred_label, y)
        self.log('train_micro_acc', train_micro_acc,
                 on_epoch=True, on_step=False)

        # record f1 score
        train_f1_score = self.train_F1(pred_label, y)
        self.log('train_F1', train_f1_score,
                 on_epoch=True, on_step=False)

        # record precision score
        train_precision = self.train_precision(pred_label, y)
        self.log('train_precision', train_precision,
                 on_epoch=True, on_step=False)

        # record recall score
        train_recall = self.train_recall(pred_label, y)
        self.log('train_recall', train_recall,
                 on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):

        img, tab, y = batch

        y_pred = self(tab)

        loss_func = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_func(y_pred, y)
        self.log('val_epoch_loss', loss, on_epoch=True, on_step=False)

        # record auc
        val_auc = self.val_auc(y_pred, y)
        self.log('val_auc', val_auc,
                 on_epoch=True, on_step=False)

        # calculate acc
        # take softmax
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
        y_pred_softmax = self.softmax(y_pred)

        # get the index of max value
        pred_label = torch.argmax(y_pred_softmax, dim=1)

        # calculate and log accuracy
        val_acc = self.val_macro_accuracy(pred_label, y)
        self.log('val_macro_acc', val_acc, on_epoch=True, on_step=False)

        # calculate and log accuracy
        val_micro_acc = self.val_micro_accuracy(pred_label, y)
        self.log('val_micro_acc', val_micro_acc, on_epoch=True, on_step=False)

        # record f1 score
        val_f1_score = self.val_F1(pred_label, y)
        self.log('train_F1', val_f1_score,
                 on_epoch=True, on_step=False)

        # record precision score
        val_precision = self.val_precision(pred_label, y)
        self.log('val_precision', val_precision,
                 on_epoch=True, on_step=False)

        # record recall score
        val_recall = self.val_recall(pred_label, y)
        self.log('val_recall', val_recall,
                 on_epoch=True, on_step=False)

        # Record all the predictions
        records = {'prediction': pred_label.cpu(), 'label': y.cpu(),
                   'epoch': self.current_epoch}
        df = pd.DataFrame(data=records)
        df.to_csv('result_tabular.csv', mode='a', index=False, header=False)
        return loss

    def test_step(self, batch, batch_idx):

        img, tab, y = batch
        y_pred = self(tab)

        loss_func = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_func(y_pred, y)

        self.log("test_loss", loss)

        return loss
