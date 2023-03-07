import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
import torchmetrics
from torch.nn import Softmax
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torchvision


class ResnetModel(LightningModule):
    '''
    Resnet Model Class including the training, validation and testing steps
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
        # output dimension is adapted from simCLR
        self.resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT)  # output features are 1000
        # change resnet fc output to 128 features
        self.resnet.fc = nn.Linear(512, self.num_classes)

        # track precision and recall
        self.train_precision = torchmetrics.Precision(
            task="multiclass", average='macro', num_classes=self.num_classes, top_k=1)
        self.val_precision = torchmetrics.Precision(
            task="multiclass", average='macro', num_classes=self.num_classes, top_k=1)
        self.test_precision = torchmetrics.Precision(
            task="multiclass", average='macro', num_classes=self.num_classes, top_k=1)

        self.train_recall = torchmetrics.Recall(
            task="multiclass", average='macro', num_classes=self.num_classes, top_k=1)
        self.val_recall = torchmetrics.Recall(
            task="multiclass", average='macro', num_classes=self.num_classes, top_k=1)
        self.test_recall = torchmetrics.Recall(
            task="multiclass", average='macro', num_classes=self.num_classes, top_k=1)
        # track F1 score
        self.train_F1 = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, top_k=1)
        self.val_F1 = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, top_k=1)
        self.test_F1 = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, top_k=1)

        # track accuracy
        self.train_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=7, top_k=1)
        self.val_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=7, top_k=1)
        self.test_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=7, top_k=1)

        self.train_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=7, top_k=1)
        self.val_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=7, top_k=1)
        self.test_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=7, top_k=1)

        self.softmax = Softmax(dim=1)

    def forward(self, x):
        """
        x is the input image data
        """
        x = self.resnet(x)

        return x

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

        y_pred = self(img)

        loss_func = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_func(y_pred, y)
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)

        # calculate acc
        # take softmax
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
        y_pred_softmax = self.softmax(y_pred)

        # get the index of max value
        pred_label = torch.argmax(y_pred_softmax, dim=1)

        # calculate and log accuracy
        self.train_macro_accuracy(pred_label, y)
        self.train_micro_accuracy(pred_label, y)
        self.train_F1(pred_label, y)
        self.train_precision(pred_label, y)
        self.train_recall(pred_label, y)

        # log metrics
        self.log('train_macro_acc', self.train_macro_accuracy,
                 on_epoch=True, on_step=False)
        self.log('train_micro_acc', self.train_micro_accuracy,
                 on_epoch=True, on_step=False)
        self.log('train_F1', self.train_F1,
                 on_epoch=True, on_step=False)
        self.log('train_precision', self.train_precision,
                 on_epoch=True, on_step=False)
        self.log('train_recall', self.train_recall,
                 on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):

        img, tab, y = batch

        y_pred = self(img)

        loss_func = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_func(y_pred, y)
        self.log('val_epoch_loss', loss, on_epoch=True, on_step=False)

        # calculate acc
        # take softmax
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
        y_pred_softmax = self.softmax(y_pred)

        # get the index of max value
        pred_label = torch.argmax(y_pred_softmax, dim=1)

        # calculate and log accuracy
        self.val_macro_accuracy(pred_label, y)
        self.val_micro_accuracy(pred_label, y)
        self.val_F1(pred_label, y)
        self.val_precision(pred_label, y)
        self.val_recall(pred_label, y)

        # log metrics
        self.log('val_macro_acc', self.val_macro_accuracy,
                 on_epoch=True, on_step=False)
        self.log('val_micro_acc', self.val_micro_accuracy,
                 on_epoch=True, on_step=False)
        self.log('val_F1', self.val_F1,
                 on_epoch=True, on_step=False)
        self.log('val_precision', self.val_precision,
                 on_epoch=True, on_step=False)
        self.log('val_recall', self.val_recall,
                 on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):

        img, tab, y = batch
        y_pred = self(img)

        loss_func = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_func(y_pred, y)
        self.log('test_epoch_loss', loss, on_epoch=True, on_step=False)

        # calculate acc
        # take softmax
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
        y_pred_softmax = self.softmax(y_pred)

        # get the index of max value
        pred_label = torch.argmax(y_pred_softmax, dim=1)

        # calculate and log accuracy
        self.test_macro_accuracy(pred_label, y)
        self.test_micro_accuracy(pred_label, y)
        self.test_F1(pred_label, y)
        self.test_precision(pred_label, y)
        self.test_recall(pred_label, y)

        # log metrics
        self.log('test_macro_acc', self.test_macro_accuracy,
                 on_epoch=True, on_step=False)
        self.log('test_micro_acc', self.test_micro_accuracy,
                 on_epoch=True, on_step=False)
        self.log('test_F1', self.test_F1,
                 on_epoch=True, on_step=False)
        self.log('test_precision', self.test_precision,
                 on_epoch=True, on_step=False)
        self.log('test_recall', self.test_recall,
                 on_epoch=True, on_step=False)

        return loss
