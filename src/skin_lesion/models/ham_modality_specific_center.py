import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from pytorch_metric_learning import losses
import torchmetrics
from torch.nn import Softmax
from center_loss import CenterLoss
import torchvision
from adversarial_loss import AdversarialLoss
import pandas as pd


class ModalityCenterModel(LightningModule):
    '''
    Uses ResNet for the image data, concatenates image and tabular data at the end
    '''

    def __init__(self, seed, learning_rate=0.013, weight_decay=0.01, alpha_center=0.01,  dropout_rate=0):

        super().__init__()
        self.use_gpu = False
        if torch.cuda.is_available():
            self.use_gpu = True

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
        self.dr = dropout_rate
        # weights of the losses
        self.alpha_center = alpha_center
        self.alpha_cross_ent = (1-alpha_center)

        # parameters for center loss
        self.num_classes = 7
        self.feature_dim = 32
        self.embedding_dimension = 64

        # IMAGE DATA
        self.resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT)  # output features are 1000
        # change resnet fc output to 128 features
        self.resnet.fc = nn.Linear(512, 128)

        # TABULAR DATA
        # fc layer for tabular data
        self.fc1 = nn.Linear(16, 128)  # tabular data is one-hot-encoded
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)

        # shared FC layer
        self.fc6 = nn.Linear(128, self.embedding_dimension)

        # TABULAR + IMAGE DATA
        # mlp projection head which takes concatenated input
        concatanation_dimension = 128
        # outputs will be used in triplet/center loss
        self.fc7 = nn.Linear(concatanation_dimension, self.feature_dim)
        self.fc8 = nn.Linear(32, 7)  # classification head

        # initiate losses
        self.center_loss_function_img = CenterLoss(
            num_classes=self.num_classes, feat_dim=self.embedding_dimension, use_gpu=self.use_gpu, seed=seed)
        self.center_loss_function_tab = CenterLoss(
            num_classes=self.num_classes, feat_dim=self.embedding_dimension, use_gpu=self.use_gpu, seed=seed)
        self.cross_ent_loss_function = nn.CrossEntropyLoss(
            weight=self.class_weights)

        # add dropout
        self.dropout = nn.Dropout(p=self.dr)

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

    def forward(self, img, tab):
        """

        img is the input image data ()
        tab is th einput tabular data

        """
        # run the model for the image
        img = self.resnet(img)
        img = self.fc6(F.relu(img))

        # forward pass for tabular data
        tab = tab.to(torch.float32)
        tab = F.relu(self.fc1(tab))
        tab = F.relu(self.fc2(tab))
        tab = F.relu(self.fc3(tab))
        tab = F.relu(self.fc4(tab))
        tab = F.relu(self.fc5(tab))
        tab = self.fc6(tab)

        # concat image and tabular data
        x = torch.cat((img, tab), dim=1)

        # get the final concatenated embedding
        x = self.fc7(x)
        # calculate the output of classification head
        x = self.fc8(F.relu(x))

        return img, tab, x

    def configure_optimizers(self):
        my_list = ['center_loss_function_img.centers',
                   'center_loss_function_tab.centers']
        center_params = list(
            filter(lambda kv: kv[0] in my_list, self.named_parameters()))
        model_params = list(
            filter(lambda kv: kv[0] not in my_list, self.named_parameters()))

        optimizer = torch.optim.Adam([
            {'params': [temp[1] for temp in model_params]},
            {'params': center_params[0][1], 'lr': 1e-4},
            {'params': center_params[1][1], 'lr': 1e-4}
        ], lr=self.lr, weight_decay=self.wd)

        return optimizer

    def training_step(self, batch, batch_idx):

        img, tab, y = batch
        embed_img, embed_tab, y_pred = self(img, tab)

        cross_ent_loss = self.cross_ent_loss_function(y_pred, y.squeeze())
        # center loss
        center_loss_img = self.center_loss_function_img(embed_img, y)
        center_loss_tab = self.center_loss_function_tab(embed_tab, y)
        # sum the losses
        loss = self.alpha_cross_ent*cross_ent_loss + \
            (self.alpha_center/2) * center_loss_img + \
            (self.alpha_center/2) * center_loss_tab

        # Log loss on every epoch
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)
        self.log('train_center_loss_img', center_loss_img,
                 on_epoch=True, on_step=False)
        self.log('train_center_loss_tab ', center_loss_tab,
                 on_epoch=True, on_step=False)
        self.log('train_cross_ent_loss', cross_ent_loss,
                 on_epoch=True, on_step=False)
        # log weights
        self.log('cross_ent_weight', self.alpha_cross_ent,
                 on_epoch=True, on_step=False)
        self.log('center_weight', self.alpha_center,
                 on_epoch=True, on_step=False)

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

        # log the metrics
        self.log('train_macro_acc',
                 self.train_macro_accuracy,
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
        embed_img, embed_tab, y_pred = self(img, tab)

        cross_ent_loss = self.cross_ent_loss_function(y_pred, y.squeeze())
        # center loss
        center_loss_img = self.center_loss_function_img(embed_img, y)
        center_loss_tab = self.center_loss_function_tab(embed_tab, y)
        # sum the losses
        loss = self.alpha_cross_ent*cross_ent_loss + \
            (self.alpha_center/2) * center_loss_img + \
            (self.alpha_center/2) * center_loss_tab

        # Log loss on every epoch
        self.log('val_epoch_loss', loss, on_epoch=True, on_step=False)
        self.log('val_center_loss_img', center_loss_img,
                 on_epoch=True, on_step=False)
        self.log('val_center_loss_tab', center_loss_tab,
                 on_epoch=True, on_step=False)
        self.log('val_cross_ent_loss', cross_ent_loss,
                 on_epoch=True, on_step=False)

        # calculate acc
        # take softmax
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
        y_pred_softmax = self.softmax(y_pred)
        # y_pred_softmax = torch.sigmoid(y_pred)

        # get the index of max value
        pred_label = torch.argmax(y_pred_softmax, dim=1)

        # calculate and log accuracy
        self.val_macro_accuracy(pred_label, y)
        self.val_micro_accuracy(pred_label, y)
        self.val_F1(pred_label, y)
        self.val_precision(pred_label, y)
        self.val_recall(pred_label, y)

        # log the metrics
        self.log('val_macro_acc',
                 self.val_macro_accuracy,
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
        embed_img, embed_tab, y_pred = self(img, tab)

        cross_ent_loss = self.cross_ent_loss_function(y_pred, y.squeeze())
        # center loss
        center_loss_img = self.center_loss_function_img(embed_img, y)
        center_loss_tab = self.center_loss_function_tab(embed_tab, y)
        # sum the losses
        loss = self.alpha_cross_ent*cross_ent_loss + \
            (self.alpha_center/2) * center_loss_img + \
            (self.alpha_center/2) * center_loss_tab

        # Log loss on every epoch
        self.log('test_epoch_loss', loss, on_epoch=True, on_step=False)
        self.log('test_center_loss_img', center_loss_img,
                 on_epoch=True, on_step=False)
        self.log('test_center_loss_tab', center_loss_tab,
                 on_epoch=True, on_step=False)
        self.log('test_cross_ent_loss', cross_ent_loss,
                 on_epoch=True, on_step=False)

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

        self.log('test_macro_acc',
                 self.test_macro_accuracy,
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
