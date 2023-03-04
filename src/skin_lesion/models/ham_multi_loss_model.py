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
import pandas as pd


class MultiLossModel(LightningModule):
    '''
    Uses ResNet for the image data, concatenates image and tabular data at the end
    '''

    def __init__(self, learning_rate=0.013, weight_decay=0.01, alpha_center=0.01, triplet_ratio=0.5):

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
        # weights of the losses
        self.alpha_center = alpha_center
        self.alpha_triplet = (1-alpha_center)*triplet_ratio
        self.alpha_cross_ent = (1-alpha_center-self.alpha_triplet)
        # self.alpha_cross_ent = (1-alpha_center)

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
        concatanation_dimension = (self.embedding_dimension * 2) - 1
        # concatanation_dimension = 128
        # outputs will be used in triplet/center loss
        self.fc7 = nn.Linear(concatanation_dimension, self.feature_dim)
        self.fc8 = nn.Linear(32, 7)  # classification head

        # initiate losses
        self.center_loss = CenterLoss(
            num_classes=self.num_classes, feat_dim=self.feature_dim, use_gpu=self.use_gpu)

        # track accuracy
        self.train_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=self.num_classes, top_k=1)
        self.val_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=self.num_classes, top_k=1)

        self.train_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=self.num_classes, top_k=1)
        self.val_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=self.num_classes, top_k=1)

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
        # x = torch.cat((img, tab), dim=1)
        img = img.unsqueeze(0)
        tab = tab.unsqueeze(1)
        x = F.conv1d(img, tab, padding=self.embedding_dimension -
                     1, groups=img.size(1))
        x = x.squeeze()

        # get the final concatenated embedding
        out1 = self.fc7(x)
        # calculate the output of classification head
        out2 = self.fc8(F.relu(out1))

        return out1, out2

    def configure_optimizers(self):
        my_list = ['center_loss.centers']
        center_params = list(
            filter(lambda kv: kv[0] in my_list, self.named_parameters()))
        model_params = list(
            filter(lambda kv: kv[0] not in my_list, self.named_parameters()))

        optimizer = torch.optim.Adam([
            {'params': [temp[1] for temp in model_params]},
            {'params': center_params[0][1], 'lr': 1e-4}
        ], lr=self.lr, weight_decay=self.wd)

        # # UNCOMMENT FOR LR SCHEDULER
        # scheduler = MultiStepLR(optimizer,
        #                         # List of epoch indices
        #                         milestones=[12],
        #                         gamma=0.05)  # Multiplicative factor of learning rate decay

        # return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, positive, negative, tab, positive_tab, negative_tab, y = batch[
            0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

        # TODO: dont feed positive and negatives to classification layer
        embeddings, y_pred = self(img, tab)
        pos_embeddings, _ = self(positive, positive_tab)
        neg_embeddings, _ = self(negative, negative_tab)

        # triplet loss
        triplet_loss_function = nn.TripletMarginLoss()
        triplet_loss = triplet_loss_function(
            embeddings, pos_embeddings, neg_embeddings)
        # cross entropy loss
        cross_ent_loss_function = nn.CrossEntropyLoss(
            weight=self.class_weights)
        cross_ent_loss = cross_ent_loss_function(y_pred, y.squeeze())
        # center loss
        center_loss = self.center_loss(embeddings, y.squeeze())
        # sum the losses
        loss = self.alpha_cross_ent*cross_ent_loss + \
            self.alpha_center * center_loss + self.alpha_triplet*triplet_loss
        # Log loss on every epoch
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)
        self.log('train_center_loss', center_loss,
                 on_epoch=True, on_step=False)
        self.log('train_cross_ent_loss', cross_ent_loss,
                 on_epoch=True, on_step=False)
        self.log('train_triplet_loss', triplet_loss,
                 on_epoch=True, on_step=False)

        # log weights
        self.log('cross_ent_weight', self.alpha_cross_ent,
                 on_epoch=True, on_step=False)
        self.log('center_weight', self.alpha_center,
                 on_epoch=True, on_step=False)
        self.log('triplet_weight', self.alpha_triplet,
                 on_epoch=True, on_step=False)

        # calculate acc
        # take softmax
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
        y_pred_softmax = self.softmax(y_pred)

        # get the index of max value
        pred_label = torch.argmax(y_pred_softmax, dim=1)

        # calculate and log macro accuracy
        train_acc = self.train_macro_accuracy(pred_label, y)
        self.log('train_macro_acc', train_acc, on_epoch=True, on_step=False)

        # calculate and log micro accuracy
        train_micro_acc = self.train_micro_accuracy(pred_label, y)
        self.log('train_micro_acc', train_micro_acc,
                 on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, positive, negative, tab, positive_tab, negative_tab, y = batch[
            0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

        # TODO: dont feed positive and negatives to classification layer
        embeddings, y_pred = self(img, tab)
        pos_embeddings, _ = self(positive, positive_tab)
        neg_embeddings, _ = self(negative, negative_tab)

        # triplet loss
        triplet_loss_function = nn.TripletMarginLoss()
        triplet_loss = triplet_loss_function(
            embeddings, pos_embeddings, neg_embeddings)
        # cross entropy loss
        cross_ent_loss_function = nn.CrossEntropyLoss(
            weight=self.class_weights)
        cross_ent_loss = cross_ent_loss_function(y_pred, y.squeeze())
        # center loss
        center_loss = self.center_loss(embeddings, y.squeeze())
        # sum the losses
        loss = self.alpha_cross_ent*cross_ent_loss + \
            self.alpha_center * center_loss + self.alpha_triplet*triplet_loss

        # Log loss on every epoch
        self.log('val_epoch_loss', loss, on_epoch=True, on_step=False)
        self.log('val_center_loss', center_loss, on_epoch=True, on_step=False)
        self.log('val_cross_ent_loss', cross_ent_loss,
                 on_epoch=True, on_step=False)
        self.log('val_triplet_loss', triplet_loss,
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
        val_acc = self.val_macro_accuracy(pred_label, y)
        self.log('val_macro_acc', val_acc, on_epoch=True, on_step=False)

        # calculate and log accuracy
        val_micro_acc = self.val_micro_accuracy(pred_label, y)
        self.log('val_micro_acc', val_micro_acc, on_epoch=True, on_step=False)

        # Record all the predictions
        records = {'prediction': pred_label.cpu(), 'label': y.cpu(),
                   'epoch': self.current_epoch}
        df = pd.DataFrame(data=records)
        df.to_csv('result_multiloss.csv', mode='a', index=False, header=False)
        return loss

    def test_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, positive, negative, tab, positive_tab, negative_tab, y = batch[
            0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

        # TODO: dont feed positive and negatives to classification layer
        embeddings, y_pred = self(img, tab)
        pos_embeddings, _ = self(positive, positive_tab)
        neg_embeddings, _ = self(negative, negative_tab)

        # triplet loss
        triplet_loss_function = nn.TripletMarginLoss()
        triplet_loss = triplet_loss_function(
            embeddings, pos_embeddings, neg_embeddings)
        # cross entropy loss
        cross_ent_loss_function = nn.CrossEntropyLoss(
            weight=self.class_weights)
        cross_ent_loss = cross_ent_loss_function(y_pred, y.squeeze())
        # center loss
        center_loss = self.center_loss(embeddings, y.squeeze())
        # sum the losses
        loss = self.alpha_cross_ent*cross_ent_loss + \
            self.alpha_center * center_loss + self.alpha_triplet*triplet_loss

        # Log loss on every epoch
        self.log('test_epoch_loss', loss, on_epoch=True, on_step=False)

        return loss
