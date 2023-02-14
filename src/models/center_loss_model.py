import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
# from monai.networks.nets.resnet_group import resnet10, resnet18, resnet34, resnet50
from models.model_blocks.resnet_block import ResNet
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from pytorch_metric_learning import losses
import torchmetrics
from torch.nn import Softmax
from center_loss import compute_center_loss, get_center_delta

# **************************************************************************
# THIS IS NOT TESTED
# THE IMPLEMENTATION WILL CHANGE (TRIPLET * CENTER / CROSS ENTROPY ' CENTER)
# **************************************************************************


class CenterLossModel(LightningModule):
    '''
    Uses ResNet for the image data, concatenates image and tabular data at the end
    '''

    def __init__(self, learning_rate=0.013, weight_decay=0.01):

        super().__init__()
        self.save_hyperparameters()

        self.lr = learning_rate
        self.wd = weight_decay
        self.num_classes = 3
        self.feature_dim = 42
        self.alpha = 0.2
        self.centers = (
            (torch.rand(self.num_classes, self.feature_dim) - 0.5) * 2)
        self.center_deltas = torch.zeros(self.num_classes, self.feature_dim)

        # IMAGE DATA
        # output dimension is adapted from simCLR
        self.resnet = ResNet()  # output features are 32

        # TABULAR DATA
        # fc layer for tabular data
        self.fc1 = nn.Linear(13, 10)

        # TABULAR + IMAGE DATA
        # mlp projection head which takes concatenated input
        resnet_out_dim = 32
        self.fc2 = nn.Linear(resnet_out_dim + 10, 3)

        # track accuracy
        self.train_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=3, top_k=1)
        self.val_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=3, top_k=1)

        self.train_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=3, top_k=1)
        self.val_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=3, top_k=1)

        self.softmax = Softmax(dim=1)

    def forward(self, img, tab):
        """

        img is the input image data ()
        tab is th einput tabular data

        """
        # run the model for the image
        img = self.resnet(img)
        img = img.view(img.size(0), -1)

        # change the dtype of the tabular data
        tab = tab.to(torch.float32)
        # forward tabular data
        tab = F.relu(self.fc1(tab))

        # concat image and tabular data
        x = torch.cat((img, tab), dim=1)
        # get the output for triplet loss
        out1 = x

        # calculate the output for classification loss
        out2 = self.fc2(x)

        return out1, out2

    def configure_optimizers(self):

        # weight decay can be added, lr can be changed
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd)
        # scheduler = MultiStepLR(optimizer,
        #                         # List of epoch indices
        #                         milestones=[18, 27],
        #                         gamma=0.1)  # Multiplicative factor of learning rate decay

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

        # add center loss
        triplet_loss_function = nn.TripletMarginLoss()
        loss = 0.4 * triplet_loss_function(
            embeddings, pos_embeddings, neg_embeddings) + 0.4 * F.cross_entropy(y_pred, y.squeeze()) + self.alpha * compute_center_loss(embeddings, self.centers, y)

        # Log loss on every epoch
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)

        # make features untrack by autograd, or there will be
        # a memory leak when updating the centers
        self.center_deltas = get_center_delta(
            embeddings, self.centers, y, self.alpha)

        # calculate acc
        # take softmax
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
        y_pred_softmax = self.softmax(y_pred)
        # y_pred_softmax = torch.sigmoid(y_pred)

        # get the index of max value
        pred_label = torch.argmax(y_pred_softmax, dim=1)

        # calculate and log accuracy
        train_acc = self.train_macro_accuracy(pred_label, y)
        self.log('train_macro_acc', train_acc, on_epoch=True, on_step=False)

        # calculate and log accuracy
        train_micro_acc = self.train_micro_accuracy(pred_label, y)
        self.log('train_micro_acc', train_micro_acc,
                 on_epoch=True, on_step=False)

        return loss

    def on_train_batch_end(self, *args, **kwargs):
        self.centers = self.centers - self.center_deltas

    def validation_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, positive, negative, tab, positive_tab, negative_tab, y = batch[
            0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

        # TODO: dont feed positive and negatives to classification layer
        embeddings, y_pred = self(img, tab)
        pos_embeddings, _ = self(positive, positive_tab)
        neg_embeddings, _ = self(negative, negative_tab)

        # add center loss
        triplet_loss_function = nn.TripletMarginLoss()
        loss = 0.4 * triplet_loss_function(
            embeddings, pos_embeddings, neg_embeddings) + 0.4 * F.cross_entropy(y_pred, y.squeeze()) + self.alpha * compute_center_loss(embeddings, self.centers, y)

        # Log loss on every epoch
        self.log('val_epoch_loss', loss, on_epoch=True, on_step=False)

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

        return loss

    def test_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, positive, negative, tab, positive_tab, negative_tab, y = batch[
            0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

        # TODO: dont feed positive and negatives to classification layer
        embeddings, y_pred = self(img, tab)
        pos_embeddings, _ = self(positive, positive_tab)
        neg_embeddings, _ = self(negative, negative_tab)

        # add center loss
        triplet_loss_function = nn.TripletMarginLoss()
        loss = 0.4 * triplet_loss_function(
            embeddings, pos_embeddings, neg_embeddings) + 0.4 * F.cross_entropy(y_pred, y.squeeze()) + self.alpha * compute_center_loss(embeddings, self.centers, y)

        # Log loss on every epoch
        self.log('test_epoch_loss', loss, on_epoch=True, on_step=False)

        return loss
