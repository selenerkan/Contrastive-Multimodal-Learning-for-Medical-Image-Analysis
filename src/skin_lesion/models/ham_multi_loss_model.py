import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from pytorch_metric_learning import losses
import torchmetrics
from torch.nn import Softmax
from center_loss import compute_center_loss, get_center_delta
import torchvision


class MultiLossModel(LightningModule):
    '''
    Uses ResNet for the image data, concatenates image and tabular data at the end
    '''

    def __init__(self, learning_rate=0.013, weight_decay=0.01):

        super().__init__()
        self.save_hyperparameters()

        self.lr = learning_rate
        self.wd = weight_decay
        # weights of the losses
        self.alpha_center = 0.2
        # self.alpha_triplet = 0.4
        self.alpha_cross_ent = 0.8

        # parameters for center loss
        self.num_classes = 7
        self.feature_dim = 32
        self.centers = (
            (torch.rand(self.num_classes, self.feature_dim) - 0.5) * 2)
        self.center_deltas = torch.zeros(self.num_classes, self.feature_dim)

        # IMAGE DATA
        self.resnet = torchvision.models.resnet18(
            pretrained=False)  # output features are 1000
        # change resnet fc output to 128 features
        self.resnet.fc = nn.Linear(512, 128)

        # TABULAR DATA
        # fc layer for tabular data
        self.fc1 = nn.Linear(3, 128)  # output features are 128

        # shared FC layer
        self.fc2 = nn.Linear(128, 64)

        # TABULAR + IMAGE DATA
        # mlp projection head which takes concatenated input
        concatanation_dimension = 128
        # outputs will be used in triplet/center loss
        self.fc3 = nn.Linear(concatanation_dimension, self.feature_dim)
        self.fc4 = nn.Linear(32, 7)  # classification head

        # track accuracy
        self.train_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=7, top_k=1)
        self.val_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=7, top_k=1)

        self.train_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=7, top_k=1)
        self.val_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=7, top_k=1)

        self.softmax = Softmax(dim=1)

    def forward(self, img, tab):
        """

        img is the input image data ()
        tab is th einput tabular data

        """
        # run the model for the image
        img = self.resnet(img)
        img = self.fc2(F.relu(img))

        # forward pass for tabular data
        tab = tab.to(torch.float32)
        tab = F.relu(self.fc1(tab))
        tab = self.fc2(tab)

        # concat image and tabular data
        x = torch.cat((img, tab), dim=1)
        # get the final concatenated embedding
        out1 = self.fc3(x)
        # calculate the output of classification head
        out2 = self.fc4(F.relu(out1))

        return out1, out2

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd)

        # UNCOMMENT FOR LR SCHEDULER
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

        # triplet loss
        # triplet_loss_function = nn.TripletMarginLoss()
        # triplet_loss = self.alpha_triplet * triplet_loss_function(
        #     embeddings, pos_embeddings, neg_embeddings)
        # cross entropy loss
        cross_entropy_loss = self.alpha_cross_ent * \
            F.cross_entropy(y_pred, y)
        # center loss
        center_loss = self.alpha_center * \
            compute_center_loss(embeddings, self.centers, y)
        # sum the losses
        loss = cross_entropy_loss + center_loss

        # Log loss on every epoch
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)

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

        # triplet loss
        # triplet_loss_function = nn.TripletMarginLoss()
        # triplet_loss = self.alpha_triplet * triplet_loss_function(
        #     embeddings, pos_embeddings, neg_embeddings)
        # cross entropy loss
        cross_entropy_loss = self.alpha_cross_ent * \
            F.cross_entropy(y_pred, y)
        # center loss
        center_loss = self.alpha_center * \
            compute_center_loss(embeddings, self.centers, y)
        # sum the losses
        loss = cross_entropy_loss + center_loss

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

        # triplet loss
        # triplet_loss_function = nn.TripletMarginLoss()
        # triplet_loss = self.alpha_triplet * triplet_loss_function(
        #     embeddings, pos_embeddings, neg_embeddings)
        # cross entropy loss
        cross_entropy_loss = self.alpha_cross_ent * \
            F.cross_entropy(y_pred, y)
        # center loss
        center_loss = self.alpha_center * \
            compute_center_loss(embeddings, self.centers, y)
        # sum the losses
        loss = cross_entropy_loss + center_loss

        # Log loss on every epoch
        self.log('test_epoch_loss', loss, on_epoch=True, on_step=False)

        return loss
