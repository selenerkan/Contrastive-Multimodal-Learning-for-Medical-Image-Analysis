import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
# from monai.networks.nets.resnet_group import resnet10, resnet18, resnet34, resnet50
from models.model_blocks.resnet_block import ResNet
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from pytorch_metric_learning import losses
import torchmetrics
from sklearn.neighbors import KNeighborsClassifier


class TripletModel(LightningModule):
    '''
    Uses ResNet for the image data, concatenates image and tabular data at the end
    '''

    def __init__(self, learning_rate=0.013, weight_decay=0.01):

        super().__init__()
        self.save_hyperparameters()

        self.lr = learning_rate
        self.wd = weight_decay

        # IMAGE DATA
        # output dimension is adapted from simCLR
        self.resnet = ResNet(n_outputs=128)  # output features are 128

        # TABULAR DATA
        # fc layer for tabular data
        self.fc1 = nn.Linear(13, 128)  # output features are 128

        # shared FC layer
        self.fc2 = nn.Linear(128, 64)

        # TABULAR + IMAGE DATA
        # mlp projection head which takes concatenated input
        concatanation_dimension = 128
        # outputs will be used in triplet loss
        self.fc3 = nn.Linear(concatanation_dimension, 32)

        # set knn neighbor parameter
        self.knn_neighbor = 5
        self.knn = None
        # self.knn = KNeighborsClassifier(
        #     n_neighbors=self.knn_neighbor, n_jobs=-1)

        # track accuracy with knn
        self.knn_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=3, top_k=1)
        self.knn_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=3, top_k=1)

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
        out = self.fc3(x)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd)
        # scheduler = MultiStepLR(optimizer,
        #                         # List of epoch indices
        #                         milestones=[18, 27],
        #                         gamma=0.1)  # Multiplicative factor of learning rate decay

        # return [optimizer], [scheduler]
        return optimizer

    def on_train_epoch_start(self):
        self.knn = KNeighborsClassifier(
            n_neighbors=self.knn_neighbor, n_jobs=-1)

    def training_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, positive, negative, tab, positive_tab, negative_tab, y = batch[
            0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

        embeddings = self(img, tab)
        pos_embeddings = self(positive, positive_tab)
        neg_embeddings = self(negative, negative_tab)

        loss_function = nn.TripletMarginLoss()
        loss = loss_function(embeddings, pos_embeddings, neg_embeddings)

        # Log loss on every epoch
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)

        # fit the knn model
        self.knn.fit(embeddings.cpu(), y.cpu())

        return loss

    def validation_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, positive, negative, tab, positive_tab, negative_tab, y = batch[
            0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

        embeddings = self(img, tab)
        pos_embeddings = self(positive, positive_tab)
        neg_embeddings = self(negative, negative_tab)

        loss_function = nn.TripletMarginLoss()
        loss = loss_function(embeddings, pos_embeddings, neg_embeddings)

        # Log loss on every epoch
        self.log('val_epoch_loss', loss, on_epoch=True, on_step=False)

        if self.knn is not None:
            # do prediction using knn
            # get predictions
            y_pred = self.knn.predict(embeddings.cpu())

            # log knn metrics
            # accuracy: (tp + tn) / (p + n)
            micro_acc = self.knn_micro_accuracy(
                y_pred, y)
            self.log("KNN micro Acc", micro_acc, on_epoch=True, on_step=False)

            macro_acc = self.knn_macro_accuracy(
                y_pred, y)
            self.log("KNN macro Acc", macro_acc, on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, positive, negative, tab, positive_tab, negative_tab = batch[
            0], batch[1], batch[2], batch[3], batch[4], batch[5]

        embeddings = self(img, tab)
        pos_embeddings = self(positive, positive_tab)
        neg_embeddings = self(negative, negative_tab)

        loss_function = nn.TripletMarginLoss()
        loss = loss_function(embeddings, pos_embeddings, neg_embeddings)

        # Log loss on every epoch
        self.log('test_epoch_loss', loss, on_epoch=True, on_step=False)

        return loss
