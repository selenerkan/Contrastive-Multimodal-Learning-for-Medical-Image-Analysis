import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
import torchmetrics
from torch.nn import Softmax
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torchvision
from ham_settings import class_weights


class BaselineModel(LightningModule):
    '''
    Resnet Model Class including the training, validation and testing steps
    '''

    def __init__(self, learning_rate=0.013, weight_decay=0.01):

        super().__init__()
        self.is_gpu = 'cpu'
        if torch.cuda.is_available():
            self.is_gpu = 'cuda'

        self.save_hyperparameters()

        self.lr = learning_rate
        self.wd = weight_decay
        self.num_classes = 7

        # IMAGE DATA
        self.conv1 = nn.Conv2d(3, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16*54*54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

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

    def forward(self, x):
        """
        x is the input image data
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
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

    def get_accuracy(self, predicted, labels):
        batch_len, correct = 0, 0
        batch_len = labels.size(0)
        correct = (predicted == labels).sum().item()
        return batch_len, correct

    def training_step(self, batch, batch_idx):

        img, tab, y = batch

        y_pred = self(img)

        loss_func = nn.CrossEntropyLoss(weight=class_weights.to(self.is_gpu))
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
        batch_len, acc = self.get_accuracy(pred_label, y)
        self.total_samples += batch_len
        self.train_acc += acc

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

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        train_epoch_acc = self.train_acc/self.total_samples
        self.log('train_acc_blog', train_epoch_acc,
                 on_epoch=True, on_step=False)

        self.total_samples = 0
        self.train_acc = 0
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def validation_step(self, batch, batch_idx):

        img, tab, y = batch

        y_pred = self(img)

        loss_func = nn.CrossEntropyLoss(weight=class_weights.to(self.is_gpu))
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

        return loss

    def test_step(self, batch, batch_idx):

        img, tab, y = batch
        y_pred = self(img)

        loss_func = nn.CrossEntropyLoss(weight=class_weights.to(self.is_gpu))
        loss = loss_func(y_pred, y)

        self.log("test_loss", loss)

        return loss
