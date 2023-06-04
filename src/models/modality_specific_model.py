import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
# from monai.networks.nets.resnet import resnet10, resnet18, resnet34, resnet50
from models.model_blocks.resnet_block import ResNet
import torchmetrics
from torch.nn import Softmax
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassF1Score
import wandb
from skin_lesion.roc_curve import roc_curve
from skin_lesion.center_loss import CenterLoss


class ModalitySpecificCenterModel(LightningModule):
    '''
    Resnet Model Class including the training, validation and testing steps
    '''

    def __init__(self, seed, learning_rate=0.013, weight_decay=0.01, alpha_center=0.01):

        super().__init__()
        self.use_gpu = False
        if torch.cuda.is_available():
            self.use_gpu = True

        self.save_hyperparameters()

        self.lr = learning_rate
        self.wd = weight_decay
        self.num_classes = 3
        self.embedding_dimension = 32
        # weights of the losses
        self.alpha_center = alpha_center
        self.alpha_cross_ent = (1-alpha_center)

        # IMAGE DATA
        # output dimension is adapted from simCLR
        self.resnet = ResNet()  # output features are 128
        self.resnet.fc = nn.Linear(
            self.embedding_dimension, self.embedding_dimension)

        # TABULAR DATA
        # fc layer for tabular data
        self.fc1 = nn.Linear(11, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, self.embedding_dimension)

        # shared FC layer
        self.fc5 = nn.Linear(self.embedding_dimension,
                             self.embedding_dimension)

        # TABULAR + IMAGE DATA
        # mlp projection head which takes concatenated input
        concatenation_dimension = 64

        # outputs will be used in triplet loss
        self.fc6 = nn.Linear(concatenation_dimension, 32)
        self.fc7 = nn.Linear(32, self.num_classes)  # classification head

        self.cross_ent_loss_function = nn.CrossEntropyLoss()
        self.center_loss_function_img = CenterLoss(
            num_classes=self.num_classes, feat_dim=self.embedding_dimension, use_gpu=self.use_gpu, seed=seed)
        self.center_loss_function_tab = CenterLoss(
            num_classes=self.num_classes, feat_dim=self.embedding_dimension, use_gpu=self.use_gpu, seed=seed)

        # TRACK METRICS

        # track predictions for ROC curve
        self.test_predictions = []
        self.test_targets = []
        self.val_predictions = []
        self.val_targets = []

        # F1 - SCORE
        self.train_macro_F1 = MulticlassF1Score(
            num_classes=self.num_classes, average='macro')
        self.val_macro_F1 = MulticlassF1Score(
            num_classes=self.num_classes, average='macro')
        self.test_macro_F1 = MulticlassF1Score(
            num_classes=self.num_classes, average='macro')

        # F1 - SCORE Per class
        self.train_class_F1 = MulticlassF1Score(
            num_classes=self.num_classes, average='none')
        self.val_class_F1 = MulticlassF1Score(
            num_classes=self.num_classes, average='none')
        self.test_class_F1 = MulticlassF1Score(
            num_classes=self.num_classes, average='none')

        # track precision for each class
        self.train_macro_precision = MulticlassPrecision(
            num_classes=self.num_classes, average='macro')
        self.val_macro_precision = MulticlassPrecision(
            num_classes=self.num_classes, average='macro')
        self.test_macro_precision = MulticlassPrecision(
            num_classes=self.num_classes, average='macro')

        # track precision for each class
        self.train_class_precision = MulticlassPrecision(
            num_classes=self.num_classes, average='none')
        self.val_class_precision = MulticlassPrecision(
            num_classes=self.num_classes, average='none')
        self.test_class_precision = MulticlassPrecision(
            num_classes=self.num_classes, average='none')

        # track recall for each class
        self.train_macro_recall = MulticlassRecall(
            num_classes=self.num_classes, average='macro')
        self.val_macro_recall = MulticlassRecall(
            num_classes=self.num_classes, average='macro')
        self.test_macro_recall = MulticlassRecall(
            num_classes=self.num_classes, average='macro')

        # track recall for each class
        self.train_class_recall = MulticlassRecall(
            num_classes=self.num_classes, average='none')
        self.val_class_recall = MulticlassRecall(
            num_classes=self.num_classes, average='none')
        self.test_class_recall = MulticlassRecall(
            num_classes=self.num_classes, average='none')

        # track accuracy
        self.train_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=self.num_classes, top_k=1)
        self.val_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=self.num_classes, top_k=1)
        self.test_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=self.num_classes, top_k=1)

        self.train_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=self.num_classes, top_k=1)
        self.val_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=self.num_classes, top_k=1)
        self.test_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=self.num_classes, top_k=1)

        self.softmax = Softmax(dim=1)

    def forward(self, img, tab):
        """
        img is the input image data
        tab is th einput tabular data
        """

        # run the model for the image
        img = self.resnet(img)
        img = self.fc5(F.relu(img))

        # forward pass for tabular data
        tab = tab.to(torch.float32)
        tab = F.relu(self.fc1(tab))
        tab = F.relu(self.fc2(tab))
        tab = F.relu(self.fc3(tab))
        tab = F.relu(self.fc4(tab))
        tab = self.fc5(tab)

        # concat image and tabular data
        x = torch.cat((img, tab), dim=1)

        # get the final concatenated embedding
        x = F.relu(self.fc6(x))
        # calculate the output of classification head
        out = self.fc7(x)

        return img, tab, out

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

        cross_ent_loss = self.cross_ent_loss_function(y_pred, y)
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

        # calculate and log metrics
        self.train_macro_accuracy(pred_label, y)  # (preds, target)
        self.train_micro_accuracy(pred_label, y)
        self.train_macro_F1(pred_label, y)
        self.train_class_F1(pred_label, y)
        self.train_macro_precision(pred_label, y)
        self.train_class_precision(pred_label, y)
        self.train_macro_recall(pred_label, y)
        self.train_class_recall(pred_label, y)

        # log the metrics
        self.log('train_macro_acc',
                 self.train_macro_accuracy,
                 on_epoch=True, on_step=False)
        self.log('train_micro_acc', self.train_micro_accuracy,
                 on_epoch=True, on_step=False)
        self.log('train_macro_F1', self.train_macro_F1,
                 on_epoch=True, on_step=False)
        self.log('train_macro_precision', self.train_macro_precision,
                 on_epoch=True, on_step=False)
        self.log('train_macro_recall', self.train_macro_recall,
                 on_epoch=True, on_step=False)

        return loss

    def on_train_epoch_end(self):
        # compute the metrics
        train_class_F1 = self.train_class_F1.compute()
        train_class_precision = self.train_class_precision.compute()
        train_class_recall = self.train_class_recall.compute()

        # log F1 scores for each class
        for i in range(self.num_classes):
            self.log("train_F1_class_" + str(i), train_class_F1[i].item())
            self.log("train_precision_class_" + str(i),
                     train_class_precision[i].item())
            self.log("train_recall_class_" + str(i),
                     train_class_recall[i].item())

        # reset the metrics
        self.train_class_F1.reset()
        self.train_class_precision.reset()
        self.train_class_recall.reset()

    def validation_step(self, batch, batch_idx):

        img, tab, y = batch

        embed_img, embed_tab, y_pred = self(img, tab)

        cross_ent_loss = self.cross_ent_loss_function(y_pred, y)
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
        self.log('val_center_loss_tab ', center_loss_tab,
                 on_epoch=True, on_step=False)
        self.log('val_cross_ent_loss', cross_ent_loss,
                 on_epoch=True, on_step=False)

        # calculate acc
        # take softmax
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
        y_pred_softmax = self.softmax(y_pred)

        # get the index of max value
        pred_label = torch.argmax(y_pred_softmax, dim=1)

        # save the predictions and targets
        self.val_predictions.append(y_pred.detach())
        self.val_targets.append(y.detach())

        # calculate and log metrics
        self.val_macro_accuracy(pred_label, y)  # (preds, target)
        self.val_micro_accuracy(pred_label, y)
        self.val_macro_F1(pred_label, y)
        self.val_class_F1(pred_label, y)
        self.val_macro_precision(pred_label, y)
        self.val_class_precision(pred_label, y)
        self.val_macro_recall(pred_label, y)
        self.val_class_recall(pred_label, y)

        # log the metrics
        self.log('val_macro_acc',
                 self.val_macro_accuracy,
                 on_epoch=True, on_step=False)
        self.log('val_micro_acc', self.val_micro_accuracy,
                 on_epoch=True, on_step=False)
        self.log('val_macro_F1', self.val_macro_F1,
                 on_epoch=True, on_step=False)
        self.log('val_macro_precision', self.val_macro_precision,
                 on_epoch=True, on_step=False)
        self.log('val_macro_recall', self.val_macro_recall,
                 on_epoch=True, on_step=False)

        return loss

    def on_validation_epoch_end(self):
        # compute the metrics
        val_class_F1 = self.val_class_F1.compute()
        val_class_precision = self.val_class_precision.compute()
        val_class_recall = self.val_class_recall.compute()

        # log F1 scores for each class
        for i in range(self.num_classes):
            self.log("val_F1_class_" + str(i),  val_class_F1[i].item())
            self.log("val_precision_class_" + str(i),
                     val_class_precision[i].item())
            self.log("val_recall_class_" + str(i),  val_class_recall[i].item())

        # reset the metrics
        self.val_class_F1.reset()
        self.val_class_precision.reset()
        self.val_class_recall.reset()

        preds = torch.cat(self.val_predictions)
        preds = torch.nn.functional.softmax(preds, dim=1)
        targets = torch.cat(self.val_targets)
        wandb.log({f"validation_roc_curve_epoch_{self.current_epoch}": roc_curve(targets.unsqueeze(1), preds,
                                                                                 labels=['CN', 'AD', 'LMCI'], title='Val ROC Epoch:{self.current_epoch}')})

        self.val_predictions = []
        self.val_targets = []

    def test_step(self, batch, batch_idx):

        img, tab, y = batch

        embed_img, embed_tab, y_pred = self(img, tab)

        cross_ent_loss = self.cross_ent_loss_function(y_pred, y)
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
        self.log('test_center_loss_tab ', center_loss_tab,
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

        # save the predictions and targets
        self.test_predictions.append(y_pred.detach())
        self.test_targets.append(y.detach())

        # calculate and log accuracy
        self.test_macro_accuracy(pred_label, y)
        self.test_micro_accuracy(pred_label, y)

        self.test_macro_F1(pred_label, y)
        self.test_class_F1(pred_label, y)
        self.test_macro_precision(pred_label, y)
        self.test_class_precision(pred_label, y)
        self.test_macro_recall(pred_label, y)
        self.test_class_recall(pred_label, y)

        # log the metrics
        self.log('test_macro_acc',
                 self.test_macro_accuracy,
                 on_epoch=True, on_step=False)
        self.log('test_micro_acc', self.test_micro_accuracy,
                 on_epoch=True, on_step=False)
        self.log('test_macro_F1', self.test_macro_F1,
                 on_epoch=True, on_step=False)
        self.log('test_macro_precision', self.test_macro_precision,
                 on_epoch=True, on_step=False)
        self.log('test_macro_recall', self.test_macro_recall,
                 on_epoch=True, on_step=False)

        return loss

    def on_test_epoch_end(self):
        # compute the metrics
        test_class_F1 = self.test_class_F1.compute()
        test_class_precision = self.test_class_precision.compute()
        test_class_recall = self.test_class_recall.compute()

        # log F1 scores for each class
        for i in range(self.num_classes):
            self.log("test_F1_class_" + str(i), test_class_F1[i].item())
            self.log("test_precision_class_" + str(i),
                     test_class_precision[i].item())
            self.log("test_recall_class_" + str(i),
                     test_class_recall[i].item())

        # reset the metrics
        self.test_class_F1.reset()
        self.test_class_precision.reset()
        self.test_class_recall.reset()

        preds = torch.cat(self.test_predictions)
        targets = torch.cat(self.test_targets)
        wandb.log({"ROC_Test": roc_curve(targets.unsqueeze(1), preds,
                                         labels=['CN', 'AD', 'LMCI'], title='ROC_Test')})
        self.test_predictions = []
        self.test_targets = []
