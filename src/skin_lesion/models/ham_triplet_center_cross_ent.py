import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torchmetrics
from torch.nn import Softmax
from center_loss import CenterLoss
import torchvision
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassF1Score
import wandb
from roc_curve import roc_curve


class TripletCenterModel(LightningModule):
    '''
    Uses ResNet for the image data, concatenates image and tabular data at the end
    '''

    def __init__(self, seed, learning_rate=0.013, weight_decay=0.01, alpha_center=0.01, alpha_triplet=0, correlation=False):

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
        self.correlation = correlation
        # weights of the losses
        self.alpha_center = alpha_center
        self.alpha_triplet = alpha_triplet
        self.alpha_cross_ent = (1-alpha_center-alpha_triplet)

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
        if self.correlation:
            concatenation_dimension = (self.embedding_dimension * 2) - 1
        else:
            concatenation_dimension = 128

        # concatanation_dimension = 128
        # outputs will be used in triplet/center loss
        self.fc7 = nn.Linear(concatenation_dimension, self.feature_dim)
        self.fc8 = nn.Linear(32, 7)  # classification head

        # initiate losses
        self.center_loss = CenterLoss(
            num_classes=self.num_classes, feat_dim=self.feature_dim, use_gpu=self.use_gpu, seed=seed)
        self.cross_ent_loss_function = nn.CrossEntropyLoss(
            weight=self.class_weights)
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

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
        if self.correlation:
            img = img.unsqueeze(0)
            tab = tab.unsqueeze(1)
            x = F.conv1d(img, tab, padding=self.embedding_dimension -
                         1, groups=img.size(1))
            x = x.squeeze()
        else:
            x = torch.cat((img, tab), dim=1)

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

        # UNCOMMENT FOR LR SCHEDULER
        # scheduler = MultiStepLR(optimizer,
        #                         # List of epoch indices
        #                         milestones=[23, 33],
        #                         gamma=0.5)  # Multiplicative factor of learning rate decay

        # return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, pos_img, neg_img, tab, pos_tab, neg_tab, y = batch[
            0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

        anchor_emb, y_pred = self(img, tab)
        pos_emb, _ = self(pos_img, pos_tab)
        neg_emb, _ = self(neg_img, neg_tab)

        cross_ent_loss = self.cross_ent_loss_function(y_pred, y.squeeze())
        center_loss = self.center_loss(anchor_emb, y.squeeze())
        triplet_loss = self.triplet_loss(anchor_emb, pos_emb, neg_emb)
        # weighted sum of the losses
        loss = self.alpha_cross_ent*cross_ent_loss + \
            self.alpha_center * center_loss + self.alpha_triplet * triplet_loss

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

        # get tabular and image data from the batch
        img, pos_img, neg_img, tab, pos_tab, neg_tab, y = batch[
            0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

        anchor_emb, y_pred = self(img, tab)
        pos_emb, _ = self(pos_img, pos_tab)
        neg_emb, _ = self(neg_img, neg_tab)

        cross_ent_loss = self.cross_ent_loss_function(y_pred, y.squeeze())
        center_loss = self.center_loss(anchor_emb, y.squeeze())
        triplet_loss = self.triplet_loss(anchor_emb, pos_emb, neg_emb)
        # weighted sum of the losses
        loss = self.alpha_cross_ent*cross_ent_loss + \
            self.alpha_center * center_loss + self.alpha_triplet * triplet_loss

        # Log loss on every epoch
        self.log('val_epoch_loss', loss, on_epoch=True, on_step=False)
        self.log('val_center_loss', center_loss,
                 on_epoch=True, on_step=False)
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
                                                                                 labels=["akiec", "bcc", 'bkl', 'df', 'mel', 'nv', 'vasc'], title='Val ROC Epoch:{self.current_epoch}')})

        self.val_predictions = []
        self.val_targets = []

    def test_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, pos_img, neg_img, tab, pos_tab, neg_tab, y = batch[
            0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

        anchor_emb, y_pred = self(img, tab)
        pos_emb, _ = self(pos_img, pos_tab)
        neg_emb, _ = self(neg_img, neg_tab)

        cross_ent_loss = self.cross_ent_loss_function(y_pred, y.squeeze())
        center_loss = self.center_loss(anchor_emb, y.squeeze())
        triplet_loss = self.triplet_loss(anchor_emb, pos_emb, neg_emb)
        # weighted sum of the losses
        loss = self.alpha_cross_ent*cross_ent_loss + \
            self.alpha_center * center_loss + self.alpha_triplet * triplet_loss

        # Log loss on every epoch
        self.log('test_epoch_loss', loss, on_epoch=True, on_step=False)
        self.log('test_center_loss', center_loss,
                 on_epoch=True, on_step=False)
        self.log('test_cross_ent_loss', cross_ent_loss,
                 on_epoch=True, on_step=False)
        self.log('test_triplet_loss', triplet_loss,
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
                                         labels=["akiec", "bcc", 'bkl', 'df', 'mel', 'nv', 'vasc'], title='ROC_Test')})
        self.test_predictions = []
        self.test_targets = []
