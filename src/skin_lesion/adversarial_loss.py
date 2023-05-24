
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):
    def __init__(self, mlp):
        super().__init__()

        self.mlp = mlp

    def forward(self, true_emb, fake_emb):
        # Compute true and fake predictions
        true_pred = self.mlp(true_emb)
        fake_pred = self.mlp(fake_emb)

        # Define target labels for the adversarial loss
        true_target = torch.ones_like(true_pred)
        fake_target = torch.zeros_like(fake_pred)

        # Compute binary cross-entropy loss for true and fake predictions
        true_loss = F.binary_cross_entropy_with_logits(true_pred, true_target)
        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, fake_target)

        # Compute total adversarial loss as the sum of the two losses
        adversarial_loss = true_loss + fake_loss

        return adversarial_loss
