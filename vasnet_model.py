__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"


import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layer_norm import *
from enhanced_attention import EnhancedAttention
import numpy as np



class SelfAttention(nn.Module):

    def __init__(self, apperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()

        self.apperture = apperture
        self.ignore_itself = ignore_itself

        self.m = input_size
        self.output_size = output_size

        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

        self.drop50 = nn.Dropout(0.5)



    def forward(self, x):
        n = x.shape[0]  # sequence length

        K = self.K(x)  # ENC (n x m) => (n x H) H= hidden size
        Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size
        V = self.V(x)

        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1,0))

        if self.ignore_itself:
            # Zero the diagonal activations (a distance of each frame with itself)
            logits[torch.eye(n).byte()] = -float("Inf")

        if self.apperture > 0:
            # Set attention to zero to frames further than +/- apperture from the current one
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float("Inf")

        att_weights_ = nn.functional.softmax(logits, dim=-1)
        weights = self.drop50(att_weights_)
        y = torch.matmul(V.transpose(1,0), weights).transpose(1,0)
        y = self.output_linear(y)

        return y, att_weights_





#Difference Attention Module 

class DAM(nn.Module):
    def __init__(self):
        super(DAM, self).__init__()

        self.dropout1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)

    # x represents our data
    def forward(self, x):
        d1 = self.fc1(torch.abs(x[1] - x[0]))
        d1 = F.relu(d1)
        # d1 = self.dropout1(d1)
        # d1 = self.fc2(d1)

        d2 = self.fc1(torch.abs(x[2] - x[0]))
        d2 = F.relu(d2)
        # d2 = self.dropout1(d2)
        # d2 = self.fc2(d2)

        d3 = self.fc1(torch.abs(x[4] - x[0]))
        d3 = F.relu(d3)
        # d3 = self.dropout1(d3)
        # d3 = self.fc2(d3)

        t = d1 + d2 + d3

        for i in range(1, len(x) - 4):
            d1 = self.fc1(torch.abs(x[i+1] - x[i]))
            d1 = F.relu(d1)
            # d1 = self.dropout1(d1)
            # d1 = self.fc2(d1)

            d2 = self.fc1(torch.abs(x[i+2] - x[i]))
            d2 = F.relu(d2)
            # d2 = self.dropout1(d2)
            # d2 = self.fc2(d2)

            d3 = self.fc1(torch.abs(x[i+4] - x[i]))
            d3 = F.relu(d3)
            # d3 = self.dropout1(d3)
            # d3 = self.fc2(td3)

            temp = d1 + d2 + d3

            t = torch.cat((t, temp))
        
        for i in range(len(x)-4, len(x)):
            t = torch.cat((t, x[i]))


        #print("shape of t ", t.shape)
        t = torch.reshape(t, (len(x), 1024))
        t = self.dropout1(t)
        
        return t     



#VASNet module
class VASNet(nn.Module):
    def __init__(self, hps=None):
        super(VASNet, self).__init__()
        self.hps = HParameters() if hps is None else hps
        self.m = 1024  # cnn features size
        
        # Enhanced attention mechanism
        self.attention = EnhancedAttention(
            input_size=self.m,
            num_heads=self.hps.num_heads,
            dropout=self.hps.dropout
        )
        
        # Frame scoring layers
        self.frame_score = nn.Sequential(
            nn.Linear(self.m, self.m),
            nn.ReLU(),
            nn.Dropout(self.hps.dropout),
            nn.LayerNorm(self.m),
            nn.Linear(self.m, 1),
            nn.Sigmoid()
        )
        
        # Temporal modeling
        self.temporal_conv = nn.Conv1d(self.m, self.m, kernel_size=3, padding=1)
        self.temporal_norm = nn.LayerNorm(self.m)

    def forward(self, x, seq_len, mn=None):
        m = x.shape[2]
        x = x.view(-1, m)
        
        # If motion features are provided, incorporate them
        if mn is not None:
            mn = mn.view(-1, m)
            x = x + mn  # Add motion features to input features
        
        # Apply enhanced attention
        attended_features, attention_weights = self.attention(x)
        
        # Temporal modeling
        temp_features = self.temporal_conv(attended_features.transpose(0, 1).unsqueeze(0))
        temp_features = temp_features.squeeze(0).transpose(0, 1)
        temp_features = self.temporal_norm(temp_features + attended_features)
        
        # Generate frame-level scores
        scores = self.frame_score(temp_features)
        scores = scores.view(1, -1)
        
        return scores, attention_weights
    
    def compute_loss(self, pred_scores, target_scores, attention_weights):
        # Base reconstruction loss
        base_loss = F.mse_loss(pred_scores, target_scores)
        
        # Temporal consistency loss
        temp_loss = F.mse_loss(
            pred_scores[:, 1:],
            pred_scores[:, :-1]
        )
        
        # Diversity loss to encourage diverse selection
        diversity_loss = -torch.mean(
            torch.abs(
                pred_scores[:, 1:] - pred_scores[:, :-1]
            )
        )
        
        # Attention regularization
        attention_loss = -torch.mean(
            torch.sum(
                attention_weights * torch.log(attention_weights + 1e-10),
                dim=-1
            )
        )
        
        # Combine losses
        total_loss = base_loss + \
                    self.hps.temporal_weight * temp_loss + \
                    self.hps.diversity_weight * diversity_loss + \
                    0.1 * attention_loss
                    
        return total_loss, {
            'base_loss': base_loss.item(),
            'temp_loss': temp_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'attention_loss': attention_loss.item()
        }



if __name__ == "__main__":
    pass
