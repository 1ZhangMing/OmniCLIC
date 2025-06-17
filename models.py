""" Componets of the OmniCLIC
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import get_realmlp_td_s_pipeline
def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
def swish(x):
    return x * torch.sigmoid(x)
class RealMLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout=0.2):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.LayerNorm(hidden_dims[i+1]),
                nn.LeakyReLU(0.25),
                nn.Dropout(dropout)
            ) for i in range(len(hidden_dims)-1)
        ])
        self.skip = nn.Linear(in_dim, hidden_dims[-1], bias=False)  # 残差连接
        self.blocks.apply(xavier_init)

    def forward(self, x):
        residual = self.skip(x)
        for block in self.blocks:
            x = block(x)
        return x + residual  # 跳跃连接
def get_dropout_rate(epoch, max_epochs):
    t = epoch / max_epochs
    return 0.15 * 0.5 * (1 + np.cos(np.pi * min(1, 2*t)))  # flat_cos schedule


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias
#多头注意力
class MultiHeadAttentionEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.0,dr=0.1):
        super(MultiHeadAttentionEncoder, self).__init__()

        # 输出层
        self.fc_out = nn.Linear(hidden_dim[2], hidden_dim[2])  # 输出层的维度应与多头注意力的输出维度一致
        self.dropout = nn.Dropout(dropout)
        # LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_dim[2])  # LayerNorm 的维度应与多头注意力的输出维度一致
        act = nn.LeakyReLU(0.01)
        
        self.net = nn.Sequential(
            ScalingLayer(in_dim), 
            NTPLinear(in_dim, 256), 
            nn.Dropout(dr),
            NTPLinear(256, 256), act,
            nn.Dropout(dr),
            NTPLinear(256, 256), act,
            nn.Dropout(dr),
            NTPLinear(256, hidden_dim[2], zero_init=True),
            act,
            self.layer_norm
        )
    def forward(self, x):
        x = x / (1 + (x/3)**2)**0.5
        return self.net(x)

    def apply_weight_decay(self):
        # 应用 weight decay
        for param in self.parameters():
            param.data.mul_(1 - self.weight_decay)

class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class OCDN(nn.Module):
    def __init__(self, num_view, num_cls, hOCDN_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hOCDN_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hOCDN_dim, num_cls)
        )
        self.model.apply(xavier_init)
        self.fc=nn.Linear(num_view*num_cls,num_cls)

    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),(-1,pow(self.num_cls,2),1))
        for i in range(2,num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)),(-1,pow(self.num_cls,i+1),1))
        OCDN_feat = torch.reshape(x, (-1,pow(self.num_cls,num_view)))
        output = self.model(OCDN_feat)

        return output


def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, dropout=0.5, dr=0.1):
    model_dict = {}
    sumdim=0
    for i in range(num_view):

        model_dict["E{:}".format(i+1)] = MultiHeadAttentionEncoder(dim_list[i], dim_he_list,dropout,dr=0.1)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[2], num_class)
        sumdim=sumdim+dim_list[i]
    
    model_dict["EU"] = MultiHeadAttentionEncoder(dim_he_list[2]*num_view, dim_he_list, dropout)
    model_dict["CU"] = Classifier_1(dim_he_list[2], num_class)
    if num_view >= 2:

        model_dict["C"] = OCDN(3, num_class, dim_hc)  
    return model_dict

def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4,weight_decay_e=1e-4, weight_decay_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()),
                lr=lr_e,
                weight_decay=weight_decay_e)
     
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c, weight_decay=weight_decay_c)
    return optim_dict






class ScalingLayer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale[None, :]


class NTPLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, zero_init: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factor = 0.0 if zero_init else 1.0
        self.weight = nn.Parameter(factor * torch.randn(in_features, out_features))
        self.bias = nn.Parameter(factor * torch.randn(1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1. / np.sqrt(self.in_features)) * (x @ self.weight) + self.bias


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul(torch.tanh(torch.nn.functional.softplus(x)))


class SimpleMLP(BaseEstimator):
    def __init__(self, is_classification: bool, device: str = 'cpu'):
        self.is_classification = is_classification
        self.device = device

    def fit(self, X, y, X_val=None, y_val=None):
        # print(f'fit {X=}')
        input_dim = X.shape[1]
        is_classification = self.is_classification

        output_dim = 1 if len(y.shape) == 1 else y.shape[1]

        if self.is_classification:
            self.class_enc_ = OrdinalEncoder(dtype=np.int64)
            y = self.class_enc_.fit_transform(y[:, None])[:, 0]
            self.classes_ = self.class_enc_.categories_[0]
            output_dim = len(self.class_enc_.categories_[0])
        else:  # standardize targets
            self.y_mean_ = np.mean(y, axis=0)
            self.y_std_ = np.std(y, axis=0)
            y = (y - self.y_mean_) / (self.y_std_ + 1e-30)
            if y_val is not None:
                y_val = (y_val - self.y_mean_) / (self.y_std_ + 1e-30)

        act = nn.SELU if is_classification else Mish
        model = nn.Sequential(
            ScalingLayer(input_dim),
            NTPLinear(input_dim, 256), act(),
            NTPLinear(256, 256), act(),
            NTPLinear(256, 256), act(),
            NTPLinear(256, output_dim, zero_init=True),
        ).to(self.device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1) if is_classification else nn.MSELoss()
        params = list(model.parameters())
        scale_params = [params[0]]
        weights = params[1::2]
        biases = params[2::2]
        opt = torch.optim.Adam([dict(params=scale_params), dict(params=weights), dict(params=biases)],
                                betas=(0.9, 0.95))

        x_train = torch.as_tensor(X, dtype=torch.float32)
        y_train = torch.as_tensor(y, dtype=torch.int64 if self.is_classification else torch.float32)
        if not is_classification and len(y_train.shape) == 1:
            y_train = y_train[:, None]

        if X_val is not None and y_val is not None:
            x_valid = torch.as_tensor(X_val, dtype=torch.float32)
            y_valid = torch.as_tensor(y_val, dtype=torch.int64 if self.is_classification else torch.float32)
            if not is_classification and len(y_valid.shape) == 1:
                y_valid = y_valid[:, None]
        else:
            x_valid = x_train[:0]
            y_valid = y_train[:0]

        train_ds = TensorDataset(x_train, y_train)
        valid_ds = TensorDataset(x_valid, y_valid)
        n_train = x_train.shape[0]
        n_valid = x_valid.shape[0]
        n_epochs = 256
        train_batch_size = min(256, n_train)
        valid_batch_size = max(1, min(1024, n_valid))

        def valid_metric(y_pred: torch.Tensor, y: torch.Tensor):
            if self.is_classification:
                # unnormalized classification error, could also convert to float and then take the mean
                return torch.sum(torch.argmax(y_pred, dim=-1) != y)
            else:
                # MSE
                return (y_pred - y).square().mean()

        train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, drop_last=True)
        valid_dl = DataLoader(valid_ds, batch_size=valid_batch_size, shuffle=False)

        n_train_batches = len(train_dl)
        base_lr = 0.04 if is_classification else 0.07
        best_valid_loss = np.Inf
        best_valid_params = None

        for epoch in range(n_epochs):
            # print(f'Epoch {epoch + 1}/{n_epochs}')
            for batch_idx, (x_batch, y_batch) in enumerate(train_dl):
                # set learning rates according to schedule
                t = (epoch * n_train_batches + batch_idx) / (n_epochs * n_train_batches)
                lr_sched_value = 0.5 - 0.5 * np.cos(2 * np.pi * np.log2(1 + 15 * t))
                lr = base_lr * lr_sched_value
                # print(f'{lr=:g}')
                opt.param_groups[0]['lr'] = 6 * lr  # for scale
                opt.param_groups[1]['lr'] = lr  # for weights
                opt.param_groups[2]['lr'] = 0.1 * lr  # for biases

                # optimization
                y_pred = model(x_batch.to(self.device))
                loss = criterion(y_pred, y_batch.to(self.device))
                loss.backward()
                opt.step()
                opt.zero_grad()
                # print(f'{loss.item()=:g}')

            # save parameters if validation score improves
            with torch.no_grad():
                if x_valid.shape[0] > 0.0:
                    y_pred_valid = torch.cat([model(x_batch.to(self.device)).detach() for x_batch, _ in valid_dl], dim=0)
                    valid_loss = valid_metric(y_pred_valid, y_valid.to(self.device)).cpu().item()
                else:
                    valid_loss = 0.0
                if valid_loss <= best_valid_loss:  # use <= for last best epoch
                    best_valid_loss = valid_loss
                    best_valid_params = [p.detach().clone() for p in model.parameters()]

        # after training, revert to best epoch
        with torch.no_grad():
            for p_model, p_copy in zip(model.parameters(), best_valid_params):
                p_model.set_(p_copy)

        self.model_ = model

        return self

    def predict(self, X):
        x = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            y_pred = self.model_(x).cpu().numpy()
        if self.is_classification:
            # return classes with highest probability
            return self.class_enc_.inverse_transform(np.argmax(y_pred, axis=-1)[:, None])[:, 0]
        else:
            return y_pred[:, 0] * self.y_std_ + self.y_mean_

    def predict_proba(self, X):
        assert self.is_classification
        self.model_.eval()
        x = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_pred = torch.softmax(self.model_(x), dim=-1).cpu().numpy()
        return y_pred


class Standalone_RealMLP_TD_S_Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, device: str = 'cpu'):
        self.device = device

    def fit(self, X, y, X_val=None, y_val=None):
        self.prep_ = get_realmlp_td_s_pipeline()
        self.model_ = SimpleMLP(is_classification=True, device=self.device)
        X = self.prep_.fit_transform(X)
        if X_val is not None:
            X_val = self.prep_.transform(X_val)
        self.model_.fit(X, y, X_val, y_val)
        self.classes_ = self.model_.classes_

    def predict(self, X):
        return self.model_.predict(self.prep_.transform(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self.prep_.transform(X))


class Standalone_RealMLP_TD_S_Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, device: str = 'cpu'):
        self.device = device

    def fit(self, X, y, X_val=None, y_val=None):
        self.prep_ = get_realmlp_td_s_pipeline()
        self.model_ = SimpleMLP(is_classification=False, device=self.device)
        X = self.prep_.fit_transform(X)
        if X_val is not None:
            X_val = self.prep_.transform(X_val)
        self.model_.fit(X, y, X_val, y_val)

    def predict(self, X):
        return self.model_.predict(self.prep_.transform(X))