import torch.distributions
import torch.optim as optim
from model import *
import util
import copy
import torch.nn.functional as F


class Trainer():
    def __init__(self, args, model, lrate, wdecay, clip, seq_out_len, scaler, device):
        self.args = args
        self.scaler = scaler
        self.device = device
        self.model = model
        self.model.to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 25, 0.5)
        self.clip = clip
        self.iter = 0
        self.seq_out_len = seq_out_len

    def train(self, input, real_val, target):
        self.iter += 1
        self.model.train()
        self.optimizer.zero_grad()

        tod_idx = (input[:, :, 0, self.args.feat_off] * 288).long()  # B x T
        args = self.args
        data = {
            'feat': input[:, :, :, :args.feat_off + 1],
            'tod_idx': tod_idx,
            'dow_onehot': input[:, :, 0, args.feat_off + 1:args.feat_off + 8],
            'target': target
        }
        output = self.model(data)

        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val.transpose(1, 2), dim=1)
        if self.args.feat_off == 1:
            predict = self.scaler.inverse_transform(output)
        else:
            predict = self.scaler[0].inverse_transform(output)

        mae = torch.abs(predict - real).mean()
        rmse = ((predict - real) ** 2).mean() ** 0.5
        mape = util.masked_mape(predict, real, 0)

        loss = mae
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return {
            'loss': loss.item(),
            'mae': mae.item(),
            'rmse': rmse.item(),
            'mape': mape.item()
        }

    def eval(self, input, real_val, target):
        self.model.eval()
        tod_idx = (input[:, :, 0, self.args.feat_off] * 288).long()  # B x T
        args = self.args
        data = {
            'feat': input[:, :, :, :args.feat_off + 1],
            'tod_idx': tod_idx,
            'dow_onehot': input[:, :, 0, args.feat_off + 1:args.feat_off + 8],
            'target': target
        }
        with torch.no_grad():
            output = self.model(data)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val.transpose(1, 2), dim=1)
        if self.args.feat_off == 1:
            predict = self.scaler.inverse_transform(output)
        else:
            predict = self.scaler[0].inverse_transform(output)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return {'mae': mae, 'rmse': rmse, 'mape': mape}
