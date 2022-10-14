import os
import random
import torch.cuda

os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
import time
from util import *
from trainer import Trainer

from model import FullModel
import setproctitle
from loguru import logger
from collections import OrderedDict

parser = argparse.ArgumentParser()
# setting params
parser.add_argument('--dataset', type=str, default='METR-LA')
parser.add_argument('--seq_in_len', type=int, default=12, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=12, help='output sequence length')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=207)

# model params
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--num_backbone_layers', type=int, default=7)
parser.add_argument('--num_paths', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--winsize', type=int, default=2)
parser.add_argument('--num_head_layers', type=int, default=2)
parser.add_argument('--num_hyper_edge', type=int, default=32)
parser.add_argument('--use_multi_scale', type=str_to_bool, default=True)
parser.add_argument('--use_hyper_graph', type=str_to_bool, default=True)
parser.add_argument('--use_interactive', type=str_to_bool, default=True)
parser.add_argument('--GSL', type=str_to_bool, default=False)
parser.add_argument('--biscale', type=str_to_bool, default=False)

# running params
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--runs', type=int, default=1, help='number of runs')
parser.add_argument('--tolerance', type=int, default=100, help='tolerance for earlystopping')
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--epochs', type=int, default=30, help='')

# logging params
parser.add_argument('--print_every', type=int, default=1, help='')
parser.add_argument('--save', type=str, default='./save/', help='save path')
parser.add_argument('--expid', type=str, default='1', help='experiment id')

args = parser.parse_args()

setproctitle.setproctitle(args.expid)
args.save = os.path.join(args.save, args.expid)
os.makedirs(args.save, exist_ok=True)
logger.add(os.path.join(args.save, 'log-{time}.txt'))
logger.add(os.path.join(args.save, 'result-log-{time}.txt'), level='WARNING')
logger.info(vars(args))

if args.dataset == 'PEMS03':
    args.stats = {'min': 0, 'max': 1852}
    args.feat_off = 1
    args.in_dim = 2
elif args.dataset == 'PEMS04':
    args.stats = {'min': 1.0, 'max': 919}
    args.feat_off = 3
    args.in_dim = 4
elif args.dataset == 'PEMS07':
    args.stats = {'min': 0, 'max': 1498}
    args.feat_off = 1
    args.in_dim = 2
elif args.dataset == 'PEMS08':
    args.stats = {'min': 1.0, 'max': 1147}
    args.feat_off = 3
    args.in_dim = 4

torch.set_num_threads(3)
device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
dataloader = load_dataset(args, args.data, args.batch_size, args.batch_size, args.batch_size)
scaler = dataloader['scaler']
predefined_A, org_adj = load_sym_adj(args, args.adj_data)
predefined_adjs, _ = load_adj(args, args.adj_data)
args.original_adj = torch.tensor(org_adj).to(device)
args.predefined_adj = torch.tensor(predefined_A).to(device)
args.predefined_adjs = [torch.tensor(adj).to(device) for adj in predefined_adjs]
args.scaler = scaler
args.device = device

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def test(engine):
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testy = torch.Tensor(y).to(device)
        input = testx  # B x T x N x 2
        tod_idx = (input[:, :, 0, args.feat_off] * 288).long()  # B x T
        data = {
            'feat': input[:, :, :, :args.feat_off + 1],
            'tod_idx': tod_idx,
            'dow_onehot': input[:, :, 0, args.feat_off + 1:args.feat_off + 8],
            'target': testy
        }
        with torch.no_grad():
            preds = engine.model(data)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze(dim=1))
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    if args.feat_off == 1:
        pred = scaler.inverse_transform(yhat)
    else:
        pred = scaler[0].inverse_transform(yhat)
    pred = torch.clamp(pred, args.stats['min'], args.stats['max'])
    metrics = metric(pred, realy)
    log = '[Testing] Overall, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
    logger.info(log.format(metrics[0], metrics[1], metrics[2]))


def main(runid):
    logger.info("runid = %s" % runid)
    model = FullModel(args)

    nParams = sum([p.nelement() for p in model.parameters()])
    logger.info('Number of model parameters is %s' % nParams)

    engine = Trainer(args, model, args.learning_rate, args.weight_decay, args.clip, args.seq_out_len, scaler, device)
    logger.info("start training...")
    his_loss = []
    val_time = []
    train_time = []
    minl = 1e5
    epoch_best = -1
    tolerance = args.tolerance
    count_lfx = 0
    batches_seen = 0

    for i in range(1, args.epochs + 1):
        dataloader['train_loader'].shuffle()
        metrics_list = OrderedDict(**{'loss': [], 'mae': [], 'rmse': [], 'mape': []})
        t1 = time.time()
        for iter, (x, y, ycl) in enumerate(dataloader['train_loader'].get_iterator()):
            batches_seen += 1
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            metrics = engine.train(trainx,
                                   trainy[:, :, :, 0],
                                   trainy)
            for k, v in metrics.items():
                metrics_list[k].append(v)
        t2 = time.time()
        train_time.append(t2 - t1)

        val_metrics_list = OrderedDict(**{'mae': [], 'rmse': [], 'mape': []})
        s1 = time.time()
        for iter, (x, y) in enumerate(
                dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testy = torch.Tensor(y).to(device)
            metrics = engine.eval(testx, testy[:, :, :, 0], testy)
            for k, v in metrics.items():
                val_metrics_list[k].append(v)
        s2 = time.time()
        val_time.append(s2 - s1)

        for k, v in metrics_list.items():
            metrics_list[k] = np.mean(metrics_list[k])
        for k, v in val_metrics_list.items():
            val_metrics_list[k] = np.mean(val_metrics_list[k])

        his_loss.append(val_metrics_list['mae'])
        engine.scheduler.step()

        log = 'Epoch: {:03d}'.format(i)
        logger.info(log)
        log = '[Training]'
        for k, v in metrics_list.items():
            log += ' {}: {:.4f},'.format(k, v)
        log += ' Training time: {:.1f}/epoch'.format(t2 - t1)
        logger.info(log)
        log = '[Validation]'
        for k, v in val_metrics_list.items():
            log += ' {}: {:.4f},'.format(k, v)
        log += ' Inference time: {:.1f}'.format(s2 - s1)
        logger.info(log)

        if val_metrics_list['mae'] < minl:
            torch.save(engine.model.state_dict(), os.path.join(args.save, 'ckpt_%s.pth' % runid))
            minl = val_metrics_list['mae']
            epoch_best = i
            count_lfx = 0
        else:
            count_lfx += 1
            if count_lfx > tolerance:
                break
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(os.path.join(args.save, 'ckpt_%s.pth' % runid), map_location='cpu'))
    logger.info("The valid loss on best model is {:.4f}, epoch:{}".format(his_loss[bestid], epoch_best))
    test(engine)


if __name__ == "__main__":
    for i in range(args.runs):
        main(i)

