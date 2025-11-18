"""
Usage:
python main.py --model PointMLP --msg demo
"""
import argparse
import os
import logging
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from data import ModelNet40
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.optim.lr_scheduler import SequentialLR
import sklearn.metrics as metrics
import numpy as np
import math
import pandas as pd

MSE = torch.nn.MSELoss()
MAE = torch.nn.L1Loss()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=50, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, epoch, optimizer, best_train_loss, best_test_loss, true_values=None, pred_values=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, best_train_loss, best_test_loss, true_values, pred_values)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, best_train_loss, best_test_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer, best_train_loss, best_test_loss, true_values=None, pred_values=None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'best_test_loss': best_test_loss,
            'best_train_loss': best_train_loss,
        }
        torch.save(state, self.path)
        self.val_loss_min = val_loss


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='PointNet', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training')
    parser.add_argument('--min_lr', default=0.00005, type=float, help='min lr')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='decay rate')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='workers')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='early stopping min_delta')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='warmup epochs')
    parser.add_argument('--warmup_start_lr', default=0.00005, type=float, help='warmup starting learning rate')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."
    device = 'cuda'
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.set_printoptions(10)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg is None:
        message = time_str
    else:
        message = "-" + args.msg
    args.checkpoint = 'checkpoints/' + args.model + message + '-' + str(args.seed)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    # Model
    printf(f"args: {args}")
    printf('==> Building model..')
    #net = models.__dict__[args.model]()
    #criterion = cal_loss
    net = models.__dict__[args.model](num_classes=1)  # 添加num_classes=1参数
    criterion = torch.nn.MSELoss()
    net = net.to(device)
    # criterion = criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    #best_test_acc = 0.  # best test accuracy
    #best_train_acc = 0.
    #best_test_acc_avg = 0.
    #best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    optimizer_dict = None

    if args.resume:
        checkpoint_path = os.path.join(args.resume, "last_checkpoint.pth")
        printf(f"Resuming checkpoint from {checkpoint_path}")
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            #best_test_acc = checkpoint['best_test_acc']
            #best_train_acc = checkpoint['best_train_acc']
            #best_test_acc_avg = checkpoint['best_test_acc_avg']
            #best_train_acc_avg = checkpoint['best_train_acc_avg']
            best_test_loss = checkpoint['best_test_loss']
            best_train_loss = checkpoint['best_train_loss']
            optimizer_dict = checkpoint['optimizer']
        else:
            printf(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")

    if not os.path.isfile(os.path.join(args.checkpoint, "last_checkpoint.pth")):
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model)
        logger.set_names(["Epoch-Num", 'Learning-Rate',
                          'Train-Loss',
                          'Valid-Loss'])
    else:
        printf(f"Resuming last checkpoint from {args.checkpoint}")
        checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        #best_test_acc = checkpoint['best_test_acc']
        #best_train_acc = checkpoint['best_train_acc']
        #best_test_acc_avg = checkpoint['best_test_acc_avg']
        #best_train_acc_avg = checkpoint['best_train_acc_avg']
        best_test_loss = checkpoint['best_test_loss']
        best_train_loss = checkpoint['best_train_loss']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model, resume=True)
        optimizer_dict = checkpoint['optimizer']

    printf('==> Preparing data..')
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=args.workers,
                             batch_size=args.batch_size // 2, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_factor = args.warmup_start_lr / args.learning_rate
    def warmup_lambda(current_epoch):
        if current_epoch < args.warmup_epochs:
            # 线性插值: start + (end - start) * progress
            return start_factor + (1 - start_factor) * (current_epoch / args.warmup_epochs)
        else:
            return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch - args.warmup_epochs, eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs])

    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    #scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr, last_epoch=start_epoch - 1)

    if args.patience > 0:
        early_stopping = EarlyStopping(patience=args.patience, delta=args.min_delta)
    else:
        early_stopping = EarlyStopping(patience=args.epoch, delta=args.min_delta)

    if best_test_loss != float("inf"):
        early_stopping.val_loss_min = best_test_loss
        early_stopping.best_score = -best_test_loss

    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(net, train_loader, optimizer, criterion, device)  # {"loss", "acc", "acc_avg", "time"}
        test_out = validate(net, test_loader, criterion, device)
        scheduler.step()

        if test_out["loss"] < best_test_loss:
            best_test_loss = test_out["loss"]
            is_best = True
            df = pd.DataFrame({
                'True Values': test_out["true_values"].flatten(), 
                'Predicted Values': test_out["pred_values"].flatten()
            })
            df.to_csv(os.path.join(args.checkpoint, 'predictions.csv'), index=False)
        else:
            is_best = False

        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss

        early_stopping(test_out["loss"], net, epoch, optimizer, best_train_loss, best_test_loss,
                       test_out["true_values"], test_out["pred_values"])

        save_model(
            net, epoch, path=args.checkpoint, acc=test_out["loss"], is_best=is_best,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict()
        )

        logger.append([epoch, optimizer.param_groups[0]['lr'],
                       train_out["loss"],test_out["loss"]])
        printf(
            f"Training loss:{train_out['loss']} Train MSE:{train_out['mse']} Train RMSE:{train_out['rmse']} Train MAE:{train_out['mae']} ")
        printf(
            f"Testing loss:{test_out['loss']} Test MSE:{test_out['mse']} Test RMSE:{test_out['rmse']} Test MAE:{test_out['mae']} ")
        printf(
            f"train time:{train_out['time']}s test time:{test_out['time']}s [best test loss: {best_test_loss}] \n\n")
        
        if early_stopping.early_stop:
            printf("Early stopping triggered!")
            printf(f"Best validation loss was: {early_stopping.val_loss_min:.6f}")
            break

    logger.close()

    printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
    printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
    printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
    if early_stopping.early_stop:
        printf(f"++  Training stopped early at epoch {epoch + 1} due to no improvement in validation loss  ++")
    printf(f"++++++++" * 5)


def train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    # --- 累加指标 ---
    total_mse = 0
    total_mae = 0
    # ---
    time_cost = datetime.datetime.now()
    for batch_idx, (data, label, scan) in enumerate(trainloader):
        data, label, scan = data.to(device), label.to(device).squeeze(), scan.to(device)
        data = data.permute(0, 2, 1)  # so, the input data shape is [batch, 3, 1024]
        optimizer.zero_grad()
        #logits = net(data)
        #loss = criterion(logits, label)
        outputs = net(data, scan)
        loss = criterion(outputs.squeeze(), label.float())
        # --- 累加每个批次的MSE和MAE ---
        total_mse += MSE(outputs.squeeze(), label).item()
        total_mae += MAE(outputs.squeeze(), label).item()
        # ---
        #loss = torch.sqrt(criterion(outputs.squeeze(), label.float()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        train_loss += loss.item()
        #preds = logits.max(dim=1)[1]

        train_true.append(label.cpu().numpy().reshape(-1))
        train_pred.append(outputs.detach().cpu().numpy().reshape(-1))

        total += data.size(0)
        #correct += preds.eq(label).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.6f ' % (train_loss / (batch_idx + 1)))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    # --- 计算整个周期的平均指标 ---
    avg_mse = total_mse / (batch_idx + 1)
    avg_mae = total_mae / (batch_idx + 1)
    avg_rmse = math.sqrt(avg_mse)
    # ---
    return {
        "loss": float("%.6f" % (train_loss / (batch_idx + 1))),
        "mse": float("%.6f" % avg_mse),
        "rmse": float("%.6f" % avg_rmse),
        "mae": float("%.6f" % avg_mae),
        "time": time_cost
    }


def validate(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    # --- 累加指标 ---
    total_mse = 0
    total_mae = 0
    # ---
    with torch.no_grad():
        for batch_idx, (data, label, scan) in enumerate(testloader):
            data, label, scan = data.to(device), label.to(device).squeeze(), scan.to(device)
            data = data.permute(0, 2, 1)
            #logits = net(data)
            #loss = criterion(logits, label)
            outputs = net(data, scan)
            loss = criterion(outputs.squeeze(), label.float())
            # --- 累加每个批次的MSE和MAE ---
            total_mse += MSE(outputs.squeeze(), label).item()
            total_mae += MAE(outputs.squeeze(), label).item()
            # ---
            #loss = torch.sqrt(criterion(outputs.squeeze(), label.float()))
            test_loss += loss.item()
            #preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy().reshape(-1))
            test_pred.append(outputs.detach().cpu().numpy().reshape(-1))
            total += data.size(0)
            #correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.6f' % (test_loss / (batch_idx + 1)))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    # --- 计算整个周期的平均指标 ---
    avg_mse = total_mse / (batch_idx + 1)
    avg_mae = total_mae / (batch_idx + 1)
    avg_rmse = math.sqrt(avg_mse)
    # ---
    return {
        "loss": float("%.6f" % (test_loss / (batch_idx + 1))),
        "mse": float("%.6f" % avg_mse),
        "rmse": float("%.6f" % avg_rmse),
        "mae": float("%.6f" % avg_mae),
        "time": time_cost,
        "true_values": test_true,
        "pred_values": test_pred
    }


if __name__ == '__main__':
    main()
