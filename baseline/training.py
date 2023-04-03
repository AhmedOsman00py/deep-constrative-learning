from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False


class training:
    def __init__(self, train_loader, val_loader, metric, device):
        super(training, self).__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metric = metric
        self.device = device

    def train(
        self,
        model,
        epochs,
        optimizer,
        criterion,
        output_fn,
        RGB=False,
        patience_LR=3,
        patience_earlystop=8,
    ):
        loss_valid, acc_valid = [], []
        loss_train, acc_train = [], []
        model = model.to(self.device)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=patience_LR)
        early_stopping = EarlyStopping(patience=patience_earlystop, delta=0)

        for epoch in tqdm(range(epochs)):
            # Training
            model.train()
            for idx, batch in enumerate(self.train_loader):
                # get the inputs; batch is a list of [inputs, labels]
                inputs, labels = batch
                if RGB:
                    inputs = torch.cat([inputs, inputs, inputs], dim=1)
                inputs = inputs.to(self.device)  # train on GPU
                labels = labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                out = model(x=inputs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                # compute loss and accuracy after an epoch on the train and valid set
            model.eval()
            with torch.no_grad():  # since we're not training, we don't need to calculate the gradients for our outputs
                idx = 0
                for batch in self.val_loader:
                    inputs, labels = batch
                    if RGB:
                        inputs = torch.cat([inputs, inputs, inputs], dim=1)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    if idx == 0:
                        t_out = model(x=inputs)
                        t_loss = criterion(t_out, labels).view(1).item()
                        t_out = output_fn(t_out).detach().cpu().numpy()
                        t_out = t_out.argmax(
                            axis=1
                        )  # the class with the highest energy is what we choose as prediction
                        ground_truth = labels.detach().cpu().numpy()
                    else:
                        out = model(x=inputs)
                        t_loss = np.hstack((t_loss, criterion(out, labels).item()))
                        t_out = np.hstack(
                            (
                                t_out,
                                output_fn(out).argmax(axis=1).detach().cpu().numpy(),
                            )
                        )
                        ground_truth = np.hstack(
                            (ground_truth, labels.detach().cpu().numpy())
                        )
                    idx += 1

                acc_valid.append(self.metric(ground_truth, t_out))
                val_loss = np.mean(t_loss)
                loss_valid.append(val_loss)
                scheduler.step(val_loss)

            if early_stopping(val_loss):
                print("Early stopping after epoch", epoch)
                break

            with torch.no_grad():
                idx = 0
                for batch in self.train_loader:
                    inputs, labels = batch
                    if RGB:
                        inputs = torch.cat([inputs, inputs, inputs], dim=1)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    if idx == 0:
                        t_out = model(x=inputs)
                        t_loss = criterion(t_out, labels).view(1).item()
                        t_out = output_fn(t_out).detach().cpu().numpy()
                        t_out = t_out.argmax(axis=1)
                        ground_truth = labels.detach().cpu().numpy()
                    else:
                        out = model(x=inputs)
                        t_loss = np.hstack((t_loss, criterion(out, labels).item()))
                        t_out = np.hstack(
                            (
                                t_out,
                                output_fn(out).argmax(axis=1).detach().cpu().numpy(),
                            )
                        )
                        ground_truth = np.hstack(
                            (ground_truth, labels.detach().cpu().numpy())
                        )
                    idx += 1

            acc_train.append(self.metric(ground_truth, t_out))
            loss_train.append(np.mean(t_loss))

            print(
                "| Epoch: {}/{} | Train: Loss {:.4f} Accuracy : {:.4f} "
                "| Val: Loss {:.4f} Accuracy : {:.4f}\n".format(
                    epoch + 1,
                    epochs,
                    loss_train[epoch],
                    acc_train[epoch],
                    loss_valid[epoch],
                    acc_valid[epoch],
                )
            )

        self.model = model
        self.loss_train = loss_train
        self.loss_valid = loss_valid
        self.acc_train = acc_train
        self.acc_valid = acc_valid
