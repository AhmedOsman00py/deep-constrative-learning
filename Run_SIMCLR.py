import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

from baseline.utils import *
from baseline.training import training

from baseline.models.encoder import Encoder
from baseline.models.linear_classifier import LinearClassifier, TwoLayersClassifier
from baseline.models.Simclr import Simclr

from pre_training_encoder import pretraining
from Contrastive_loss import Contrastive_loss
from data_augmentation import get_contrastive_loader


class experiment:
    def __init__(
        self, numbers_exp, contrastive_loader, train_loader, val_loader, device
    ):
        self.numbers_exp = numbers_exp
        self.contrastive_loader = contrastive_loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def pre_training(self, num_epochs, model, optimizer, criterion, use_scheduler=True):
        device = self.device
        model = model.to(device)
        criterion = criterion.to(device)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        contrastive_loss, contrastive_lr = [], []
        contrastive_loader = self.contrastive_loader
        for epoch in range(1, num_epochs + 1):
            loss, lr = pretraining(
                epoch, model, contrastive_loader, optimizer, criterion
            )
            if use_scheduler:
                scheduler.step()
            contrastive_loss.append(loss)
            contrastive_lr.append(lr)
        plt.figure(figsize=(8, 8))
        plt.plot(
            range(1, len(contrastive_lr) + 1),
            contrastive_lr,
            color="b",
            label="learning rate",
        )
        plt.legend(), plt.ylabel("learning_rate"), plt.xlabel("epochs"), plt.title(
            "Learning Rate"
        ),
        plt.savefig("results/Learning rate.jpeg")

        plt.figure(figsize=(8, 8))
        plt.plot(
            range(1, len(contrastive_loss) + 1),
            contrastive_loss,
            color="b",
            label="loss",
        )
        plt.legend(), plt.ylabel("loss"), plt.xlabel("epochs"), plt.title("Loss"),
        plt.savefig("results/Loss function.jpeg")

        PATH = "results/pre_trained_simclr.pth"
        torch.save(model.state_dict(), PATH)

    def training(self, epochs, model, optimizer, criterion, nbr_layer):
        PATH = "results/pre_trained_supcon.pth"
        model.load_state_dict(torch.load(PATH))
        for name, param in model.named_parameters():
            param.requires_grad = False

        device = self.device
        metric = get_accuracy
        train_loader = self.train_loader
        val_loader = self.val_loader
        if nbr_layer == 1:
            classifier = LinearClassifier()
        else:
            classifier = TwoLayersClassifier()
        model.head = classifier
        output_fn = torch.nn.Softmax(dim=1)
        train_class = training(
            train_loader=train_loader,
            val_loader=val_loader,
            metric=metric,
            device=device,
        )
        train_class.train(
            model=model,
            epochs=epochs,
            optimizer=optimizer,
            criterion=criterion,
            output_fn=output_fn,
            RGB=False,
            patience_LR=3,
            patience_earlystop=7,
        )
        plt.figure(figsize=(8, 8)), plt.plot(
            range(1, epochs + 1), train_class.loss_train, label="train loss"
        ),
        plt.plot(range(1, epochs + 1), train_class.loss_valid, label="valid loss"),
        plt.xlabel("epochs"), plt.ylabel("Loss"), plt.title("Loss functions SIMCLR"),
        plt.legend(), plt.savefig("results/Loss functions SIMCLR.png")

        plt.figure(figsize=(8, 8)), plt.plot(
            range(1, epochs + 1), train_class.acc_train, label="train accuracy"
        ),
        plt.plot(range(1, epochs + 1), train_class.acc_valid, label="valid accuracy"),
        plt.xlabel("epochs"), plt.ylabel("Loss"), plt.title(
            "Accuracy functions SIMCLR"
        ),
        plt.legend(), plt.savefig("results/Accuracy functions SIMCLR.png")

        self.acc_valid = train_class.acc_valid[epochs - 1]
        PATH = "results/trained_simclr.pth"
        torch.save(model.state_dict(), PATH)


def main():
    train_data, test_data = get_data()
    _, train_loader, test_loader = get_loader(train_data, test_data)
    contrastive_loader = get_contrastive_loader(batch_size=64)
    device = get_device()
    numbers_exp = 5
    exp = experiment(numbers_exp, contrastive_loader, train_loader, test_loader, device)
    # Pre-training
    epochs_pretraining = 100
    encoder = Encoder()
    head_type = "mlp"
    feat_dim = 128
    model = Simclr(encoder, head_type, feat_dim)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3
    )
    criterion = Contrastive_loss(temperature=0.07)
    exp.pre_training(epochs_pretraining, model, optimizer, criterion)
    # Training
    accuracy_list = []
    for k in range(numbers_exp):
        epochs = 25
        encoder = Encoder()
        model = Simclr(encoder, head_type, feat_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-3,
        )
        nbr_layer = 1
        exp.training(epochs, model, optimizer, criterion, nbr_layer)
        accuracy_list.append(exp.acc_valid)
    mean_accuracy = np.mean(accuracy_list)
    std_accuracy = np.std(accuracy_list)
    print(f"The mean accuracy is {mean_accuracy}")
    print(f"The std of accuracy is {std_accuracy}")


if __name__ == "__main__":
    main()
