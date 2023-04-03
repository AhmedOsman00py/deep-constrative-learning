import torch
from collections import defaultdict


class MetricMonitor:
    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name,
                    avg=metric["avg"],
                    float_precision=self.float_precision,
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def pretraining(epoch, model, contrastive_loader, optimizer, criterion):
    "Contrastive pre-training over an epoch"
    metric_monitor = MetricMonitor()
    model.train()
    for batch_idx, (data, labels) in enumerate(contrastive_loader):
        data = torch.cat([data[0], data[1]], dim=0)
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        data, labels = torch.autograd.Variable(data, False), torch.autograd.Variable(
            labels
        )
        bsz = labels.shape[0]
        features = model(data)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Learning Rate", optimizer.param_groups[0]["lr"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(
        "[Epoch: {epoch:03d}] Contrastive Pre-train | {metric_monitor}".format(
            epoch=epoch, metric_monitor=metric_monitor
        )
    )
    return (
        metric_monitor.metrics["Loss"]["avg"],
        metric_monitor.metrics["Learning Rate"]["avg"],
    )
