import sys
import cv2
import time
import torch
import numpy as np
from tqdm import tqdm as tqdm
from segmentation_models_pytorch.utils.meter import AverageValueMeter


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y, result_epoch_path):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def save_image(self, x, y, z, result_epoch_path):
        y = (y > 0.5).float()
        z = (z > 0.5).float()
        x = self.normalize_image(x)
        y = self.normalize_label(y)
        z = self.normalize_label(z)

        all = np.hstack((np.hstack((x, y)), z))
        cv2.imwrite(result_epoch_path + "/" + str(int(time.time())) + ".png", all)

        # cv2.imwrite(result_epoch_path + "/x_" + str(int(time.time())) + ".png", x)
        # cv2.imwrite(result_epoch_path + "/y_" + str(int(time.time())) + ".png", y)
        # cv2.imwrite(result_epoch_path + "/z_" + str(int(time.time())) + ".png", z)

    def normalize_image(self, x):
        x = x.cpu().detach().numpy()
        if x.ndim == 4:
            x = x[0]
        if x.ndim == 3:
            x = np.moveaxis(x, 0, -1)

        x = self.one_channel_to_three(x) * 255.0

        return x

    def normalize_label(self, x):
        x = x.cpu().detach().numpy()
        if x.ndim == 4:
            x = x[0]
        if x.ndim == 3:
            x1 = x[0]
            x2 = x[1]
            x1 = self.one_channel_to_three(x1) * 255.0
            x2 = self.one_channel_to_three(x2) * 255.0
            x = np.hstack((x1, x2))

        return x

    def one_channel_to_three(self, x):
        if x.shape[-1] != 3:
            x = cv2.merge((x, x, x))
        return x

    def run(self, dataloader, result_epoch_path=""):
        self.on_epoch_start()
        logs = {}
        loss_meter = AverageValueMeter()
        writer_iter = 0
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                for aug in range(y.shape[1]):
                    x_single, y_single = torch.unsqueeze(x[0][aug].to(self.device, dtype=torch.float), 0), \
                                         torch.unsqueeze(y[0][aug].to(self.device, dtype=torch.float), 0)

                    loss, y_pred = self.batch_update(x_single, y_single, result_epoch_path)

                    # update loss logs
                    loss_value = loss.cpu().detach().numpy()
                    loss_meter.add(loss_value)
                    loss_logs = {self.loss.__name__: loss_meter.mean}
                    logs.update(loss_logs)

                    # update metrics logs
                    for metric_fn in self.metrics:
                        metric_value = metric_fn(y_pred, y_single).cpu().detach().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value)
                    metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                    logs.update(metrics_logs)

                    if self.verbose:
                        s = self._format_logs(logs)
                        iterator.set_postfix_str(s)

                    # Save image
                    if "Train" in result_epoch_path:
                        if writer_iter % 20 == 0:
                            self.save_image(x_single, y_single, y_pred, result_epoch_path)
                    else:
                        self.save_image(x_single, y_single, y_pred, result_epoch_path)

                    writer_iter += 1

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, result_epoch_path):
        self.optimizer.zero_grad()
        x = x.type(torch.FloatTensor).to(self.device)

        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)

        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, result_epoch_path):
        with torch.no_grad():

            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)

            self.save_image(x, y, prediction, result_epoch_path)
        return loss, prediction

