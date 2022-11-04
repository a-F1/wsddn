import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import torch

import pynvml
import argparse

from tqdm import tqdm
from torch import optim
from wlip import Wlip
from wsddn import WSDDN
from my_dataset import myDataSet
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

# download pre_trained weights of vgg16
import torchvision.models.vgg


def parse_args():

        # Parse input arguments
        parser = argparse.ArgumentParser(description='Train WSDDN')

        parser.add_argument("--model", default='Wlip', help="model")
        parser.add_argument('--root_path', default='./', help='dataset')
        parser.add_argument("--optimizer", default='ADAM', help="optimizer")
        parser.add_argument("--lr", type=float, default=1e-8, help="Learning rate")
        parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
        parser.add_argument("--epochs", type=int, default=60, help="Epoch count")
        # default=5010
        parser.add_argument("--num_images", type=int, default=5010, help="Images count")
        parser.add_argument("--state_period", type=int, default=5, help="State saving period")
        parser.add_argument('--state_path', default='./weight/wlip_4_60.pt', help='dataset')

        return parser.parse_args()


def get_gpu_memory(handle):
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free = meminfo.free/1024/1024/1000
    return free


# def make_optimizer(args, my_model):
#     trainable = filter(lambda x: x.requires_grad, my_model.parameters())
#
#     if args.optimizer == 'SGD':
#         optimizer_function = optim.SGD
#         kwargs = {'momentum': args.momentum}
#     elif args.optimizer == 'ADAM':
#         optimizer_function = optim.Adam
#         kwargs = {}
#         # kwargs = {
#         #     'betas': (args.beta1, args.beta2),
#         #     'eps': args.epsilon
#         # }
#     elif args.optimizer == 'RMSprop':
#         optimizer_function = optim.RMSprop
#         kwargs = {'eps': args.epsilon}
#
#     kwargs['lr'] = args.lr
#     kwargs['weight_decay'] = args.wd
#
#     return optimizer_function(trainable, **kwargs)


def main():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        print("using {} device.".format(device))

        args = parse_args()
        print('args:')
        print(args)


        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(1)
        print('初始显存：%.4f G' % get_gpu_memory(handle))

        model = args.model
        root_path = args.root_path
        epochs = args.epochs
        lr = args.lr
        wd = args.wd
        num_images = args.num_images
        state_period = args.state_period
        state_path = args.state_path

        # Create the dataset and data loader
        # "trainval.txt"
        train_ds = myDataSet(root_path, "test.txt", num_images)
        train_dl = DataLoader(train_ds, num_workers=16, batch_size=1)

        # Create the network


        print(model)
        if model == 'Wlip':
            net = Wlip(use_checkpoint=False)
        else:
            net = WSDDN()

        print(net)
        net.to(device)
        net.train()
        net.load_state_dict(torch.load(state_path))
        print('加载 Wlip 模型后，剩余显存：%.4f G' % get_gpu_memory(handle))

        # 冻结 feature 的参数
        for name, param in net.named_parameters():
            # print(name)
            if "feature" in name:
                param.requires_grad = False

        # pg = [name for name, param in net.named_parameters() if not param.requires_grad]
        # print(pg)

        # Set the loss function and optimizer
        loss_function = net.calculate_loss
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        # optimizer = make_optimizer(args, net)
        scheduler = MultiStepLR(optimizer, milestones=[30, 60], gamma=0.5)

        # writer = SummaryWriter('./runs/log')

        for epoch in range(1, epochs + 1):

            # if epoch == 2:
            #     break

            mean_loss = torch.zeros(1).to(device)
            optimizer.zero_grad()
            train_bar = tqdm(train_dl, file=sys.stdout)

            # with torch.autograd.profiler.profile(
            #         enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:

            for step, data in enumerate(train_bar):
                # print(step)
                # if step == 1:
                #     break

                imgs = data[0].to(device)
                labels = data[1].to(device)
                boxes = data[2].to(device)

                combines_scores = net(imgs, boxes)
                # print(combines_scores.requires_grad)

                loss = loss_function(combines_scores, labels[0])
                loss.backward()

                mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
                train_bar.desc = "[epoch {}] mean loss {}, memory used {} G".format(
                    epoch, round(mean_loss.item(), 3), round(get_gpu_memory(handle), 4))
                # train_bar.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

                optimizer.step()

                # add loss, acc and lr into tensorboard
                # print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
                # tags = ["train_loss", "accuracy", "learning_rate"]
                # writer.add_scalar(tags[0], mean_loss, epoch)

            if epoch % state_period == 0:
                path = os.path.join(root_path, "weight", f"wlip_4_{epoch}.pt")
                torch.save(net.state_dict(), path)
                tqdm.write(f"State saved to {path}")

            # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            # prof.export_chrome_trace('./wlip_profile.json')

            scheduler.step()
        print('Finished Training')


if __name__ == "__main__":
        main()