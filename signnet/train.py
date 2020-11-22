import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.vgg import vgg11
from options.train_options import TrainOptions
from dlcommon.logger import TensorboardLogger
from dlcommon.json import save_dict


def train(args, model, device, train_loader, optimizer, epoch, logger=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        logger.step()

        if batch_idx % args.log_interval == 0:
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(pred) * 100.0

            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.8f}, Accuracy: {}/{} ({:.0f}%)'.format(
                epoch, args.epochs, batch_idx * len(data), len(train_loader.dataset),
                                    100. * batch_idx / len(train_loader),
                loss.item(), correct, len(pred), acc))

            if logger is not None:
                logger.add_scalar('Loss/train', loss.item())
                logger.add_scalar('Acc/train', acc)

            if args.dry_run:
                break


def validate(model, device, test_loader, logger=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for (data, target) in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_count += 1
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.8}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def get_dataloader(use_cuda, args):
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    train_dataset_path = os.path.join(args.data_path, args.train_set)
    test_dataset_path = os.path.join(args.data_path, args.val_set)

    dataset1 = datasets.ImageFolder(root=train_dataset_path, transform=transform)
    dataset2 = datasets.ImageFolder(root=test_dataset_path, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader


def save_label_dict(path, dataset):
    save_dict({'classes': dataset.classes, 'class2idx': dataset.class_to_idx}, path)


def main():
    args = TrainOptions().parse()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = get_dataloader(use_cuda, args)
    model = vgg11(len(train_loader.dataset.classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    if args.resume_name is not None:
        log_dir = os.path.join(args.checkpoints_dir, args.resume_name)
        checkpoints = torch.load(os.path.join(log_dir, 'checkpoint_latest.pth'))
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        logger = TensorboardLogger.load_state_dict(checkpoints['logger_state_dict'])
        start_epoch = checkpoints['epoch'] + 1
        print('Resume training from epoch ' + str(start_epoch))
    else:
        log_dir = os.path.join(args.checkpoints_dir, args.name)
        logger = TensorboardLogger(log_dir)
        save_label_dict(os.path.join(log_dir, 'class_map.json'), train_loader.dataset)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(start_epoch, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, logger)
        scheduler.step()

        if args.save_model and epoch % args.save_interval == 0:
            validate(model, device, test_loader, logger)
            # save_path = os.path.join(args.checkpoints_dir, args.name, 'checkpoint_' + str(epoch) + '.pth')
            save_path = os.path.join(log_dir, 'checkpoint_latest.pth')
            checkpoints = {'epoch': epoch,
                           'model_state_dict': model.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict(),
                           'logger_state_dict': logger.state_dict(),
                           }
            print('Save model into ' + save_path)
            torch.save(checkpoints, save_path)
            continue

        if epoch % args.val_interval == 0:
            validate(model, device, test_loader, logger)

    logger.close()


if __name__ == '__main__':
    main()
