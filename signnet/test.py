import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from options.test_options import TestOptions
from dlcommon.json import save_dict
from model.vgg import vgg11


def test(model, device, test_loader):
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
    test_dataset_path = os.path.join(args.data_path, args.val_set)

    dataset2 = datasets.ImageFolder(root=test_dataset_path, transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return test_loader


def save_label_dict(path, dataset):
    save_dict({'classes': dataset.classes, 'class2idx': dataset.class_to_idx}, path)


def main():
    args = TestOptions().parse()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_loader = get_dataloader(use_cuda, args)
    model = vgg11(len(test_loader.dataset.classes)).to(device)

    log_dir = os.path.join(args.checkpoints_dir, args.resume_name)
    checkpoints = torch.load(os.path.join(log_dir, 'checkpoint_latest.pth'))
    model.load_state_dict(checkpoints['model_state_dict'])

    test(model, device, test_loader)


if __name__ == '__main__':
    main()
