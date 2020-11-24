# work around for:
# OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.vgg import vgg11
from options.test_options import TestOptions


def test(model, device, test_loader):
    model.eval()
    start = time.time()
    count = 0
    length = len(test_loader.dataset)
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            count += 1
            if count % 100 == 0:
                print('{}/{}'.format(count, length))
    total_time = time.time() - start
    avg = total_time / length

    print('Average Processing time: {}, FPS: {}'.format(avg, 1.0 / avg))


def get_dataloader(use_cuda, args):
    test_kwargs = {'batch_size': 1}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
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


def main():
    args = TestOptions().parse()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_loader = get_dataloader(use_cuda, args)
    model = vgg11(len(test_loader.dataset.classes)).to(device)

    log_dir = os.path.join(args.checkpoints_dir, args.resume_name)
    checkpoints = torch.load(os.path.join(log_dir, 'checkpoint_latest.pth'), map_location=device)
    model.load_state_dict(checkpoints['model_state_dict'])
    test(model, device, test_loader)


if __name__ == '__main__':
    main()
