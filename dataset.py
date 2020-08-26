import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def get_loader(batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    train_data = torchvision.datasets.CIFAR10(
        'data', train=True, transform=transform, download=True)
    test_data = torchvision.datasets.CIFAR10(
        'data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader