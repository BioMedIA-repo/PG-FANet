import torch
import torchvision.transforms as tvt

OPTIM = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# train_transforms = tvt.Compose([
#     # transforms.Resize((224, 224)),
#     tvt.ToPILImage(),
#     tvt.RandomAffine([-90, 90], translate=[0.01, 0.1],
#                             scale=[0.9, 1.1]),
#     tvt.RandomRotation((-10, 10)),
#     tvt.RandomHorizontalFlip(),
#     tvt.RandomVerticalFlip(),
#     tvt.ToTensor(),
#     tvt.Normalize(mean=mean,
#                   std=std)
# ])
#
# gt_transform = tvt.Compose([
#     tvt.ToPILImage(),
#     tvt.RandomAffine([-90, 90], translate=[0.01, 0.1],
#                             scale=[0.9, 1.1]),
#     tvt.RandomRotation((-10, 10)),
#     tvt.RandomHorizontalFlip(),
#     tvt.RandomVerticalFlip(),
#     tvt.ToTensor(),
# ])

active_transform = tvt.Compose([
    # tvt.ToPILImage(),
    # tvt.ColorJitter(0.02, 0.02, 0.02, 0.01),
    tvt.ToTensor(),
    tvt.Normalize(mean=mean,
                  std=std)
])
