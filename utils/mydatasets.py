import torchvision
import torch


class MYCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, **kwargs):
        super(MYCIFAR10, self).__init__(**kwargs)

    def __getitem__(self, index):
        img, target = super(MYCIFAR10, self).__getitem__(index)
        diffculity = index / len(self.data)
        return img, target, index, diffculity

class MYIMAGENET_AUG(torchvision.datasets.ImageFolder):
    def __init__(self, repeat=10, **kwargs):
        super(MYIMAGENET_AUG, self).__init__(**kwargs)
        self.repeat = repeat

    def __getitem__(self, index):
        img_all = []
        target_all = []
        for i in range(self.repeat):
            img, target = super(MYIMAGENET_AUG, self).__getitem__(index)
            img_all.append(img)
            target_all.append(target)
        img_all = torch.stack(img_all)
        target_all = torch.tensor(target_all, dtype=torch.long)
        return img_all, target_all

class MYCIFAR100_AUG(torchvision.datasets.CIFAR100):
    def __init__(self, repeat=10, **kwargs):
        super(MYCIFAR100_AUG, self).__init__(**kwargs)
        self.repeat = repeat
        assert self.train == True

    def __getitem__(self, index):
        img_all = []
        target_all = []
        for i in range(self.repeat):
            img, target = super(MYCIFAR100_AUG, self).__getitem__(index)
            img_all.append(img)
            target_all.append(target)
        img_all = torch.stack(img_all)
        target_all = torch.tensor(target_all, dtype=torch.long)
        return img_all, target_all


class MYCIFAR10_AUG(torchvision.datasets.CIFAR10):
    def __init__(self, repeat=10, **kwargs):
        super(MYCIFAR10_AUG, self).__init__(**kwargs)
        self.repeat = repeat
        # assert self.train == True

    def __getitem__(self, index):
        img_all = []
        target_all = []
        for i in range(self.repeat):
            img, target = super(MYCIFAR10_AUG, self).__getitem__(index)
            img_all.append(img)
            target_all.append(target)
        img_all = torch.stack(img_all)
        target_all = torch.tensor(target_all, dtype=torch.long)
        return img_all, target_all

from PIL import Image

class MYCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, **kwargs):
        super(MYCIFAR10, self).__init__(**kwargs)

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class SVHN_AUG(torchvision.datasets.SVHN):
    def __init__(self, repeat=10, **kwargs):
        super(SVHN_AUG, self).__init__(**kwargs)
        self.repeat = repeat
        # assert self.train == True

    def __getitem__(self, index):
        img_all = []
        target_all = []
        for i in range(self.repeat):
            img, target = super(SVHN_AUG, self).__getitem__(index)
            img_all.append(img)
            target_all.append(target)
        img_all = torch.stack(img_all)
        target_all = torch.tensor(target_all, dtype=torch.long)
        return img_all, target_all


def my_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    # import pdb; pdb.set_trace()
    if isinstance(elem, torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    elif isinstance(elem, tuple):
        return ((my_collate(samples) for samples in zip(*batch)))
    else:
        raise NotImplementedError


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


if __name__ == "__main__":
    import torchvision.transforms as transforms

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = MYCIFAR10_AUG(repeat=10, root='/home/zequn/datatset/', train=True, download=True,
                             transform=transform_train)
    sampler = torch.utils.data.RandomSampler(trainset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=10, sampler=sampler,
                                               num_workers=2)  # , collate_fn = my_collate)
    for (data, labels) in train_loader:
        print(labels)
        print(data.shape)
        break
