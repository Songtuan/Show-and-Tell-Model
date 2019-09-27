import torch
from torch.utils.data import Dataset
import torchvision.transforms as trn
import h5py


class CaptionDataset(Dataset):
    def __init__(self, input_file, transform=None):
        h = h5py.File(input_file)
        self.imgs = h['images']
        self.captions = h['captions']
        self.captions_per_img = h.attrs['captions_per_image']
        # self.captions_unencode = h['captions_uncode']
        assert self.captions.shape[0] // self.imgs.shape[0] == self.captions_per_img

        if transform is not None:
            self.transform = transform
        else:
            self.transform = trn.Compose([trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        assert self.imgs.shape[0] * 1 == self.captions.shape[0]

    def __getitem__(self, item):
        img = self.imgs[item // self.captions_per_img]
        img = trn.ToTensor()(img)
        if img[img > 1].shape[0] != 0 or img[img < 0].shape[0] != 0:
            img = self.transform(img)
        assert img.shape == torch.Size([3, 256, 256])

        caption = self.captions[item]
        caption = torch.from_numpy(caption).long()

        # caption_unencode = self.captions_unencode[item]

        # data = {'image': img, 'caption': caption, 'caption_unencode': caption_unencode}
        data = {'image': img, 'caption': caption}
        return data

    def __len__(self):
        return self.captions.shape[0]