from model import DeconvNet
import cv2
from torchvision.transforms import Compose, ToTensor
from typing import Union
import torch
import numpy as np
from matplotlib import pyplot as plt


def rescale(in_: Union[torch.Tensor, np.array]):
    return (in_ - in_.min())/(in_.max()-in_.min())


image = cv2.imread('puppy-dog.jpg')[:, :, ::-1]
tfms = Compose([ToTensor()])
image = cv2.resize(image, (224, 224))
image = tfms(image)
if torch.cuda.is_available():
    net = DeconvNet(pretrained=True).cuda()
    x = image.cuda()
else:
    net = DeconvNet(pretrained=True)
    x = image

_, y = net(x)

fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(3, 20))

for i, _ in enumerate(y):
    ax[i].imshow(rescale(np.transpose(
        y[i].detach().cpu().numpy()[0], (1, 2, 0))))
    ax[i].axis('off')
plt.show()
