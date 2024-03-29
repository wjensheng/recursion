from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import random
import numpy as np
from torchvision import transforms as T
from albumentations import Resize
from albumentations import Compose, RandomRotate90, Flip, Transpose, Resize


def tta_transform(split='train',
                  size=512,
                  num_tta=1,
                  per_image_norm=False,
                  **_):
    resize = Resize(height=size, width=size, always_apply=True)
    means = np.array([127.5, 127.5, 127.5, 127.5, 127.5, 127.5])
    stds = np.array([255.0, 255.0, 255.0, 255.0, 255.0, 255.0])

    def transform(image):
        if size != image.shape[0]:
            image = resize(image=image)['image']
        image = image.astype(np.float32)

        if per_image_norm:
            mean = np.mean(image.reshape(-1, 6), axis=0)
            std = np.std(image.reshape(-1, 6), axis=0)
            image -= mean
            image /= (std + 0.0000001)
        else:
            image -= means
            image /= stds

        if num_tta == 1:
            images = [image]
        else:
            assert num_tta == 4 or num_tta == 8
            images = [image]
            images.append(np.fliplr(image))
            images.append(np.flipud(image))
            images.append(np.fliplr(images[-1]))
            if num_tta == 8:
                images.append(np.transpose(image, (1,0,2)))
                images.append(np.flipud(images[-1]))
                images.append(np.fliplr(images[-2]))
                images.append(np.flipud(images[-1]))

        images = np.stack(images, axis=0)
        images = np.transpose(images, (0, 3, 1, 2))
        assert images.shape == (num_tta, 6, size, size), 'shape: {}'.format(images.shape)

        return images

    return transform