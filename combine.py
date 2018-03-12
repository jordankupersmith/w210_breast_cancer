from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import os
import sys

# imgs = raw_input('In which directory are the images(defalut:"images")?\n')
# if imgs == '':
#     imgs = 'images'
# os.chdir(imgs)
#
# dataset_size = raw_input('how large do you want your dataset in bytes?(default:1,000,000,000 or 1GB): n=')
# if dataset_size == '':
#     dataset_size = 10**9
# else:
#     dataset_size = int(dataset_size)
imgs = 'images'
os.chdir(imgs)
dataset_size = 10**9
images = [x for x in os.listdir('.') if x.endswith('.png')]

pos = []
neg = []
labels = []

for img in tqdm(images):
    image = Image.open(img)

    labels.append(img[-5])
    if img.endswith('_1.png'):
        pos.append(np.array(image.getdata()).reshape(128, 128).astype(np.uint8))
    elif img.endswith('_0.png'):
        neg.append(np.array(image.getdata()).reshape(128, 128).astype(np.uint8))
    if len(pos+neg)%5000 == 0:
        data = pos + random.sample(neg, len(pos))
        dataset = np.stack(data, axis=0)
        nplabels = np.stack(labels, axis=0)
        np.save('dataset.npy', dataset)
        np.save('labels.npy', nplabels)
        if dataset.nbytes > dataset_size:
            break
    image.close()
#
# dataset = np.stack(arrays, axis=0)
# labels = np.stack(labels, axis=0)
# np.save('dataset.npy', dataset)
# np.save('labels.npy', labels)
