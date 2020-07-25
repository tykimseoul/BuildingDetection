import rasterio as rio
from rasterio.mask import mask, raster_geometry_mask
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import re


def mask_image(id):
    dataset = rio.open('./AOI_4_Shanghai_Train/RGB-PanSharpen/RGB-PanSharpen_AOI_4_Shanghai_img{}.tif'.format(id))
    data_read = dataset.read()
    raw_image = parse_rgb(data_read)
    save_image(raw_image, './train_data/rgb/Shanghai_img{}.png'.format(id))
    geom = json.loads(open('./AOI_4_Shanghai_Train/geojson/buildings/buildings_AOI_4_Shanghai_img{}.geojson'.format(id)).read())
    geom = [p['geometry'] for p in geom['features']]
    if len(geom) == 0:
        empty_mask = np.zeros(raw_image.shape)
        masked_image = empty_mask
        out_mask = empty_mask
    else:
        out_image, _ = mask(dataset, geom, crop=False)
        masked_image = parse_rgb(out_image)
        out_mask, _, _ = raster_geometry_mask(dataset, geom)
    save_image(masked_image, './train_data/clipped/Shanghai_img{}.png'.format(id))
    save_image(out_mask, './train_data/mask/Shanghai_img{}.png'.format(id), 'gray_r')


def normalize_channel(tif, idx):
    channel = tif[:, :, idx]
    channel = channel / np.max(channel)
    return channel


def save_image(data, path, cmap='viridis'):
    # plt.figure()
    # plt.imshow(data, cmap=cmap)
    plt.imsave(path, data, cmap=cmap)
    # plt.show()


def parse_rgb(image):
    image = image.transpose(1, 2, 0)
    channels = tuple(map(lambda idx: normalize_channel(image, idx), [0, 1, 2]))
    data = np.dstack(channels)
    return data


def file_id(file_name):
    return int(re.search(r'img(\d+)', file_name).group(1))


images = sorted(os.listdir('./AOI_4_Shanghai_Train/RGB-PanSharpen'), key=file_id)
print(images)
for image in images[:100]:
    id = file_id(image)
    print(id)
    mask_image(id)
