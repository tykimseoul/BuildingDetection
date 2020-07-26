import rasterio as rio
from rasterio.mask import mask, raster_geometry_mask
import numpy as np
import json
import os
import re
import cv2


def mask_image(id, training=True):
    if training:
        dataset = rio.open('./AOI_4_Shanghai_Train/RGB-PanSharpen/RGB-PanSharpen_AOI_4_Shanghai_img{}.tif'.format(id))
        data_read = dataset.read()
        raw_image = parse_rgb(data_read)
        save_image(raw_image, './train_data/rgb/Shanghai_img{}.png'.format(id), cv2.INTER_LINEAR)
    else:
        dataset = rio.open('./AOI_4_Shanghai_Test/RGB-PanSharpen/RGB-PanSharpen_AOI_4_Shanghai_img{}.tif'.format(id))
        data_read = dataset.read()
        raw_image = parse_rgb(data_read)
        save_image(raw_image, './test_data/rgb/Shanghai_img{}.png'.format(id), cv2.INTER_LINEAR)
        return
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
    save_image(masked_image, './train_data/clipped/Shanghai_img{}.png'.format(id), cv2.INTER_LINEAR)
    save_image(255 - np.float32(out_mask) * 255, './train_data/mask/Shanghai_img{}.png'.format(id), cv2.INTER_NEAREST)


def normalize_channel(tif, idx):
    channel = tif[:, :, idx]
    channel = channel / np.max(channel) * 255
    return channel


def save_image(data, path, interp):
    data = cv2.resize(data, dsize=(512, 512), interpolation=interp)
    cv2.imwrite(path, data)


def parse_rgb(image):
    image = image.transpose(1, 2, 0)
    channels = tuple(map(lambda idx: normalize_channel(image, idx), [2, 1, 0]))
    data = np.dstack(channels)
    return data


def file_id(file_name):
    return int(re.search(r'img(\d+)', file_name).group(1))


if __name__ == "__main__":
    train_images = sorted(os.listdir('./AOI_4_Shanghai_Train/RGB-PanSharpen'), key=file_id)
    print(train_images)
    for image in train_images[:100]:
        id = file_id(image)
        print(id)
        mask_image(id)
    test_images = sorted(os.listdir('./AOI_4_Shanghai_Test/RGB-PanSharpen'), key=file_id)
    print(test_images)
    for image in test_images[:100]:
        id = file_id(image)
        print(id)
        mask_image(id, training=False)
