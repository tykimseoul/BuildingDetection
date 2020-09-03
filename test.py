import os
import cv2
from train import *

Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]
COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                       Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def test_generator(test_path, target_size=(256, 256), flag_multi_class=True):
    for i in sorted(os.listdir(test_path))[:10]:
        print(i)
        img = cv2.imread(test_path + i)
        img = img / 255
        img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_NEAREST)
        img = np.reshape(img, (1,) + img.shape)
        yield img


def label_visualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def save_result(npyfile, flag_multi_class=False, num_class=2):
    print(npyfile.shape)
    for i, item in enumerate(npyfile):
        print(item.shape)
        img = item[:, :, 0] * 255 // 128 * 255
        cv2.imwrite('./results/Shanghai_img{}.png'.format(i), img)


testGene = test_generator('./test_data/rgb/')
model = Unet(pretrained=True)
model.load_weights("unet_membrane (5).hdf5")
results = model.predict(testGene, 30, verbose=1)
save_result(results)
