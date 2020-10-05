from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


def cast_f(x):
    return K.cast(x, K.floatx())


def cast_b(x):
    return K.cast(x, bool)


def iou_loss_core(true, pred):  # this can be used as a loss if you make it negative
    intersection = true * pred
    not_true = 1 - true
    union = true + (not_true * pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())


def iou_metric(true, pred):  # any shape can go - can't be a loss function

    thresholds = [0.5 + (i * .05) for i in range(10)]

    # flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = cast_f(K.greater(pred, 0.5))

    # total white pixels - (batch,)
    true_sum = K.sum(true, axis=-1)
    pred_sum = K.sum(pred, axis=-1)

    # has mask or not per image - (batch,)
    true1 = cast_f(K.greater(true_sum, 1))
    pred1 = cast_f(K.greater(pred_sum, 1))

    # to get images that have mask in both true and pred
    true_positive_mask = cast_b(true1 * pred1)

    # separating only the possible true positives to check iou
    test_true = tf.boolean_mask(true, true_positive_mask)
    test_pred = tf.boolean_mask(pred, true_positive_mask)

    # getting iou and threshold comparisons
    iou = iou_loss_core(test_true, test_pred)
    true_positives = [cast_f(K.greater(iou, tres)) for tres in thresholds]

    # mean of thresholds for true positives and total sum
    true_positives = K.mean(K.stack(true_positives, axis=-1), axis=-1)
    true_positives = K.sum(true_positives)

    # to get images that don't have mask in both true and pred
    true_negatives = (1 - true1) * (1 - pred1)  # = 1 -true1 - pred1 + true1*pred1
    true_negatives = K.sum(true_negatives)

    return (true_positives + true_negatives) / cast_f(K.shape(true)[0])


class Unet:
    def __init__(self, pretrained=False, input_size=(224, 224, 3)):
        self.base_model = None
        self.pretrained = pretrained
        self.input_size = input_size
        self.model = self.build()

    def build(self):
        inputs = Input(self.input_size)
        if self.pretrained:
            base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            base_model.trainable = False
            self.base_model = base_model
            layer_names = [
                'block_1_expand_relu',  # 64x64
                'block_3_expand_relu',  # 32x32
                'block_6_expand_relu',  # 16x16
                'block_13_expand_relu',  # 8x8
                'block_16_project',  # 4x4
            ]
            layers = [base_model.get_layer(name).output for name in layer_names]

            down_stack = Model(inputs=base_model.input, outputs=layers)

            down_stack.trainable = False

            up_stack = [
                Conv2DTranspose(512, (3, 3), 2, padding='same'),
                Conv2DTranspose(256, (3, 3), 2, padding='same'),
                Conv2DTranspose(128, (3, 3), 2, padding='same'),
                Conv2DTranspose(64, (3, 3), 2, padding='same')
            ]

            x = inputs

            # 모델을 통해 다운샘플링합시다
            skips = down_stack(x)
            x = skips[-1]
            skips = reversed(skips[:-1])

            # 건너뛰기 연결을 업샘플링하고 설정하세요
            for up, skip in zip(up_stack, skips):
                x = up(x)
                concat = Concatenate()
                x = concat([x, skip])

            # 이 모델의 마지막 층입니다
            last = Conv2DTranspose(1, 3, strides=2, padding='same')  # 64x64 -> 128x128

            x = last(x)

            model = Model(inputs=inputs, outputs=x)
        else:
            conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            drop4 = Dropout(0.5)(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

            conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
            conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
            drop5 = Dropout(0.5)(conv5)

            up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
            merge6 = concatenate([drop4, up6], axis=3)
            conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
            conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

            up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
            merge7 = concatenate([conv3, up7], axis=3)
            conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
            conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

            up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
            merge8 = concatenate([conv2, up8], axis=3)
            conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
            conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

            up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
            merge9 = concatenate([conv1, up9], axis=3)
            conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
            conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = Conv2D(9, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

            model = Model(inputs, conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=[iou_metric])

        model.summary()

        return model

    def fit(self, data, steps_per_epoch, epochs, callbacks):
        self.model.fit_generator(data, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)

    def load_weights(self, path):
        self.model.load_weights(path)

    def predict(self, testGene, batch_size, verbose):
        return self.model.predict(testGene, batch_size, verbose=verbose)


def adjust_data(img, mask, flag_multi_class, num_class):
    if flag_multi_class:
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            # for one pixel in the image, find the class in mask and convert it into one-hot vector
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2], new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (
            new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    elif np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask


def train_data_generator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                         mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                         flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(224, 224), seed=1):
    """
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    """
    count = 0
    train_size = 2000
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        count += 1
        if count > train_size:
            return
        img, mask = adjust_data(img, mask, flag_multi_class, num_class)
        yield img, mask


if __name__ == "__main__":
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    myGene = train_data_generator(2, './train_data', 'rgb', 'mask', data_gen_args, image_color_mode="rgb", mask_color_mode="grayscale", save_to_dir=None)
    model = Unet(pretrained=True)
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit(myGene, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])
