#!/usr/bin/env python3
import time
import imageio
import tensorflow 
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image  
from tensorflow.keras.models import *
import numpy as np
from PIL import Image

def anet_bs(num_filters=64, num_res_blocks=16, res_block_scaling=None):
    x_in = Input(shape=(256,256,1))
    x = b = Conv2D(num_filters, (3, 3), padding='same')(x_in)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, (3, 3), padding='same')(b)
    x = Add()([x, b])
    x = Conv2D(1, (3, 3), padding='same')(x)
    return Model(x_in, x, name="Huy-BS")

def res_block(x_in, filters, scaling):
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x_in)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x
anet_bs = anet_bs(num_filters=64, num_res_blocks=16, res_block_scaling=0.1)

anet_bs.load_weights("vutxuong.h5") 

print("Load mod done")

img = Image.open('coxuong.png')
img = img.resize((256,256)) 

x = image.img_to_array(img)
x = x.astype('float32') / 255

x1 = np.expand_dims(x, axis=0)

pred = anet_bs.predict(x1)


test_img = np.reshape(pred, (256,256,1)) 
imageio.imwrite('koxuong.png', test_img)
print("Done")
