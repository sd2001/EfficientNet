from utils import *
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Reshape, multiply, add, Activation

def EfficientNet_B0(channels,
                    expansion_coefs,
                    repeats,
                    strides,
                    kernel_sizes,
                    d_coef,
                    w_coef,
                    r_coef,
                    dropout_rate,
                    include_top,
                    se_ratio = 0.25,
                    classes=1000):
   
    inputs = Input(shape=(224, 224, 3))
    
    stage1 = ConvBlock(inputs,
                       filters=32,
                       kernel_size=3,
                       stride=2)
    
    stage2 = MBConvBlock(stage1, 
                         scaled_channels(channels[0], w_coef),
                         scaled_channels(channels[1], w_coef),
                         kernel_sizes[0],
                         expansion_coefs[0],
                         se_ratio,
                         strides[0],
                         scaled_repeats(repeats[0], d_coef),
                         dropout_rate=dropout_rate)
    
    stage3 = MBConvBlock(stage2, 
                         scaled_channels(channels[1], w_coef),
                         scaled_channels(channels[2], w_coef),
                         kernel_sizes[1],
                         expansion_coefs[1],
                         se_ratio,
                         strides[1],
                         scaled_repeats(repeats[1], d_coef),
                         dropout_rate=dropout_rate)
    
    stage4 = MBConvBlock(stage3, 
                         scaled_channels(channels[2], w_coef),
                         scaled_channels(channels[3], w_coef),
                         kernel_sizes[2],
                         expansion_coefs[2],
                         se_ratio,
                         strides[2],
                         scaled_repeats(repeats[2], d_coef),
                         dropout_rate=dropout_rate)
    
    stage5 = MBConvBlock(stage4, 
                         scaled_channels(channels[3], w_coef),
                         scaled_channels(channels[4], w_coef),
                         kernel_sizes[3],
                         expansion_coefs[3],
                         se_ratio,
                         strides[3],
                         scaled_repeats(repeats[3], d_coef),
                         dropout_rate=dropout_rate)

    stage6 = MBConvBlock(stage5, 
                         scaled_channels(channels[4], w_coef),
                         scaled_channels(channels[5], w_coef),
                         kernel_sizes[4],
                         expansion_coefs[4],
                         se_ratio,
                         strides[4],
                         scaled_repeats(repeats[4], d_coef),
                         dropout_rate=dropout_rate)
    
    stage7 = MBConvBlock(stage6, 
                         scaled_channels(channels[5], w_coef),
                         scaled_channels(channels[6], w_coef),
                         kernel_sizes[5],
                         expansion_coefs[5],
                         se_ratio,
                         strides[5],
                         scaled_repeats(repeats[5], d_coef),
                         dropout_rate=dropout_rate)
    
    stage8 = MBConvBlock(stage7, 
                         scaled_channels(channels[6], w_coef),
                         scaled_channels(channels[7], w_coef),
                         kernel_sizes[6],
                         expansion_coefs[6],
                         se_ratio,
                         strides[6],
                         scaled_repeats(repeats[6], d_coef),
                         dropout_rate=dropout_rate)
       
    stage9 = ConvBlock(stage8,
                       filters=scaled_channels(channels[8], w_coef),
                       kernel_size=1,
                       padding='same')
    
    if include_top:
        stage9 = GlobalAveragePooling2D()(stage9)

        if dropout_rate > 0:
            stage9 = Dropout(dropout_rate)(stage9)

        stage9 = Dense(classes, 
                       activation='softmax',
                       kernel_initializer=DENSE_KERNEL_INITIALIZER)(stage9)

    model = Model(inputs, stage9)

    return model

include_top = True
channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
expansion_coefs = [1, 6, 6, 6, 6, 6, 6]
repeats = [1, 2, 2, 3, 3, 4, 1]
strides = [1, 2, 2, 2, 1, 2, 1]
kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
d_coef, w_coef, r_coef, dropout_rate = efficientnet_params('efficientnet-b0')   

conv_base = EfficientNet_B0(channels,
                            expansion_coefs,
                            repeats,
                            strides,
                            kernel_sizes,
                            d_coef,
                            w_coef,
                            r_coef,
                            dropout_rate,
                            include_top=include_top)
conv_base.summary()
if include_top:
    conv_base.load_weights('data/pretrained_model_imagenet_top.h5')
else:
    conv_base.load_weights('data/pretrained_model_imagenet_notop.h5')