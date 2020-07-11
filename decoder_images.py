
import tensorflow as tf
import numpy as np
def decoder_images(img,img_height,img_width):
  
    img=tf.io.read_file(img)
    img=tf.image.decode_jpeg(img)
    img=tf.image.resize(img,[img_height,img_width])
    img=tf.cast(img,tf.float32)/255.0
    img=np.expand_dims(img,axis=0)
    return img

