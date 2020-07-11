import tensorflow as tf
import numpy as np

    # initialization of convolutional layers
def convolution2d(layer_name,layer_entry,out_channels,stride,kernel_size=[3,3]) :
        with tf.variable_scope(layer_name):
            in_channels = layer_entry.shape[-1]
            w=tf.get_variable(name='weights_conv',trainable=True,
                             shape=[kernel_size[0],kernel_size[1],in_channels,out_channels],
                             initializer=tf.initializers.glorot_uniform())
            
            conv_layer=tf.nn.conv2d(layer_entry,w,stride,padding='SAME',name='layer_conv')
            activation=tf.nn.relu(conv_layer,name='relu')
            return activation
    #   
# initialization of max-poiling        
def max_pooling2d( layer_name, layer_entry, kernel, stride):
        with tf.name_scope(layer_name):
            
            max_pool = tf.nn.max_pool(layer_entry, kernel, stride, padding='SAME', name=layer_name)
            return max_pool
          
   
# initialization of a fully connected layer
def fc_dense(layer_name, input_tensor, out_nodes):
        shape = input_tensor.get_shape()
        if len(shape) == 4:  
            size = shape[1].value * shape[2].value * shape[3].value
        else:  
            size = shape[-1].value
        with tf.variable_scope(layer_name):
            w = tf.get_variable('weights_dense',
                                shape=[size, out_nodes],
                                initializer=tf.initializers.glorot_uniform())
          
            flat_x = tf.reshape(input_tensor, [-1, size])
            bottom = tf.matmul(flat_x, w)
            bottom = tf.nn.relu(bottom)
            return bottom
    
# initialization of the output layer            
def out_dense(layer_name, input_tensor, out_nodes) :
        shape = input_tensor.shape[-1]
        print(shape)
        w = tf.get_variable('weights_dense_out',
                                shape=[ shape , out_nodes],
                                initializer=tf.initializers.glorot_uniform())
        out_dense= tf.matmul( input_tensor, w)
        return  out_dense
# initialization of batch normalization    
def batch_normalization(layer_name, layer):
        with tf.name_scope(layer_name):
            epsilon = 1e-3
            layer_out= tf.layers.batch_normalization(layer, epsilon=epsilon)
            return layer_out

