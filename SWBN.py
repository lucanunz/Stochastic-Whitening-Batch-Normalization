#author: Luca Nunziante
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np

def standardize_data(X, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance elementwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Normalize
    Y = (X - moving_mean)*inv
    return Y

class SWBN(layers.Layer):
    def __init__(self, criterion="fro", alpha=1e-5):
        super(SWBN, self).__init__()
        self.criterion=criterion
        self.alpha=alpha

    def build(self, input_shape):
        weight_shape = [input_shape[-1], 1]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        self.whitening_matrix = self.add_weight(name='whitening_matrix', 
                              shape=(weight_shape[0],weight_shape[0]),initializer=tf.initializers.identity,
                              trainable=False)
        #This variable will contain the covariance matrix of the whitened data and will be plotted as a heatmap
        self.cov = tf.Variable(tf.fill(dims=(weight_shape[0],weight_shape[0]),value=0.0),trainable=False)

    # Momentum is used to govern the aggregation over past mean and variance estimates
    def assign_moving_average(self, variable, value):
        momentum = 0.95
        avg = (1.0 - momentum) * value + momentum * variable
        return variable.assign(avg)
    
	#This method implements the update of the whitening matrix w according to Eq. (3)
    def assign_white_matrix(self, w, cov_matrix, alpha=1e-5):
      upper= (w @ cov_matrix @ tf.transpose(w) -tf.eye(cov_matrix.shape[0])) @ w

      if(self.criterion == "fro"):
        inv=tf.norm(tf.eye(cov_matrix.shape[0]) - w @ cov_matrix @ tf.transpose(w), ord=self.criterion, axis=[0,1])
        delta = upper @ cov_matrix / inv
      elif(self.criterion == "kl"):
        delta=upper
      else:
        raise ValueError('Criterion should be either flo or kl, instead found ' + self.criterion)
	
      value = w - alpha*delta
      value = 0.5 * (value+tf.transpose(value))
      return w.assign(value)

    @tf.function
    def call(self, inputs, training):
      shape = K.int_shape(inputs)
      if len(shape) == 4:
          w, h, c = shape[1:]
      elif len(shape) == 2:
          w, h, c = 1, 1, shape[-1]
          inputs = tf.expand_dims(inputs, 1)
          inputs = tf.expand_dims(inputs, 1)
      else:
          raise ValueError('shape not support:{}'.format(shape))
      #bs is the batch size
      #w,h,c are width, height, number of channels respectively
      #the input is in the standard keras form (bs,w,h,c)
      bs = tf.shape(inputs)[0]
      
      #we transpose the input so that it is now (c,bs,w,h)
      x_t = tf.transpose(inputs, (3, 0, 1, 2))

      # and flatten it to become (c,bs*w*h)
      x_flat = tf.reshape(x_t, (c, -1))
      
      if training:
         #This first block of code computes the batch mean and variance, and updates them
          batch_mean = tf.reduce_mean(x_flat, axis=1, keepdims=True)
          batch_variance = tf.reduce_mean(tf.math.squared_difference(
              x_flat, tf.stop_gradient(batch_mean)), axis=1, keepdims=True)
          
          mean_update = self.assign_moving_average(
              self.moving_mean, batch_mean)
          variance_update = self.assign_moving_average(
              self.moving_variance, batch_variance)
          mean, variance = batch_mean, batch_variance
          self.add_update(mean_update)
          self.add_update(variance_update)

		      #Here data is normalized, i.e. we subtract the mean and divide by the sqrt of the variance + eps
          X_s = standardize_data(x_flat, moving_mean=mean, moving_var=variance, eps=1e-8)

          #This is the sample covariance matrix, used in the method "assign_white_matrix" to compute the update
          #of Eq. (3)
          cov_matrix = tf.matmul(X_s, X_s, transpose_b=True) / (tf.cast(bs*w*h, tf.float32))
		
          matrix_update = self.assign_white_matrix(
              self.whitening_matrix, cov_matrix)

          self.add_update(matrix_update)
      else:
          mean, variance = self.moving_mean, self.moving_variance
          X_s = standardize_data(x_flat, moving_mean=mean, moving_var=variance, eps=1e-8)

	    #Here, we  whiten the data and compute the covariance matrix of the whitened data, "heatmap_matrix"
      #that will be put in the ad hoc variable defined
      X_w = self.whitening_matrix @ X_s
    
      heatmap_matrix = tf.matmul(X_w, X_w, transpose_b=True) / (tf.cast(bs*w*h, tf.float32))
      self.cov.assign(heatmap_matrix)

      #We apply scale and shift to the whitened data
      output = X_w * self.gamma + self.beta

      #Finally, we reshape the data to the original shape
      output = tf.reshape(output, (c, bs, w, h))
      output = tf.transpose(output, [1, 2, 3, 0])
      if w == 1:
        output = tf.squeeze(output, 1)
      if h == 1:
        output = tf.squeeze(output, 1)
      return output
