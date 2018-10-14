def _downsample_block(self, inputs, nb_filters):
    net = slim.conv2d(inputs, nb_filters, 3, padding='VALID', stride=1)
    net = slim.conv2d(net, nb_filters, 3, padding='VALID', stride=1)

    return net, slim.max_pool2d(net, 2, stride=2)

def _upsample_block(self, inputs, downsample_reference, nb_filters):
  net = slim.conv2d_transpose(inputs, nb_filters, 2, stride=2)

  downsample_size = downsample_reference[0].get_shape().as_list()[0]
  target_size = net[0].get_shape().as_list()[0]
  size_difference = downsample_size - target_size

  crop_topleft_y = int(np.floor(size_difference / float(2)))
  crop_topleft_x = int(np.floor(size_difference / float(2)))

  net = tf.concat([net, tf.image.crop_to_bounding_box(
    downsample_reference, crop_topleft_y, crop_topleft_x, target_size,
    target_size)], axis=-1)

  net = slim.conv2d(net, nb_filters, 3, padding='VALID', stride=1)
  net = slim.conv2d(net, nb_filters, 3, padding='VALID', stride=1)

  return net

def build_network(self, image_batch, channel_means):
  image_batch = tf.stack(image_batch)
  if (not (image_batch[0].get_shape() == self.input_image_dims)):
    print("Real size of {} is not requested size of {}".format(
      image_batch[0].get_shape(), self.input_image_dims))
    assert(image_batch[0].get_shape() == self.input_image_dims)

  image_batch = self.preprocess(image_batch, channel_means)

  with tf.variable_scope("UNet", values=[image_batch]):
    with slim.arg_scope(self._arg_scope()):
      print(image_batch)
      ds1, pool1 = self._downsample_block(image_batch, 64)
      print(pool1)
      ds2, pool2 = self._downsample_block(pool1, 128)
      print(pool2)
      ds3, pool3 = self._downsample_block(pool2, 256)
      print(pool3)
      ds4, pool4 = self._downsample_block(pool3, 512)
      print(pool4)
      ds5, _ = self._downsample_block(pool4, 1024)
      print(ds5)
      us1 = self._upsample_block(ds5, ds4, 512)
      print(us1)
      us2 = self._upsample_block(us1, ds3, 256)
      print(us2)
      us3 = self._upsample_block(us2, ds2, 128)
      print(us3)
      us4 = self._upsample_block(us3, ds1, 64)
      print(us4)

      final = slim.conv2d(us4, 2, 1, padding='VALID', stride=1,
                          activation_fn=None)

      print(final)

      return final