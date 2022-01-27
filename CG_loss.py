import tensorflow as tf

# Set default values
LAMBDA = 10

# Computes the cross-entropy loss between true labels and predicted labels.
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# function to calculate discriminator loss (fool the discriminator)
def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real) # tf.ones_like creates a tensor of all ones with same shape as the input

  generated_loss = loss_obj(tf.zeros_like(generated), generated) # tf.ones_like creates a tensor of all zeros with same shape as the input

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

# function to calculate generator loss
def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated) # tf.ones_like creates a tensor of all ones with same shape as the input

# function to calculate the cycle consistency loss to make sure the translation results are close to the original images
def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss