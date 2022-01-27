import tensorflow as tf
import time
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import CG_loss

# Set default values
AUTOTUNE = tf.data.AUTOTUNE
OUTPUT_CHANNELS = 3
EPOCHS = 20
BATCH_SIZE = 20

# Path to save progress images
save_dir_train = 'Output_DLV_CG_train'
save_dir_test = 'Output_DLV_CG_test'

# Path to checkpoints of training
checkpoint_path = "CycleGAN/checkpoints/train"

# Initialize generators and discriminators
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

# Set optimizer for generator and discriminator
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# TRAINING
ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')


def generate_images_train(model, test_input, cartoon_input, epoch):
  prediction = model(test_input)

  # Saves one generated image
  img = image.array_to_img(prediction.numpy() * 255., scale=False)
  img.save(os.path.join(save_dir, 'generated_face' + str(epoch) + '.png'))
  # Saves one real image for comparison
  img = image.array_to_img(test_input.numpy() * 255., scale=False) 
  img.save(os.path.join(save_dir, 'real_face' + str(epoch) + '.png'))
  # Saves one cartoon image
  img = image.array_to_img(cartoon_input.numpy() * 255., scale=False) 
  img.save(os.path.join(save_dir, 'real_face' + str(epoch) + '.png'))
  '''  
  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()
  '''

def generate_images_test(model, test_input):
  prediction = model(test_input)

  # Saves one generated image
  img = image.array_to_img(prediction * 255., scale=False)
  img.save(os.path.join(save_dir, 'generated_face' + str(epoch) + '.png'))
  # Saves one real image for comparison
  img = image.array_to_img(test_input * 255., scale=False) 
  img.save(os.path.join(save_dir, 'real_face' + str(epoch) + '.png'))



def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
    
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = CG_loss.generator_loss(disc_fake_y)
    gen_f_loss = CG_loss.generator_loss(disc_fake_x)
    
    total_cycle_loss = CG_loss.calc_cycle_loss(real_x, cycled_x) + CG_loss.calc_cycle_loss(real_y, cycled_y)
    
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + CG_loss.identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + CG_loss.identity_loss(real_x, same_x)

    disc_x_loss = CG_loss.discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = CG_loss.discriminator_loss(disc_real_y, disc_fake_y)
  
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)
  
  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))
  
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))
  
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

def start_train(train_base_tf, train_cartoon_tf, sample_base, sample_cartoon):

  #start = 0
  for epoch in range(EPOCHS):
    start = time.time()
    # Set max size
    #stop = start + BATCH_SIZE
    # Samples of base images
    #train_img_base = train_base[int(start): int(stop)]
    # Samples of cartoon images
    #train_img_cartoon = train_cartoon[int(start): int(stop)]
    # Transform numpy arrays to tensors
    #train_base_tf = tf.data.Dataset.from_tensor_slices((train_img_base))
    #train_cartoon_tf = tf.data.Dataset.from_tensor_slices((train_img_cartoon))

    n = 0
    for image_x, image_y in tf.data.Dataset.zip((train_base_tf, train_cartoon_tf)):
      train_step(image_x, image_y)
      if n % 10 == 0:
        print ('.', end='')
      n += 1

    clear_output(wait=True)
    # Using a consistent image (sample_base) so that the progress of the model
    # is clearly visible.
    generate_images_train(generator_g, sample_base, sample_cartoon, epoch)
    print("first run")

    # start += BATCH_SIZE
    #if start > len(train_img_base) - batch_size:
     #   start = 0

    if (epoch + 1) % 5 == 0:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                          ckpt_save_path))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))