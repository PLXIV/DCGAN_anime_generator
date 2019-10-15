import os
import numpy as np
import time
from PIL import Image
import matplotlib.gridspec as gridspec
from keras import Model
from keras.optimizers import Adam
from nn_models import NN_architectures
from keras.models import model_from_json
import matplotlib.pyplot as plt
from collections import deque
from keras.models import model_from_json
import pickle

np.random.seed(1337)

def image_normalization(image):
    image = image.astype('float64')
    for i in range(3):
        image[:, :, i] = image[:, :, i] / 255.0
    return image

def image_denormalization(image):
    for i in range(3):
        image[:, :, i] = image[:, :, i] * 255.0
    return image.astype(np.uint8)

def load_images(dataset, batch_size):

    training_sample_filenames = np.asarray(os.listdir(dataset))
    names = training_sample_filenames[np.random.randint(0, len(training_sample_filenames), size=batch_size)]
    ima = []
    for name in names:
        im = np.array(Image.open(dataset + name))
        im = image_normalization(im)
        ima.append(im)
    return np.asarray(ima)

def gen_noise(batch_size, noise_shape):
    return np.random.normal(0, 1, size=(batch_size,) + noise_shape)

def visualize_generation(img_batch, img_save_dir):
    plt.figure(figsize=(8, 8))
    gs1 = gridspec.GridSpec(8, 8)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(img_batch.shape[0], 64, replace=False)
    for i in range(64):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = img_batch[rand_index, :, :, :]
        fig = plt.imshow(image_denormalization(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(img_save_dir, bbox_inches='tight', pad_inches=0)
    plt.show()

def store_model(model, filename, step):
    model_json = model.to_json()
    with open(filename + '_model.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(filename + "_latest_weights_and_arch.hdf5")
    model.save_weights(filename + '_' + str(step) +  "_latest_weights_and_arch.hdf5")

def load_architecture(filename):
    with open(filename, 'r') as model:
        model_json = model.read()
    return model_json

if __name__ == '__main__':
    save_model = True
    load_models = True
    pickle_load = False

    noise_shape = (1, 1, 100)
    image_shape = (64, 64, 3)

    num_steps = 20000
    batch_size = 256

    save_model_step = 100
    visualize_every = 20

    #folder_save = "/home/nct01/nct01009/new_stuff/anime/DCGAN_anime_generator/tmp/"
    #log_dir = '/home/nct01/nct01009/new_stuff/anime/DCGAN_anime_generator/logs/'
    #model_dir = '/home/nct01/nct01009/new_stuff/anime/DCGAN_anime_generator/model/'

    folder_save = '../tmp/'
    log_dir = '../logs/'
    model_dir = '../model/'
    dataset = '../../dataset_normalized_2/'


    avg_disc_fake_loss = deque([0], maxlen=250)
    avg_disc_real_loss = deque([0], maxlen=250)
    avg_GAN_loss = deque([0], maxlen=250)

    if load_models:
        if pickle_load:
            with open(model_dir + 'network_state.pickle', 'w') as pi:
                networks = pickle.load(pi)
        else:
            networks = NN_architectures(noise_shape, image_shape, initialize=False)

            gen_opt = Adam(lr=0.00015, beta_1=0.5)
            generator = load_architecture(model_dir + "generator_model.json")
            networks.generator = model_from_json(generator)
            networks.generator.load_weights(model_dir + "generator_latest_weights_and_arch.hdf5")
            networks.generator.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])
            networks.generator.summary()

            dis_opt = Adam(lr=0.0002, beta_1=0.5)
            discriminator = load_architecture(model_dir + "discriminator_model.json")
            networks.discriminator = model_from_json(discriminator)
            networks.discriminator.load_weights(model_dir + "discriminator_latest_weights_and_arch.hdf5")
            networks.discriminator.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])
            networks.discriminator.summary()

    else:
        networks = NN_architectures(noise_shape, image_shape, initialize=True)
    networks.gan = networks.gan_generator()

    for step in range(num_steps):
        initial_time = time.time()

        img_X = load_images(dataset, batch_size)
        gen_X = networks.generator.predict(gen_noise(batch_size, noise_shape))

        if (step % visualize_every) == 0:
            step_num = str(step).zfill(4)
            visualize_generation(gen_X, folder_save + step_num + ".jpg")

        X = np.concatenate([img_X, gen_X])
        img_y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
        gen_y = np.random.random_sample(batch_size) * 0.2
        y = np.concatenate((img_y, gen_y))

        networks.discriminator.trainable = True
        networks.generator.trainable = False

        dis_metrics_real = networks.discriminator.train_on_batch(img_X, img_y)
        dis_metrics_fake = networks.discriminator.train_on_batch(gen_X, gen_y)
        avg_disc_fake_loss.append(dis_metrics_fake[0])
        avg_disc_real_loss.append(dis_metrics_real[0])
        print("Disc: real loss: %f fake loss: %f" % (dis_metrics_real[0], dis_metrics_fake[0]))

        networks.discriminator.trainable = False
        networks.generator.trainable = True

        GAN_X = gen_noise(batch_size, noise_shape)
        GAN_Y = img_y
        gan_metrics = networks.gan.train_on_batch(GAN_X, GAN_Y)
        avg_GAN_loss.append(gan_metrics[0])
        print("GAN loss: %f" % (gan_metrics[0]))

        text_file = open(log_dir + "training_log.txt", "a")
        text_file.write("%d Image loss: %f Fake loss: %f Generated loss: %f\n" % (step, dis_metrics_real[0],
                                                                                  dis_metrics_fake[0], gan_metrics[0]))
        text_file.close()

        end_time = time.time()
        diff_time = int(end_time - initial_time)
        print("Step %d completed. Time took: %s secs." % (step, diff_time))

        if (step % save_model_step) == 0 and save_model:
            networks.discriminator.trainable = True
            networks.generator.trainable = True

            store_model(networks.generator, model_dir + 'generator', step)
            store_model(networks.discriminator, model_dir + 'discriminator', step)
            store_model(networks.gan, model_dir + 'gan', step)

