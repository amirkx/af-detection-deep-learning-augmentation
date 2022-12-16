import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


def saveImage(img, folderName: str, fileName: str):
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='gray')
    plt.savefig(folderName + fileName + '.eps', format='eps')


def imageCropper(samples, MAXIMUM_REMOVED_CELLS: int): # Returns samples with required size
    cropCount = float('inf')
    PADDING_VAL = samples[0][0][0]
    """
    We know that the size of all images in our dataset is 360x360
    """
    for sample in samples:
        sampleMaxCrop = float('inf')
        for row in range(360):
            rowMaxCrop = 0
            for col in range(180):
                if sample[row][col] == sample[row][359 - col] == PADDING_VAL:
                    rowMaxCrop += 1
                else:
                    break
            sampleMaxCrop = min(rowMaxCrop, sampleMaxCrop)
        cropCount = min(sampleMaxCrop, cropCount)
        for col in range(360):
            colMaxCrop = 0
            for row in range(180):
                if sample[row][col] == sample[359 - row][col] == PADDING_VAL:
                    colMaxCrop += 1
                else:
                    break
            sampleMaxCrop = min(colMaxCrop, sampleMaxCrop)
        cropCount = min(sampleMaxCrop, cropCount)

    print('initial size: ', samples.shape)
    cropCount = min(cropCount, MAXIMUM_REMOVED_CELLS)
    if cropCount > 0:
        samples = samples[:, cropCount:-cropCount, cropCount:-cropCount]
        print('after cropping: ', samples.shape)

    AFTER_CROP_SIZE = samples.shape[1]
    REQUIRED_SIZE = 360 - 2 * MAXIMUM_REMOVED_CELLS
    if AFTER_CROP_SIZE > REQUIRED_SIZE:
        shrinked_arr = np.empty(
            shape=(samples.shape[0], REQUIRED_SIZE, REQUIRED_SIZE))
        for i in range(samples.shape[0]):
            shrinked_arr[i] = resize(
                samples[i], (REQUIRED_SIZE, REQUIRED_SIZE))
        print('after interpolation: ', shrinked_arr.shape)
        samples = shrinked_arr
    return samples, AFTER_CROP_SIZE, PADDING_VAL, cropCount


def augmentation(SAMPLES, N_FAKE_SAMPLES):
    from keras.models import Sequential  # for assembling a Neural Network model
    # adding layers to the Neural Network model
    from keras.layers import Dense, Reshape, Flatten, Conv2D, LeakyReLU, Dropout
#    from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, ReLU, LeakyReLU, Dropout
    from keras.utils import plot_model  # for plotting model diagram
    from keras.optimizers import Adam  # for model optimization
    # Data manipulation
    # for scaling inputs used in the generator and discriminator
    from sklearn.preprocessing import MinMaxScaler
    # For helping the garbage collector
    import gc


    MAXIMUM_REMOVED_CELLS = 86 # From each side (top, right, bottom, left) by both cropping and interpolation
    SAMPLES, AFTER_CROP_SIZE, PADDING_VAL, cropCount = imageCropper(SAMPLES, MAXIMUM_REMOVED_CELLS)
    GAN_INPUT_SHAPE = SAMPLES.shape[1]
    SAMPLES = np.expand_dims(SAMPLES, axis=3)
    print('Input Shape: ', SAMPLES.shape)

    # In[164]:

    # Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    N_SAMPLES = SAMPLES.shape[0]
    IMAGE_SIZE = SAMPLES.shape[1]

    # Select images that we want to use from model trainng
    # Reshape array
    SAMPLES = SAMPLES.reshape(-1, 1)
    print("Reshaped SAMPLES: ", SAMPLES.shape)

    # Fit the scaler
    scaler.fit(SAMPLES)

    # Scale the array
    SAMPLES = scaler.transform(SAMPLES)

    # Reshape back to the original shape
    SAMPLES = SAMPLES.reshape(N_SAMPLES, IMAGE_SIZE, IMAGE_SIZE, 1)
    print("Shape of the scaled array: ", SAMPLES.shape)
    gc.collect()

    # ### Step 3 - Setup GAN

    # #### Define Generator model

    # In[23]:

    def generator(latent_dim):
        model = Sequential(name="Generator")  # Model

        # Hidden Layer 1: Start with 8 x 8 image
        # number of nodes in the first hidden layer
        n_nodes = GAN_INPUT_SHAPE * GAN_INPUT_SHAPE * N_SAMPLES
        model.add(Dense(n_nodes, input_dim=latent_dim,
                  name='Generator-Hidden-Layer-1'))
        model.add(Reshape((GAN_INPUT_SHAPE, GAN_INPUT_SHAPE, N_SAMPLES),
                  name='Generator-Hidden-Layer-Reshape-1'))

        # Output Layer (Note, we use 1 filter because we have 1 channel for a grayscale image)
        model.add(Conv2D(filters=1, kernel_size=(5, 5), activation='tanh',
                  padding='same', name='Generator-Output-Layer'))
        return model

    # Instantiate
    latent_dim = 100  # latent space
    gen_model = generator(latent_dim)
    # Show model summary and plot model diagram
    gen_model.summary()
    # , to_file='generator_structure.png')
    plot_model(gen_model, show_shapes=True, show_layer_names=True, dpi=400)

    # #### Define a Discriminator model

    # In[24]:

    def discriminator(in_shape=(GAN_INPUT_SHAPE, GAN_INPUT_SHAPE, 1)):
        model = Sequential(name="Discriminator")  # Model

        # Hidden Layer 1
        model.add(Conv2D(filters=GAN_INPUT_SHAPE, kernel_size=(4, 4), strides=(
            2, 2), padding='same', input_shape=in_shape, name='Discriminator-Hidden-Layer-1'))
        model.add(
            LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-1'))

        # Hidden Layer 2
        model.add(Conv2D(filters=GAN_INPUT_SHAPE * 2, kernel_size=(4, 4), strides=(
            2, 2), padding='same', input_shape=in_shape, name='Discriminator-Hidden-Layer-2'))
        model.add(
            LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-2'))

        # Hidden Layer 3
        model.add(Conv2D(filters=GAN_INPUT_SHAPE * 2, kernel_size=(4, 4), strides=(
            2, 2), padding='same', input_shape=in_shape, name='Discriminator-Hidden-Layer-3'))
        model.add(
            LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-3'))

        # Flatten the shape
        model.add(Flatten(name='Discriminator-Flatten-Layer'))
        # Randomly drop some connections for better generalization
        model.add(Dropout(0.3, name='Discriminator-Flatten-Layer-Dropout'))
        model.add(Dense(1, activation='sigmoid',
                  name='Discriminator-Output-Layer'))  # Output Layer

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=Adam(
            learning_rate=0.00005, beta_1=0.5), metrics=['accuracy'])
        return model

    # Instantiate
    dis_model = discriminator()

    # Show model summary and plot model diagram
    dis_model.summary()
    # , to_file='discriminator_structure.png')
    plot_model(dis_model, show_shapes=True, show_layer_names=True, dpi=400)

    # #### Combine Generator and Discriminator models into trainable GAN

    # In[25]:

    def def_gan(generator, discriminator):

        # We don't want to train the weights of discriminator at this stage. Hence, make it not trainable
        discriminator.trainable = False

        # Combine
        model = Sequential(name="DCGAN")  # GAN Model
        model.add(generator)  # Add Generator
        model.add(discriminator)  # Add Disriminator

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=Adam(
            learning_rate=0.00005, beta_1=0.5))
        return model

    # Instantiate
    gan_model = def_gan(gen_model, dis_model)

    # Show model summary and plot model diagram
    gan_model.summary()
    # , to_file='dcgan_structure.png')
    plot_model(gan_model, show_shapes=True, show_layer_names=True, dpi=400)

    # ### Step 4 - Setup functions to:
    # - sample the latent space
    # - sample real images
    # - generate fake images with the generator model

    # ##### Set up a function to sample real images

    # In[26]:

    def real_samples(n, dataset):

        # Samples of real data
        X = dataset[np.random.choice(dataset.shape[0], n, replace=True), :]

        # Class labels
        y = np.ones((n, 1))
        return X, y

    # ##### Generate points in the latent space, which we will use as inputs for the generator

    # In[27]:

    def latent_vector(latent_dim, n):

        # Generate points in the latent space
        latent_input = np.random.randn(latent_dim * n)

        # Reshape into a batch of inputs for the network
        latent_input = latent_input.reshape(n, latent_dim)
        return latent_input

    # ##### The below function will use the generator to generate n fake examples together with class labels

    # In[28]:

    def fake_samples(generator, latent_dim, n):

        # Generate points in latent space
        latent_output = latent_vector(latent_dim, n)

        # Predict outputs (i.e., generate fake samples)
        X = generator.predict(latent_output)

        # Create class labels
        y = np.zeros((n, 1))
        return X, y

    # ### Step 5 - Setup functions for model performance evaluation and training

    # ##### Show Discriminator model accuracy and plot real vs. fake (generated) comparison

    # In[29]:

    def performance_summary(generator, discriminator, dataset, latent_dim, n=50):

        # Get samples of the real data
        x_real, y_real = real_samples(n, dataset)
        # Evaluate the descriminator on real data
        _, real_accuracy = discriminator.evaluate(x_real, y_real, verbose=0)

        # Get fake (generated) samples
        x_fake, y_fake = fake_samples(generator, latent_dim, n)
        # Evaluate the descriminator on fake (generated) data
        _, fake_accuracy = discriminator.evaluate(x_fake, y_fake, verbose=0)

        # summarize discriminator performance
        print("*** Evaluation ***")
        print("Discriminator Accuracy on REAL images: ", real_accuracy)
        print("Discriminator Accuracy on FAKE (generated) images: ", fake_accuracy)

        # Create a 2D scatter plot to show real and fake (generated) data points
        # Display 6 fake images
        x_fake_inv_trans = x_fake.reshape(-1, 1)
        x_fake_inv_trans = scaler.inverse_transform(x_fake_inv_trans)
        x_fake_inv_trans = x_fake_inv_trans.reshape(
            n, GAN_INPUT_SHAPE, GAN_INPUT_SHAPE, 1)

        _, axs = plt.subplots(
            5, 3, sharey=False, tight_layout=True, figsize=(12, 15), facecolor='white')
        plt.xticks([])
        plt.yticks([])
        k = 0
        for row in range(0, 5):
            for col in range(0, 3):
                axs[row, col].matshow(
                    x_fake_inv_trans[k][:, :, 0], cmap='gray')

                k += 1
        plt.show()

    # ##### Define a function to train our DCGAN model (generator and discriminator)

    # In[30]:

    def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, n_batch=16, n_eval=75):

        # Our batch to train the discriminator will consist of half real images and half fake (generated) images
        half_batch = int(n_batch / 2)

        # We will manually enumare epochs
        for i in range(n_epochs):

            # Discriminator training
            # Prep real samples
            x_real, y_real = real_samples(half_batch, dataset)
            # Prep fake (generated) samples
            x_fake, y_fake = fake_samples(g_model, latent_dim, half_batch)

            # Train the discriminator using real and fake samples
            X, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
            discriminator_loss, _ = d_model.train_on_batch(X, y)

        # Generator training
            # Get values from the latent space to be used as inputs for the generator
            x_gan = latent_vector(latent_dim, n_batch)
            # While we are generating fake samples,
            # we want GAN generator model to create examples that resemble the real ones,
            # hence we want to pass labels corresponding to real samples, i.e. y=1, not 0.
            y_gan = np.ones((n_batch, 1))

            # Train the generator via a composite GAN model
            generator_loss = gan_model.train_on_batch(x_gan, y_gan)

            # Evaluate the model at every n_eval epochs
            if (i) % n_eval == 0:
                print("Epoch number: ", i)
                print("*** Training ***")
                print("Discriminator Loss ", discriminator_loss)
                print("Generator Loss: ", generator_loss)
                performance_summary(g_model, d_model, dataset, latent_dim)

    # ### Step 6 - Train the model and plot the results

    # ##### Use the above train function to train our GAN model

    # In[31]:

    # Train DCGAN model
#    train(gen_model, dis_model, gan_model, SAMPLES, latent_dim, n_epochs=1)

    train(gen_model, dis_model, gan_model, SAMPLES, latent_dim, n_epochs=149)

    train(gen_model, dis_model, gan_model, SAMPLES, latent_dim, n_epochs=149)

    train(gen_model, dis_model, gan_model, SAMPLES, latent_dim, n_epochs=70)

    # Add library
    # from keras.models import load_model
    # Load saved models
    # gen_model = load_model(main_dir+"/data/models/007b_Generator_3k.h5")
    # dis_model = load_model(main_dir+"/data/models/007b_Discriminator_3k.h5")
    # gan_model = load_model(main_dir+"/data/models/007b_GAN_3k.h5")

    # ---

    # ### Step 7 - Evaluate model performace and use the Generator to create a bunch of images

    # In[37]:

    def performance_eval(epoch, generator, discriminator, dataset, latent_dim, n=40):

        # Get samples of the real data
        x_real, y_real = real_samples(n, dataset)
        # Evaluate the descriminator on real data
        _, real_accuracy = discriminator.evaluate(x_real, y_real, verbose=0)

        # Get fake (generated) samples
        x_fake, y_fake = fake_samples(generator, latent_dim, n)
        # Evaluate the descriminator on fake (generated) data
        _, fake_accuracy = discriminator.evaluate(x_fake, y_fake, verbose=0)

        # summarize discriminator performance
        print("Epoch number: ", epoch)
        print("Discriminator Accuracy on REAL images: ", real_accuracy)
        print("Discriminator Accuracy on FAKE (generated) images: ", fake_accuracy)

        # Create a 2D scatter plot to show real and fake (generated) data points
        # Display 20 fake images
        # Display 6 fake images
        x_fake_inv_trans = x_fake.reshape(-1, 1)
        x_fake_inv_trans = scaler.inverse_transform(x_fake_inv_trans)
        x_fake_inv_trans = x_fake_inv_trans.reshape(
            n, GAN_INPUT_SHAPE, GAN_INPUT_SHAPE, 1)

        _, axs = plt.subplots(
            4, 5, sharey=False, tight_layout=True, figsize=(48, 24), facecolor='white')
        k = 0

        plt.xticks([])
        plt.yticks([])
        for row in range(0, 4):
            for col in range(0, 5):
                axs[row, col].matshow(
                    x_fake_inv_trans[k][:, :, 0], cmap='gray')
                """
                Uncomment the following lines if you want to save some of the generated images
                """
                # if k < 10: saveImage(
                #     img=x_fake_inv_trans[k][:, :, 0], folderName='/content/drive/MyDrive/Projects/af-detection/Images/Fake_AF_Images/', fileName=str(k))
                k += 1
        plt.show()

    # In[34]:

    performance_eval(0, gen_model, dis_model, SAMPLES, latent_dim)
    del SAMPLES
    gc.collect()

    # In[38]:

    # ### Appendix - Generate one random Image

    x_fake, _ = fake_samples(gen_model, latent_dim, N_FAKE_SAMPLES)
    # return x_fake[:, :, :, 0]

    x_fake = x_fake[:, :, :, 0]
    shrinked_arr = np.full(
            shape=(x_fake.shape[0], 360, 360), fill_value=PADDING_VAL)
    for i in range(x_fake.shape[0]):
        shrinked_arr[i, cropCount:-cropCount, cropCount:-cropCount] = resize(
            x_fake[i], (AFTER_CROP_SIZE, AFTER_CROP_SIZE))
    print('after interpolation: ', shrinked_arr.shape)

    """
    Uncomment the following lines if you want to view/save the GAN model's diagrams
    """
    # import tensorflow as tf
    # folderName = '/content/drive/MyDrive/Projects/af-detection/Images/Model_Diagrams/'
    # tf.keras.utils.plot_model(
    #     gan_model, to_file=folderName + "GAN_model.png", show_shapes=True)
    # tf.keras.utils.plot_model(
    #     dis_model, to_file=folderName + "discriminator_model.png", show_shapes=True)
    # tf.keras.utils.plot_model(
    #     gen_model, to_file=folderName + "generator_model.png", show_shapes=True)
    return shrinked_arr
