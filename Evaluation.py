
def evaluation(normal_1, af_1, augment_1, normal_2, af_2, augment_2,
               normal_3, af_3, augment_3, val_normal, val_af, checkpoint_filepath: str, showModelSummary=False):
    """
    Import Modules
    """

    import numpy as np
    from keras.layers import BatchNormalization
    from keras import layers, models
    import tensorflow as tf
    from sklearn.utils import shuffle
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import gc

    """
    Set Datasets
    """

    train = np.concatenate(
        [normal_1, normal_2, normal_3, af_1, af_2, af_3], axis=0)
    n_normals = normal_1.shape[0] + normal_2.shape[0] + normal_3.shape[0]
    n_af = af_1.shape[0] + af_2.shape[0] + af_3.shape[0]
    if augment_1 is not None:
        train = np.concatenate(
            [train, augment_1, augment_2, augment_3], axis=0)
        n_af += (augment_1.shape[0] +
                 augment_2.shape[0] + augment_3.shape[0])
        print('Added fake samples')

    trainLabels = np.concatenate([np.zeros(n_normals), np.ones(n_af)], axis=0)
    train, trainLabels = shuffle(train, trainLabels)
    trainLabels = pd.DataFrame(trainLabels)
    trainLabels = OneHotEncoder().fit_transform(trainLabels).toarray()

    val = np.concatenate([val_normal, val_af], axis=0)
    valLabels = np.concatenate(
        [np.zeros(val_normal.shape[0]), np.ones(val_af.shape[0])], axis=0)
    val, valLabels = shuffle(val, valLabels)
    valLabels = pd.DataFrame(valLabels)
    valLabels = OneHotEncoder().fit_transform(valLabels).toarray()

    """
    Define Model
    """

    model = models.Sequential()
    model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(
        train.shape[1], train.shape[2], 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    # model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(BatchNormalization())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(BatchNormalization())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))

    if showModelSummary: model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Recall(name="recall"),
                           tf.keras.metrics.Precision(name="precision")
                           ])

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        verbose=1,
        mode='max',
        save_best_only=True)

    """
    Train Model
    """

    gc.collect()
    model.fit(train, trainLabels, epochs=25,
                        validation_data=(val, valLabels), callbacks=[model_checkpoint_callback])

    """
    Load Best Weights
    """

    model.load_weights(checkpoint_filepath)

    """
    Compute Accuracy for Train Data
    """

    y_pred_train = model.predict(train)
    trainLabels = tf.argmax(trainLabels, axis=1)
    print(trainLabels)

    train_prediction = tf.argmax(y_pred_train, axis=1)
    print(train_prediction)

    con_mat = tf.math.confusion_matrix(
        labels=trainLabels, predictions=train_prediction).numpy()
    print(con_mat)

    plt.hist(trainLabels, range=(0, 3), bins=8)
    plt.show()
    plt.hist(train_prediction, range=(0, 3), bins=8)
    plt.show()

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    classes = ['N', 'A']
    con_mat_norm = np.around(con_mat.astype('float') /
                             con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=classes,
                              columns=classes)

    """
    Compute Accuracy for Test Data
    """

    y_pred_test = model.predict(val)
    valLabels = tf.argmax(valLabels, axis=1)
    test_prediction = tf.argmax(y_pred_test, axis=1)
    con_mat = tf.math.confusion_matrix(
        labels=valLabels, predictions=test_prediction).numpy()
    print(con_mat)

    classes = ['N', 'A']
    con_mat_norm = np.around(con_mat.astype('float') /
                             con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=classes,
                              columns=classes)

    plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()