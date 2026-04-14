from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_detection_model(input_shape=(224, 224, 1)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.LayerNormalization(axis=-1)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.LayerNormalization(axis=-1)(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.LayerNormalization(axis=-1)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="tumour_detection_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True,
    )
    return model


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_unet(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D()(c3)

    b = conv_block(p3, 512)

    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1, c3])
    c4 = conv_block(u1, 256)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = conv_block(u2, 128)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3, c1])
    c6 = conv_block(u3, 64)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c6)

    model = models.Model(inputs, outputs, name="brain_tumour_unet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True,
    )
    return model


def build_classifier(num_classes=4, input_shape=(224, 224, 1), pretrained=False):
    weights = "imagenet" if pretrained and input_shape[-1] == 3 else None
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(base.input, outputs, name="tumour_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True,
    )
    return model


def build_generator(latent_dim=100, output_shape=(64, 64, 1)):
    channels = output_shape[-1]
    model = models.Sequential(
        [
            layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(latent_dim,)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Reshape((8, 8, 256)),
            layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(channels, 4, strides=2, padding="same", activation="tanh"),
        ],
        name="generator",
    )
    return model


def build_discriminator(input_shape=(64, 64, 1)):
    model = models.Sequential(
        [
            layers.Conv2D(64, 4, strides=2, padding="same", input_shape=input_shape),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Conv2D(128, 4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True,
    )
    return model


def build_gan(generator, discriminator):
    discriminator.trainable = False

    z = layers.Input(shape=(generator.input_shape[-1],))
    img = generator(z)
    validity = discriminator(img)

    gan = models.Model(z, validity, name="gan")
    gan.compile(
        optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
        loss="binary_crossentropy",
        run_eagerly=True,
    )
    return gan
