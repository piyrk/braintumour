from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .evaluation import frechet_distance


_GAN_CHECKPOINT_RE = re.compile(r"epoch_(\d{3})\.keras$")


def _latest_gan_epoch(checkpoint_dir: Path) -> int | None:
    candidates = sorted(checkpoint_dir.glob("generator_epoch_*.keras"))
    if not candidates:
        return None

    latest = max(
        candidates,
        key=lambda path: int(_GAN_CHECKPOINT_RE.search(path.name).group(1)) if _GAN_CHECKPOINT_RE.search(path.name) else -1,
    )
    match = _GAN_CHECKPOINT_RE.search(latest.name)
    return int(match.group(1)) if match else None


def _load_history(history_path: Path) -> dict[str, list[float]]:
    if not history_path.exists():
        return {"generator_loss": [], "discriminator_loss": [], "fid": []}

    with open(history_path, encoding="utf-8") as stream:
        history = json.load(stream)

    return {
        "generator_loss": list(history.get("generator_loss", [])),
        "discriminator_loss": list(history.get("discriminator_loss", [])),
        "fid": list(history.get("fid", [])),
    }


def save_generated_grid(images, output_path: str | Path, grid_size: int = 4):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = np.asarray(images)
    images = (images + 1.0) / 2.0
    images = np.clip(images, 0.0, 1.0)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            axes[row, col].imshow(images[index].squeeze(), cmap="gray")
            axes[row, col].axis("off")
            index += 1

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def train_gan(
    generator,
    discriminator,
    dataset,
    latent_dim: int,
    epochs: int,
    output_dir: str | Path,
):
    output_dir = Path(output_dir)
    sample_dir = output_dir / "samples"
    checkpoint_dir = output_dir / "checkpoints"
    history_path = output_dir / "history.json"
    sample_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = _load_history(history_path)
    start_epoch = len(history["generator_loss"]) + 1

    latest_epoch = _latest_gan_epoch(checkpoint_dir)
    if latest_epoch is not None:
        generator = tf.keras.models.load_model(checkpoint_dir / f"generator_epoch_{latest_epoch:03d}.keras")
        discriminator = tf.keras.models.load_model(checkpoint_dir / f"discriminator_epoch_{latest_epoch:03d}.keras")
        start_epoch = max(start_epoch, latest_epoch + 1)

    if start_epoch > epochs:
        print(f"GAN training already completed through epoch {epochs}.")
        return history

    discriminator.trainable = True
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    gan_input = tf.keras.layers.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = tf.keras.Model(gan_input, gan_output, name="gan_train_model")
    gan.compile(optimizer=generator_optimizer, loss=bce)

    first_batch = next(iter(dataset), None)

    if first_batch is None:
        raise ValueError("GAN dataset is empty")

    for epoch in range(start_epoch, epochs + 1):
        generator_losses = []
        discriminator_losses = []

        for real_images in dataset:
            batch_size = int(real_images.shape[0] or tf.shape(real_images)[0].numpy())
            noise = tf.random.normal((batch_size, latent_dim))
            generated_images = generator(noise, training=True)

            real_labels = tf.ones((batch_size, 1))
            fake_labels = tf.zeros((batch_size, 1))

            with tf.GradientTape() as disc_tape:
                real_predictions = discriminator(real_images, training=True)
                fake_predictions = discriminator(generated_images, training=True)
                real_loss = bce(real_labels, real_predictions)
                fake_loss = bce(fake_labels, fake_predictions)
                discriminator_loss = real_loss + fake_loss

            disc_gradients = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

            noise = tf.random.normal((batch_size, latent_dim))
            misleading_labels = tf.ones((batch_size, 1))

            with tf.GradientTape() as gen_tape:
                generated_images = generator(noise, training=True)
                predictions = discriminator(generated_images, training=False)
                generator_loss = bce(misleading_labels, predictions)

            gen_gradients = gen_tape.gradient(generator_loss, generator.trainable_variables)
            gan.optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

            generator_losses.append(float(generator_loss.numpy()))
            discriminator_losses.append(float(discriminator_loss.numpy()))

        epoch_generator_loss = float(np.mean(generator_losses)) if generator_losses else 0.0
        epoch_discriminator_loss = float(np.mean(discriminator_losses)) if discriminator_losses else 0.0
        history["generator_loss"].append(epoch_generator_loss)
        history["discriminator_loss"].append(epoch_discriminator_loss)

        sample_noise = tf.random.normal((16, latent_dim))
        samples = generator(sample_noise, training=False).numpy()
        save_generated_grid(samples, sample_dir / f"epoch_{epoch:03d}.png")

        fid_value = frechet_distance(first_batch[:16].numpy(), samples)
        history["fid"].append(float(fid_value))

        generator.save(checkpoint_dir / f"generator_epoch_{epoch:03d}.keras")
        discriminator.save(checkpoint_dir / f"discriminator_epoch_{epoch:03d}.keras")

        print(
            f"Epoch {epoch:03d}/{epochs} - generator_loss: {epoch_generator_loss:.4f} - discriminator_loss: {epoch_discriminator_loss:.4f} - fid: {fid_value:.4f}"
        )

    with open(history_path, "w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["generator_loss"], label="generator_loss")
    ax.plot(history["discriminator_loss"], label="discriminator_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GAN Training Losses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["fid"], label="fid")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID")
    ax.set_title("GAN FID vs Epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fid_curve.png", dpi=200)
    plt.close(fig)

    return history
