from pathlib import Path

from src.gan_brain_tumour_challenge.config import CONFIG
from src.gan_brain_tumour_challenge.data import load_unlabeled_image_dataset
from src.gan_brain_tumour_challenge.gan_training import train_gan
from src.gan_brain_tumour_challenge.models import build_generator, build_discriminator
from src.gan_brain_tumour_challenge.runtime import configure_training_gpu


def main():
    selected_gpu = configure_training_gpu(require_name_contains=None, require_gpu=True)
    print(f"Using GPU: {selected_gpu}")

    generator = build_generator(CONFIG.latent_dim, (*CONFIG.gan_image_size, CONFIG.grayscale_channels))
    discriminator = build_discriminator((*CONFIG.gan_image_size, CONFIG.grayscale_channels))
    generator.summary()
    discriminator.summary()

    output_dir = Path("artifacts/gan")

    dataset_root = Path("data/processed/gan_images")
    if not dataset_root.exists():
        raise FileNotFoundError("Expected unlabeled GAN images under data/processed/gan_images")

    dataset = load_unlabeled_image_dataset(
        dataset_root,
        image_size=CONFIG.gan_image_size,
        batch_size=CONFIG.batch_size_gan,
    )
    history = train_gan(
        generator=generator,
        discriminator=discriminator,
        dataset=dataset,
        latent_dim=CONFIG.latent_dim,
        epochs=CONFIG.epochs_gan,
        output_dir=output_dir,
    )
    print(history)


if __name__ == "__main__":
    main()
