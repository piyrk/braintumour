from pathlib import Path

from src.gan_brain_tumour_challenge.config import CONFIG
from src.gan_brain_tumour_challenge.data import load_segmentation_dataset, split_dataset
from src.gan_brain_tumour_challenge.models import build_unet
from src.gan_brain_tumour_challenge.runtime import configure_training_gpu
from src.gan_brain_tumour_challenge.training import make_callbacks, resume_training_model


def main():
    output_dir = Path("artifacts/segmentation")
    selected_gpu = configure_training_gpu(require_name_contains=None, require_gpu=True)
    print(f"Using GPU: {selected_gpu}")

    images_dir = Path("data/processed/segmentation/images")
    masks_dir = Path("data/processed/segmentation/masks")

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(
            "Expected segmentation data under data/processed/segmentation/images and data/processed/segmentation/masks"
        )

    dataset = load_segmentation_dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=CONFIG.segmentation_image_size,
        batch_size=CONFIG.batch_size_segmentation,
    )
    train_ds, val_ds = split_dataset(dataset)

    model = build_unet((*CONFIG.segmentation_image_size, CONFIG.grayscale_channels))
    model, initial_epoch = resume_training_model(model, output_dir)
    model.summary()
    if initial_epoch:
        print(f"Resuming segmentation training from epoch {initial_epoch + 1}")

    callbacks = make_callbacks(output_dir)
    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=CONFIG.epochs_segmentation,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
