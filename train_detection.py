from pathlib import Path

from src.gan_brain_tumour_challenge.config import CONFIG
from src.gan_brain_tumour_challenge.data import load_binary_detection_dataset, split_dataset
from src.gan_brain_tumour_challenge.models import build_detection_model
from src.gan_brain_tumour_challenge.runtime import configure_training_gpu
from src.gan_brain_tumour_challenge.training import make_callbacks, resume_training_model


def main():
    output_dir = Path("artifacts/detection")
    selected_gpu = configure_training_gpu(require_name_contains=None, require_gpu=True)
    print(f"Using GPU: {selected_gpu}")

    dataset_root = Path("data/processed/detection")
    if not dataset_root.exists():
        raise FileNotFoundError("Expected detection data under data/processed/detection/{normal,tumour}")

    dataset, class_names = load_binary_detection_dataset(
        dataset_root,
        image_size=CONFIG.detection_image_size,
        batch_size=CONFIG.batch_size_detection,
    )
    train_ds, val_ds = split_dataset(dataset)

    model = build_detection_model((*CONFIG.detection_image_size, CONFIG.grayscale_channels))
    model, initial_epoch = resume_training_model(model, output_dir)
    model.summary()
    if initial_epoch:
        print(f"Resuming detection training from epoch {initial_epoch + 1}")

    callbacks = make_callbacks(output_dir)
    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=CONFIG.epochs_detection,
        callbacks=callbacks,
    )
    print(f"Trained detection model for classes: {class_names}")


if __name__ == "__main__":
    main()
