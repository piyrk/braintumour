from pathlib import Path

from src.gan_brain_tumour_challenge.config import CONFIG
from src.gan_brain_tumour_challenge.data import load_multiclass_dataset, split_dataset
from src.gan_brain_tumour_challenge.models import build_classifier
from src.gan_brain_tumour_challenge.runtime import configure_training_gpu
from src.gan_brain_tumour_challenge.training import make_callbacks, resume_training_model


def main():
    output_dir = Path("artifacts/classifier")
    selected_gpu = configure_training_gpu(require_name_contains=None, require_gpu=True)
    print(f"Using GPU: {selected_gpu}")

    dataset_root = Path("data/processed/classification")
    if not dataset_root.exists():
        raise FileNotFoundError("Expected classification folders under data/processed/classification/<class_name>")

    dataset, class_names = load_multiclass_dataset(
        dataset_root,
        image_size=CONFIG.detection_image_size,
        batch_size=CONFIG.batch_size_classifier,
    )
    if len(class_names) < 2:
        raise ValueError(
            "Classifier training requires at least 2 classes under data/processed/classification. "
            f"Found classes: {class_names}"
        )

    train_ds, val_ds = split_dataset(dataset)

    expected_classes = len(class_names)
    model = build_classifier(expected_classes, (*CONFIG.detection_image_size, CONFIG.grayscale_channels))
    initial_epoch = 0
    try:
        resumed_model, resumed_epoch = resume_training_model(model, output_dir)
        resumed_units = int(resumed_model.output_shape[-1])
        if resumed_units == expected_classes:
            model = resumed_model
            initial_epoch = resumed_epoch
        else:
            print(
                "Found incompatible classifier checkpoint "
                f"(units={resumed_units}, expected={expected_classes}); starting fresh training."
            )
    except Exception as exc:
        print(f"Could not resume classifier checkpoint ({exc}); starting fresh training.")

    model.summary()
    if initial_epoch:
        print(f"Resuming classifier training from epoch {initial_epoch + 1}")

    callbacks = make_callbacks(output_dir)
    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=CONFIG.epochs_classifier,
        callbacks=callbacks,
    )
    print(f"Trained classifier for classes: {class_names}")


if __name__ == "__main__":
    main()
