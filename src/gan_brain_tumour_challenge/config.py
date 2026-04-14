from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    detection_image_size: tuple[int, int] = (224, 224)
    segmentation_image_size: tuple[int, int] = (256, 256)
    gan_image_size: tuple[int, int] = (64, 64)
    grayscale_channels: int = 1
    num_classes: int = 4
    latent_dim: int = 100
    batch_size_detection: int = 32
    batch_size_segmentation: int = 8
    batch_size_classifier: int = 16
    batch_size_gan: int = 64
    learning_rate_detection: float = 1e-3
    learning_rate_segmentation: float = 1e-4
    learning_rate_classifier: float = 1e-4
    learning_rate_gan: float = 2e-4
    epochs_detection: int = 30
    epochs_segmentation: int = 50
    epochs_classifier: int = 40
    epochs_gan: int = 160


CONFIG = AppConfig()
