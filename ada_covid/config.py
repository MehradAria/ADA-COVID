"""Configuration and hyperparameters for ADA-COVID (paper Section 4)."""

from typing import Any, Dict


def get_default_config() -> Dict[str, Any]:
    return {
        "network_name": "ResNet50",
        "inp_dims": (224, 224, 3),
        "embedding_size": 64,
        "drop_classifier": 0.5,
        "drop_discriminator": 0.5,
        "batch_size": 32,
        "num_iterations": 20000,
        "test_interval": 200,
        "snapshot_interval": 500,
        "class_loss_weight": 4.0,
        "dis_loss_weight": 1.0,
        "lr_classifier": 1e-4,
        "b1_classifier": 0.9,
        "b2_classifier": 0.999,
        "lr_discriminator": 1e-5,
        "b1_discriminator": 0.9,
        "b2_discriminator": 0.999,
        "lr_combined": 1e-5,
        "b1_combined": 0.9,
        "b2_combined": 0.999,
        "source_path": "Source.txt",
        "target_path": "Target.txt",
        "number_of_gpus": 1,
        "dataset_name": "COVID",
    }


SEED = 27
