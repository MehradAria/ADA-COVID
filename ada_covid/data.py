"""Data loading, normalization, augmentation, and batching utilities."""

import os
from typing import Iterator, List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Lambda, RandomContrast, RandomFlip, RandomRotation
from tensorflow.keras.utils import to_categorical


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def data_loader(filepath: str, inp_dims: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    imgs: List[np.ndarray] = []
    labels: List[int] = []
    with open(filepath) as fp:
        for line in fp:
            token = line.strip().split()
            if len(token) < 2:
                continue
            img = pil_loader(token[0])
            img = img.resize((inp_dims[0], inp_dims[1]), Image.LANCZOS)
            imgs.append(np.array(img, dtype=np.float32))
            labels.append(int(token[1]))

    return np.array(imgs), np.array(labels, dtype=np.int32)


def normalize_data(imgs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    mu = imgs.mean()
    sigma = imgs.std()
    return (imgs - mu) / (sigma + eps)


def one_hot_encoding(
    source_label: np.ndarray, target_label: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    all_classes = np.concatenate([source_label, target_label])
    num_classes = int(np.max(all_classes)) + 1
    s = to_categorical(source_label, num_classes=num_classes)
    t = to_categorical(target_label, num_classes=num_classes)
    return s, t


def batch_generator(data: List[np.ndarray], batch_size: int) -> Iterator[List[np.ndarray]]:
    n = len(data[0])
    while True:
        idx = np.random.choice(n, size=batch_size, replace=False)
        yield [arr[idx] for arr in data]


def build_augmentation_pipeline() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(20 / 360),
            RandomContrast(0.10),
            Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.10)),
            Lambda(
                lambda x: tf.image.adjust_jpeg_quality(
                    tf.cast(x * 127.5 + 127.5, tf.uint8),
                    jpeg_quality=tf.random.uniform([], 50, 100, dtype=tf.int32),
                )
                / 255.0
            ),
        ],
        name="augmentation",
    )


def load_and_preprocess_datasets(config: dict) -> dict:
    Xs_raw, ys_raw = data_loader(config["source_path"], config["inp_dims"])
    Xt_raw, yt_raw = data_loader(config["target_path"], config["inp_dims"])

    Xs = normalize_data(Xs_raw)
    Xt = normalize_data(Xt_raw)

    ys, yt = one_hot_encoding(ys_raw, yt_raw)

    ys_adv = np.zeros(len(Xs), dtype=np.float32)
    yt_adv = np.ones(len(Xt), dtype=np.float32)

    return {
        "source_data": Xs,
        "source_label": ys,
        "target_data": Xt,
        "target_label": yt,
        "source_domain_label": ys_adv,
        "target_domain_label": yt_adv,
        "num_classes": ys.shape[1],
    }


def create_dataset_txt(
    root_dir: str,
    output_file: str,
    covid_subdir: str = "COVID",
    noncovid_subdir: str = "non-COVID",
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> None:
    lines = []
    for label, subdir in [(1, covid_subdir), (0, noncovid_subdir)]:
        folder = os.path.join(root_dir, subdir)
        if not os.path.isdir(folder):
            print(f"Warning: directory not found: {folder}")
            continue
        for fname in sorted(os.listdir(folder)):
            if any(fname.lower().endswith(ext) for ext in extensions):
                lines.append(f"{os.path.join(folder, fname)} {label}")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    print(f"Written {len(lines)} entries to {output_file}")
