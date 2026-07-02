"""Model architecture: ResNet50 feature extractor, classifier, and discriminator heads."""

from typing import Dict

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    Lambda,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .layers import GradientReversalLayer
from .losses import triplet_semihard_loss


def build_feature_extractor(inp: tf.keras.Input) -> tf.Tensor:
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    base.trainable = True
    return base(inp)


def build_classifier_head(features: tf.Tensor, config: dict) -> tf.Tensor:
    embedding_size = config["embedding_size"]
    drop_rate = config["drop_classifier"]

    x = Dense(400, name="class_dense1")(features)
    x = BatchNormalization(name="class_bn1")(x)
    x = Activation("relu", name="class_act1")(x)
    x = Dropout(drop_rate, name="class_drop1")(x)

    x = Dense(100, name="class_dense2")(x)
    x = BatchNormalization(name="class_bn2")(x)
    x = Activation("relu", name="class_act2")(x)
    x = Dropout(drop_rate, name="class_drop2")(x)

    embeddings = Dense(embedding_size, activation=None, name="class_embed")(x)
    embeddings = Lambda(
        lambda v: tf.math.l2_normalize(v, axis=1), name="class_l2_norm"
    )(embeddings)
    return embeddings


def build_discriminator_head(features: tf.Tensor, config: dict) -> tf.Tensor:
    drop_rate = config["drop_discriminator"]

    x = GradientReversalLayer(hp_lambda=1.0, name="grl")(features)

    x = Dense(400, name="dis_dense1")(x)
    x = BatchNormalization(name="dis_bn1")(x)
    x = Activation("relu", name="dis_act1")(x)
    x = Dropout(drop_rate, name="dis_drop1")(x)

    x = Dense(100, name="dis_dense2")(x)
    x = BatchNormalization(name="dis_bn2")(x)
    x = Activation("relu", name="dis_act2")(x)
    x = Dropout(drop_rate, name="dis_drop2")(x)

    x = Dense(1, name="dis_dense_last")(x)
    x = BatchNormalization(name="dis_bn_last")(x)
    x = Activation("sigmoid", name="dis_act_last")(x)
    return x


def build_inference_head(embeddings: tf.Tensor, num_classes: int) -> tf.Tensor:
    x = Dense(num_classes, name="infer_dense")(embeddings)
    x = BatchNormalization(name="infer_bn")(x)
    x = Activation("softmax", name="infer_softmax")(x)
    return x


def build_all_models(config: dict) -> Dict[str, object]:
    num_gpus = config["number_of_gpus"]

    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        inp = Input(shape=config["inp_dims"], name="input_image")

        features = build_feature_extractor(inp)
        classifier = build_classifier_head(features, config)
        discriminator = build_discriminator_head(features, config)

        m_classifier = Model(inputs=inp, outputs=classifier, name="classifier_model")
        m_classifier.compile(
            optimizer=Adam(
                learning_rate=config["lr_classifier"],
                beta_1=config["b1_classifier"],
                beta_2=config["b2_classifier"],
            ),
            loss=triplet_semihard_loss,
        )

        m_discriminator = Model(
            inputs=inp, outputs=discriminator, name="discriminator_model"
        )
        m_discriminator.compile(
            optimizer=Adam(
                learning_rate=config["lr_discriminator"],
                beta_1=config["b1_discriminator"],
                beta_2=config["b2_discriminator"],
            ),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        m_combined = Model(
            inputs=inp, outputs=[classifier, discriminator], name="combined_model"
        )
        m_combined.compile(
            optimizer=Adam(
                learning_rate=config["lr_combined"],
                beta_1=config["b1_combined"],
                beta_2=config["b2_combined"],
            ),
            loss={
                "class_l2_norm": triplet_semihard_loss,
                "dis_act_last": "binary_crossentropy",
            },
            loss_weights={
                "class_l2_norm": config["class_loss_weight"],
                "dis_act_last": config["dis_loss_weight"],
            },
            metrics={"dis_act_last": "accuracy"},
        )

    return {
        "combined_classifier": m_classifier,
        "combined_discriminator": m_discriminator,
        "combined_model": m_combined,
        "inp": inp,
        "classifier_output": classifier,
        "features": features,
    }


def build_inference_model(trained_classifier: tf.keras.Model, config: dict) -> tf.keras.Model:
    num_classes = config["num_classes"]

    embedding_model = trained_classifier
    embedding_model.trainable = False

    inp = embedding_model.input
    emb = embedding_model.output

    out = Dense(num_classes, name="infer_dense")(emb)
    out = BatchNormalization(name="infer_bn")(out)
    out = Activation("softmax", name="infer_softmax")(out)

    inference_model = Model(inputs=inp, outputs=out, name="inference_model")
    inference_model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return inference_model
