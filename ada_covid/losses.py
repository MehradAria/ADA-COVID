"""Semi-hard triplet loss (TF2-native, eager-compatible)."""

import tensorflow as tf


def pairwise_distance(feature: tf.Tensor, squared: bool = False) -> tf.Tensor:
    sq_sum = tf.reduce_sum(tf.square(feature), axis=1, keepdims=True)
    distances_sq = (
        sq_sum + tf.transpose(sq_sum) - 2.0 * tf.matmul(feature, tf.transpose(feature))
    )
    distances_sq = tf.maximum(distances_sq, 0.0)
    error_mask = tf.less_equal(distances_sq, 0.0)

    if squared:
        distances = distances_sq
    else:
        distances = tf.sqrt(distances_sq + tf.cast(error_mask, tf.float32) * 1e-16)

    distances = distances * tf.cast(tf.logical_not(error_mask), tf.float32)
    num_data = tf.shape(feature)[0]
    mask_offdiag = tf.ones_like(distances) - tf.linalg.diag(tf.ones([num_data]))
    return distances * mask_offdiag


def masked_maximum(data: tf.Tensor, mask: tf.Tensor, dim: int = 1) -> tf.Tensor:
    axis_min = tf.reduce_min(data, axis=dim, keepdims=True)
    return tf.reduce_max((data - axis_min) * mask, axis=dim, keepdims=True) + axis_min


def masked_minimum(data: tf.Tensor, mask: tf.Tensor, dim: int = 1) -> tf.Tensor:
    axis_max = tf.reduce_max(data, axis=dim, keepdims=True)
    return tf.reduce_min((data - axis_max) * mask, axis=dim, keepdims=True) + axis_max


def triplet_semihard_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    del y_true
    margin = 1.0

    labels = tf.cast(y_pred[:, :1], dtype=tf.int32)
    embeddings = y_pred[:, 1:]

    pdist_matrix = pairwise_distance(embeddings, squared=True)

    adjacency = tf.equal(labels, tf.transpose(labels))
    adjacency_not = tf.logical_not(adjacency)

    batch_size = tf.shape(labels)[0]

    pdist_tile = tf.tile(pdist_matrix, [batch_size, 1])
    mask = tf.logical_and(
        tf.tile(adjacency_not, [batch_size, 1]),
        tf.greater(pdist_tile, tf.reshape(tf.transpose(pdist_matrix), [-1, 1])),
    )
    mask_final = tf.reshape(
        tf.greater(tf.reduce_sum(tf.cast(mask, tf.float32), 1, keepdims=True), 0.0),
        [batch_size, batch_size],
    )
    mask_final = tf.transpose(mask_final)

    adjacency_not_f = tf.cast(adjacency_not, tf.float32)
    mask_f = tf.cast(mask, tf.float32)

    negatives_outside = tf.reshape(
        masked_minimum(pdist_tile, mask_f), [batch_size, batch_size]
    )
    negatives_outside = tf.transpose(negatives_outside)

    negatives_inside = tf.tile(
        masked_maximum(pdist_matrix, adjacency_not_f), [1, batch_size]
    )

    semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_pos = tf.cast(adjacency, tf.float32) - tf.linalg.diag(tf.ones([batch_size]))
    num_positives = tf.reduce_sum(mask_pos)

    loss = tf.math.truediv(
        tf.reduce_sum(tf.maximum(loss_mat * mask_pos, 0.0)),
        num_positives,
        name="triplet_semihard_loss",
    )
    return loss
