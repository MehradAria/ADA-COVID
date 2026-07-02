"""Adversarial two-domain training loop for ADA-COVID."""

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score

from .data import batch_generator


def prepare_triplet_labels(one_hot_labels: np.ndarray) -> np.ndarray:
    return one_hot_labels.argmax(axis=1).reshape(-1, 1).astype(np.float32)


def _nearest_centroid_predict(embeds: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    dists = np.linalg.norm(embeds[:, None, :] - centroids[None, :, :], axis=2)
    return dists.argmin(axis=1)


def train(config: dict, models: dict, dataset: dict) -> Dict[str, list]:
    m_comb = models["combined_model"]
    m_cls = models["combined_classifier"]
    m_dis = models["combined_discriminator"]
    net_name = config.get("network_name", "ADA-COVID")

    Xs, ys = dataset["source_data"], dataset["source_label"]
    Xt, yt = dataset["target_data"], dataset["target_label"]
    num_classes = dataset["num_classes"]

    half_bs = config["batch_size"] // 2

    y_adv_1 = np.array([1] * half_bs + [0] * half_bs, dtype=np.float32)
    y_adv_2 = np.array([0] * half_bs + [1] * half_bs, dtype=np.float32)

    weight_class = np.array([1.0] * half_bs + [0.0] * half_bs, dtype=np.float32)
    weight_adv = np.ones(config["batch_size"], dtype=np.float32)

    s_gen = batch_generator([Xs, ys], half_bs)
    t_gen = batch_generator([Xt, np.zeros((len(Xt), ys.shape[1]))], half_bs)

    best_target_acc = 0.0
    history: Dict[str, list] = {"iter": [], "src_acc": [], "tgt_acc": [], "dis_acc": []}
    gap_snap = 0

    for i in range(config["num_iterations"]):
        Xsb, ysb = next(s_gen)
        Xtb, _ = next(t_gen)

        X_adv = np.concatenate([Xsb, Xtb], axis=0)
        ysb_int = prepare_triplet_labels(ysb)
        zero_t = np.zeros((half_bs, 1), dtype=np.float32)
        y_class_triplet = np.concatenate([ysb_int, zero_t], axis=0)

        dis_weights = [
            layer.get_weights()
            for layer in m_comb.layers
            if layer.name.startswith("dis_") or layer.name == "grl"
        ]
        m_comb.train_on_batch(
            X_adv,
            {"class_l2_norm": y_class_triplet, "dis_act_last": y_adv_1},
            sample_weight={"class_l2_norm": weight_class, "dis_act_last": weight_adv},
        )
        k = 0
        for layer in m_comb.layers:
            if layer.name.startswith("dis_") or layer.name == "grl":
                layer.set_weights(dis_weights[k])
                k += 1

        cls_weights = [
            layer.get_weights()
            for layer in m_comb.layers
            if not (layer.name.startswith("dis_") or layer.name == "grl")
        ]
        m_dis.train_on_batch(X_adv, y_adv_2)
        k = 0
        for layer in m_comb.layers:
            if not (layer.name.startswith("dis_") or layer.name == "grl"):
                layer.set_weights(cls_weights[k])
                k += 1

        if (i + 1) % config["test_interval"] == 0:
            ys_embed = m_cls.predict(Xs, verbose=0)
            yt_embed = m_cls.predict(Xt, verbose=0)
            ys_adv_pred = m_dis.predict(Xs, verbose=0)
            yt_adv_pred = m_dis.predict(Xt, verbose=0)

            class_ids = ys.argmax(1)
            centroids = np.array(
                [ys_embed[class_ids == c].mean(axis=0) for c in range(num_classes)]
            )

            ys_pred_cls = _nearest_centroid_predict(ys_embed, centroids)
            yt_pred_cls = _nearest_centroid_predict(yt_embed, centroids)

            src_acc = accuracy_score(ys.argmax(1), ys_pred_cls)
            tgt_acc = accuracy_score(yt.argmax(1), yt_pred_cls)
            src_dom_acc = accuracy_score(np.zeros(len(Xs)), np.round(ys_adv_pred.flatten()))
            tgt_dom_acc = accuracy_score(np.ones(len(Xt)), np.round(yt_adv_pred.flatten()))

            log_str = (
                f"[Iter {i+1:05d}] "
                f"Src Cls: {src_acc*100:.2f}% | Tgt Cls: {tgt_acc*100:.2f}% | "
                f"Src Dom: {src_dom_acc*100:.2f}% | Tgt Dom: {tgt_dom_acc*100:.2f}%"
            )
            print(log_str)

            history["iter"].append(i + 1)
            history["src_acc"].append(src_acc * 100)
            history["tgt_acc"].append(tgt_acc * 100)
            history["dis_acc"].append((src_dom_acc + tgt_dom_acc) / 2 * 100)

            if tgt_acc > best_target_acc and gap_snap >= config["snapshot_interval"]:
                best_target_acc = tgt_acc
                m_cls.save(f"Best_Classifier_{net_name}.keras")
                print(f"New best target accuracy: {best_target_acc*100:.2f}% - model saved.")
                gap_snap = 0

            with open(f"Log_{net_name}.txt", "a") as f:
                f.write(log_str + "\n")

            gap_snap += 1

    return history


def stage2_train(inference_model, dataset: dict, seed: int, epochs: int = 30):
    from sklearn.model_selection import StratifiedKFold
    import tensorflow as tf

    Xs = dataset["source_data"]
    ys = dataset["source_label"]
    ys_int = ys.argmax(1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(Xs, ys_int)):
        print(f"--- Fold {fold + 1}/5 ---")
        X_train, X_val = Xs[train_idx], Xs[val_idx]
        y_train, y_val = ys[train_idx], ys[val_idx]

        hist = inference_model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True, monitor="val_accuracy"
                )
            ],
        )
        val_acc = max(hist.history["val_accuracy"])
        fold_results.append(val_acc)
        print(f"Fold {fold+1} best val accuracy: {val_acc*100:.2f}%")

    print(
        f"5-Fold CV mean accuracy: {np.mean(fold_results)*100:.2f}% "
        f"(+/-{np.std(fold_results)*100:.2f}%)"
    )
    return fold_results
