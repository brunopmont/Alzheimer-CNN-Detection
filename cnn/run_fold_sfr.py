#!/usr/bin/env python
# coding: utf-8
"""
Treina UM fold do k-fold do SFR_training.ipynb e sai.

Roda como subprocesso isolado (chamado pelo notebook via subprocess.run, um
processo por fold) para garantir que a RAM seja devolvida ao SO quando o
processo termina -- K.clear_session() + gc.collect() nao garantem 100% de
liberacao de memoria residente do TF/Keras entre folds no mesmo processo, o
que estava acumulando ate matar o kernel do Jupyter num fold especifico
(fold 4 na ultima execucao).

Uso:
    python run_fold_sfr.py --fold-idx 4 --n-folds 5 --results-dir <dir> \
        --kfold-images <path.npy> --kfold-labels <path.npy> \
        --mask-path <path.nii> --alpha 0.0 --epochs 200 --patience 30 \
        --batch-size 16 --class-names cn ad

Escreve <results_dir>/kfold_<fold_idx>/fold_result.json ao final. O notebook
le esse arquivo pra decidir se o fold ja foi concluido.
"""
import argparse
import gc
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # sem display no servidor: plt.show() do utils nao pode tentar abrir janela

import numpy as np
import nibabel as nib
import tensorflow as tf
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from skimage.transform import resize
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.metricas_e_visualizacao import get_classification_report, plot_confusion_matrix


def save_auc(pred, true_labels, results_dir, subset):
    y_probs = pred[:, 1] if pred.ndim > 1 and pred.shape[1] > 1 else pred
    auc_value = roc_auc_score(true_labels, y_probs)
    with open(os.path.join(results_dir, f"{subset}_auc.txt"), "w") as f:
        f.write(f"AUC Score: {auc_value:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold-idx", type=int, required=True)
    p.add_argument("--n-folds", type=int, required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--kfold-images", required=True)
    p.add_argument("--kfold-labels", required=True)
    p.add_argument("--mask-path", required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--patience", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--last-conv-layer", default="conv3d_2")
    p.add_argument("--class-names", nargs="+", required=True)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def build_mask_crop(mask_path):
    mask = nib.load(mask_path).get_fdata()
    mask_crop = mask[18:174, 17:212, 15:175]
    mask_crop = resize(mask_crop, (18, 22, 18), order=0, preserve_range=True, anti_aliasing=False)
    mask_crop = (mask_crop > 0.5).astype(float)
    mask_crop = 1.0 - mask_crop
    return tf.cast(mask_crop, tf.float32)


def create_model_3d(input_shape, n_classes):
    from tensorflow.keras import Input
    from tensorflow.keras.regularizers import l2

    inputs = Input(shape=input_shape)
    x = layers.Conv3D(8, (3, 3, 3), padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.3)(x)
    x = layers.AveragePooling3D(pool_size=(3, 3, 3), padding='same')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(16, (3, 3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.3)(x)
    x = layers.AveragePooling3D(pool_size=(3, 3, 3), padding='same')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(32, (3, 3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.3)(x)
    x = layers.AveragePooling3D(pool_size=(3, 3, 3), padding='same')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(32, kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LeakyReLU(negative_slope=0.3)(x)

    x = layers.Dense(16, kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LeakyReLU(negative_slope=0.3)(x)

    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs)


class SFRModel(tf.keras.Model):
    def __init__(self, model, alpha, mask_crop, last_conv_layer_name, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.alpha = alpha
        self.mask_crop = tf.cast(mask_crop, tf.float32)
        self.last_conv_layer_name = last_conv_layer_name

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.saliency_tracker = tf.keras.metrics.Mean(name="saliency_loss")
        self.classification_tracker = tf.keras.metrics.Mean(name="classification_loss")

        conv_layer = self.model.get_layer(self.last_conv_layer_name)
        self.grad_model = tf.keras.models.Model([self.model.inputs], [conv_layer.output, self.model.output])

    def get_config(self):
        base_config = super().get_config()
        config = {
            "model": self.model,
            "alpha": self.alpha,
            "mask_crop": self.mask_crop.numpy(),
            "last_conv_layer_name": self.last_conv_layer_name,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _shared_step(self, data, training):
        x, y = data
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        if self.alpha == 0:
            predictions = self.model(x, training=training)
            main_loss = self.compute_loss(x=x, y=y, y_pred=predictions)
            total_loss = main_loss
            mean_saliency_penalty = tf.constant(0.0, dtype=tf.float32)
        else:
            with tf.GradientTape() as gradcam_tape:
                conv_outputs, predictions = self.grad_model(x, training=training)
                class_loss = predictions[:, 1]
            grads = gradcam_tape.gradient(class_loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(1, 2, 3))
            heatmap = tf.reduce_sum(
                tf.multiply(pooled_grads[:, tf.newaxis, tf.newaxis, tf.newaxis, :], conv_outputs), axis=-1
            )
            heatmap = tf.nn.relu(heatmap)
            heatmap_norm = heatmap / (tf.reduce_sum(heatmap, axis=(1, 2, 3), keepdims=True) + 1e-6)
            saliency_penalty_batch = tf.reduce_sum(heatmap_norm * self.mask_crop, axis=(1, 2, 3))
            mean_saliency_penalty = tf.reduce_mean(saliency_penalty_batch)
            main_loss = self.compute_loss(x=x, y=y, y_pred=predictions)
            total_loss = main_loss + (self.alpha * mean_saliency_penalty)

        return x, y, predictions, main_loss, total_loss, mean_saliency_penalty

    def train_step(self, data):
        with tf.GradientTape() as total_tape:
            x, y, predictions, main_loss, total_loss, mean_saliency_penalty = self._shared_step(data, training=True)
        gradients = total_tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return self._update_and_log(y, predictions, main_loss, total_loss, mean_saliency_penalty)

    def test_step(self, data):
        _, y, predictions, main_loss, total_loss, mean_saliency_penalty = self._shared_step(data, training=False)
        return self._update_and_log(y, predictions, main_loss, total_loss, mean_saliency_penalty)

    def _update_and_log(self, y, predictions, main_loss, total_loss, mean_saliency_penalty):
        self.total_loss_tracker.update_state(total_loss)
        self.saliency_tracker.update_state(mean_saliency_penalty)
        self.classification_tracker.update_state(main_loss)

        logs = {
            "total_loss": self.total_loss_tracker.result(),
            "saliency_loss": self.saliency_tracker.result(),
            "classification_loss": self.classification_tracker.result(),
        }
        for metric in self.metrics:
            if metric.name in ("total_loss", "saliency_loss", "classification_loss", "loss"):
                continue
            metric.update_state(y, predictions)
            result = metric.result()
            if isinstance(result, dict):
                logs.update(result)
            else:
                logs[metric.name] = result
        return logs

    @property
    def metrics(self):
        return super().metrics

    def call(self, x):
        return self.model(x)


def memmap_batch_generator(images, labels, indices, batch_size):
    """Le cada lote direto do .npy mapeado em disco.

    Nunca materializa o fold inteiro em RAM, so o lote da vez. Os indices ficam
    em ordem crescente de proposito: leitura sequencial no arquivo mapeado e bem
    mais rapida que acesso espalhado, e o pool ja foi embaralhado quando o cache
    foi escrito, entao a ordem no arquivo ja e aleatoria em relacao a classe.
    """
    total_n = len(indices)
    if total_n == 0:
        raise ValueError("O generator recebeu uma lista vazia de indices!")

    while True:
        for i in range(0, total_n, batch_size):
            batch_idx = indices[i:i + batch_size]
            yield np.asarray(images[batch_idx]), np.asarray(labels[batch_idx])


def get_predictions_memmap(images, labels, indices, batch_size, model):
    pred = []
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        pred.append(model.predict(np.asarray(images[batch_idx]), verbose=0))
    pred = np.concatenate(pred)
    true_labels = np.argmax(np.asarray(labels[indices]), axis=1)
    pred_labels = np.argmax(pred, axis=1)
    return pred_labels, true_labels, pred


def main():
    args = parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    mixed_precision.set_global_policy("mixed_float16")
    tf.get_logger().setLevel('ERROR')

    fold_idx = args.fold_idx
    fold_dir = os.path.join(args.results_dir, f"kfold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    fold_result_path = os.path.join(fold_dir, "fold_result.json")

    if os.path.exists(fold_result_path):
        print(f"Fold {fold_idx} ja possui resultado salvo -- pulando.")
        return

    kfold_images = np.load(args.kfold_images, mmap_mode='r')
    kfold_labels = np.load(args.kfold_labels, mmap_mode='r')
    kfold_y = np.argmax(kfold_labels, axis=1)
    sample_shape = kfold_images.shape[1:]

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)
    splits = list(skf.split(kfold_images, kfold_y))
    train_idx, val_idx = splits[fold_idx - 1]

    K.clear_session()
    gc.collect()

    mask_crop = build_mask_crop(args.mask_path)

    # Lotes lidos sob demanda do arquivo mapeado -- nenhum dos dois folds e
    # materializado inteiro em RAM
    fold_train_gen = memmap_batch_generator(kfold_images, kfold_labels, train_idx, args.batch_size)
    fold_val_gen = memmap_batch_generator(kfold_images, kfold_labels, val_idx, args.batch_size)

    fold_base_model = create_model_3d(sample_shape, len(args.class_names))
    fold_model = SFRModel(
        model=fold_base_model,
        alpha=args.alpha,
        mask_crop=mask_crop,
        last_conv_layer_name=args.last_conv_layer,
    )
    fold_model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")],
    )

    fold_checkpoint = ModelCheckpoint(
        filepath=os.path.join(fold_dir, "model.keras"),
        monitor='val_categorical_accuracy',
        save_best_only=True,
        mode='max',
    )
    fold_early_stop = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=1)
    fold_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
    fold_csv_log = CSVLogger(os.path.join(fold_dir, "log_treino.csv"), append=False)

    history = fold_model.fit(
        fold_train_gen,
        epochs=args.epochs,
        verbose=1,
        validation_data=fold_val_gen,
        steps_per_epoch=len(train_idx) // args.batch_size,
        validation_steps=max(1, len(val_idx) // args.batch_size),
        callbacks=[fold_checkpoint, fold_early_stop, fold_reduce_lr, fold_csv_log],
    )

    with open(os.path.join(fold_dir, "history.json"), "w") as f:
        json.dump(history.history, f, indent=2)

    fold_pred_labels, fold_true_labels, fold_pred = get_predictions_memmap(
        kfold_images, kfold_labels, val_idx, args.batch_size, fold_model
    )
    get_classification_report(fold_true_labels, fold_pred_labels, fold_dir, 'val')
    plot_confusion_matrix(fold_true_labels, fold_pred_labels, fold_dir, 'val', args.class_names)
    save_auc(fold_pred, fold_true_labels, fold_dir, 'val')

    fold_acc = float(np.mean(fold_pred_labels == fold_true_labels))
    fold_auc = float(roc_auc_score(fold_true_labels, fold_pred[:, 1] if fold_pred.shape[1] > 1 else fold_pred))
    fold_result = {"fold": fold_idx, "val_accuracy": fold_acc, "val_auc": fold_auc}
    with open(fold_result_path, "w") as f:
        json.dump(fold_result, f, indent=2)

    fold_report_text = classification_report(fold_true_labels, fold_pred_labels)
    with open(os.path.join(fold_dir, "metrics_report.txt"), "w") as f:
        f.write(f"Fold {fold_idx}/{args.n_folds}\n")
        f.write(f"Accuracy: {fold_acc:.4f}\n")
        f.write(f"AUC: {fold_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(fold_report_text)

    print(f"Fold {fold_idx} concluido: acc={fold_acc:.4f} auc={fold_auc:.4f}")


if __name__ == "__main__":
    main()
