#!/usr/bin/env python
# coding: utf-8
"""
Driver do k-fold do SFR_training, para rodar via terminal (`python run_sfr_kfold.py`),
NAO pelo Jupyter.

Motivo: mesmo isolando cada fold num subprocesso (run_fold_sfr.py), o kernel do
Jupyter continuava morrendo -- e sem nenhum output novo do subprocesso, ou seja,
o processo que estava sem RAM era o proprio kernel do notebook, nao o subprocesso
de treino. Jupyter mantem vivo o historico de saida de todas as celulas (Out[],
cache de figuras do backend inline do matplotlib, arrays de celulas anteriores
que nunca saem do namespace), entao o baseline de RAM do kernel so cresce ao
longo da sessao e nunca e devolvido ao SO.

O PreAugmentCrossValidationDGX.py roda esse mesmo tipo de carga (dados 3D,
K-fold, TF/Keras) como script .py puro sem esse problema -- este driver segue
o mesmo padrao: carrega os dados uma vez, participa do loop de folds chamando
run_fold_sfr.py como subprocesso (assim cada fold ainda ganha um processo
isolado pro treino em si), e nao acumula nenhum estado de notebook.
"""
import gc
import json
import os
import subprocess
import sys

import numpy as np
import nibabel as nib
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# ===== Caminhos: ajuste para o ambiente onde este script vai rodar =====
dir_base = "/mnt/c/Users/Paulo Pires/Desktop/Alzheimer_cnn/ADNI/ADNI_NORMALIZED"
train_dir = f'{dir_base}/train'
val_dir = f'{dir_base}/validation'
results_dir = f'{dir_base}/results/neurips_results/grad_loss_mask_pre'
mask_nii_path = "../pre_processing/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii"

class_names = ['cn', 'ad']
n_classes = len(class_names)

N_FOLDS = 5
FOLD_ALPHA = 0.00
FOLD_EPOCHS = 200
FOLD_PATIENCE = 30
batch_size = 16


def load_nifti_data_balanced(base_dir, class_names):
    images, labels, paths = [], [], []
    for label in class_names:
        print(f"carregando diretorio {label}")
        label_dir = os.path.join(base_dir, label)
        count = 0
        for fname in os.listdir(label_dir):
            img_path = os.path.join(label_dir, fname)
            img = nib.load(img_path).get_fdata(dtype=np.float16)
            paths.append(img_path)
            images.append(img)
            labels.append(label)
            count += 1
        print(f"diretorio carregado {count}")

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(class_names)
    labels_encoded = label_encoder.transform(labels)
    labels_one_hot = to_categorical(labels_encoded, num_classes=len(class_names))

    images = np.array(images).reshape((-1, *images[0].shape, 1))
    labels_one_hot = np.array(labels_one_hot)

    images, labels_one_hot, paths = shuffle(images, labels_one_hot, paths, random_state=42)
    return images, labels_one_hot, paths, label_encoder.classes_


def main():
    os.makedirs(results_dir, exist_ok=True)

    n = len(os.listdir(results_dir))
    folder_name = f"test_{n + 1}"
    fold_results_dir = os.path.join(results_dir, folder_name)
    os.makedirs(fold_results_dir, exist_ok=True)
    print(f"pasta {folder_name} criada")

    print("Carregando treino...")
    train_images, train_labels, train_paths, _ = load_nifti_data_balanced(train_dir, class_names)
    print(f"N treino: {len(train_paths)}")

    print("Carregando validacao...")
    val_images, val_labels, val_paths, _ = load_nifti_data_balanced(val_dir, class_names)
    print(f"N validation: {len(val_paths)}")

    kfold_images = np.concatenate([train_images, val_images], axis=0)
    kfold_labels = np.concatenate([train_labels, val_labels], axis=0)
    del train_images, val_images, train_labels, val_labels
    gc.collect()

    kfold_images_path = os.path.join(fold_results_dir, "_kfold_images.npy")
    kfold_labels_path = os.path.join(fold_results_dir, "_kfold_labels.npy")
    np.save(kfold_images_path, kfold_images)
    np.save(kfold_labels_path, kfold_labels)
    del kfold_images, kfold_labels
    gc.collect()

    fold_metrics = []

    def write_kfold_summary():
        accs = [m["val_accuracy"] for m in fold_metrics]
        aucs = [m["val_auc"] for m in fold_metrics]
        summary = (
            f"Folds concluidos: {len(fold_metrics)}/{N_FOLDS}\n"
            f"Accuracy por fold: {[round(a, 4) for a in accs]}\n"
            f"Accuracy media: {np.mean(accs):.4f} +/- {np.std(accs):.4f}\n"
            f"AUC por fold: {[round(a, 4) for a in aucs]}\n"
            f"AUC media: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}\n"
        )
        print(summary)
        with open(os.path.join(fold_results_dir, "kfold_summary.txt"), "w") as f:
            f.write(summary)

    for fold_idx in range(1, N_FOLDS + 1):
        print(f"\n===== FOLD {fold_idx}/{N_FOLDS} =====")

        fold_dir = os.path.join(fold_results_dir, f"kfold_{fold_idx}")
        fold_result_path = os.path.join(fold_dir, "fold_result.json")

        if os.path.exists(fold_result_path):
            with open(fold_result_path) as f:
                fold_metrics.append(json.load(f))
            print(f"Fold {fold_idx} ja possui resultado salvo -- pulando retreino.")
            write_kfold_summary()
            continue

        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "run_fold_sfr.py"),
            "--fold-idx", str(fold_idx),
            "--n-folds", str(N_FOLDS),
            "--results-dir", fold_results_dir,
            "--kfold-images", kfold_images_path,
            "--kfold-labels", kfold_labels_path,
            "--mask-path", mask_nii_path,
            "--alpha", str(FOLD_ALPHA),
            "--epochs", str(FOLD_EPOCHS),
            "--patience", str(FOLD_PATIENCE),
            "--batch-size", str(batch_size),
            "--last-conv-layer", "conv3d_2",
            "--class-names", *class_names,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"run_fold_sfr.py falhou no fold {fold_idx} (returncode={result.returncode})")

        with open(fold_result_path) as f:
            fold_metrics.append(json.load(f))
        write_kfold_summary()

    print("\n===== K-Fold concluido =====")
    write_kfold_summary()


if __name__ == "__main__":
    main()
