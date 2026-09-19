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
o mesmo padrao: escreve o pool train+val num .npy pre-alocado em disco (um
volume por vez, sem nunca ter o dataset inteiro em RAM) e roda o loop de folds
chamando run_fold_sfr.py como subprocesso, que le esse .npy mapeado por lote.

Resume: passe --results-dir apontando para a pasta test_N de uma execucao
anterior. O cache do pool e os folds ja concluidos sao reaproveitados.
"""
import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np
import nibabel as nib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== Caminhos: ajuste para o ambiente onde este script vai rodar =====
dir_base = "/mnt/c/Users/Paulo Pires/Desktop/Alzheimer_cnn/ADNI/ADNI_NORMALIZED"
train_dir = f'{dir_base}/train'
val_dir = f'{dir_base}/validation'
results_root = f'{dir_base}/results/neurips_results/grad_loss_mask_pre'

# resolvido a partir do proprio script, nao do cwd -- assim funciona rodando de
# qualquer diretorio, e o subprocesso recebe um caminho absoluto
mask_nii_path = os.path.join(
    SCRIPT_DIR, "..", "pre_processing", "mni_icbm152_nlin_asym_09c_nifti",
    "mni_icbm152_nlin_asym_09c", "mni_icbm152_t1_tal_nlin_asym_09c_mask.nii",
)
mask_nii_path = os.path.abspath(mask_nii_path)

class_names = ['cn', 'ad']
n_classes = len(class_names)

N_FOLDS = 5
FOLD_ALPHA = 0.00
FOLD_EPOCHS = 200
FOLD_PATIENCE = 30
batch_size = 16
SHUFFLE_SEED = 42


def build_kfold_cache(base_dirs, images_path, labels_path, seed=SHUFFLE_SEED):
    """Escreve o pool train+val direto num .npy pre-alocado em disco.

    Carrega um volume por vez para dentro do memmap, entao o pico de RAM e o de
    UM volume -- e nao o de 2x o dataset inteiro, que era o que acontecia com o
    load_nifti_data_balanced do notebook (lista de volumes -> np.array (2a
    copia) -> shuffle por fancy indexing (3a copia)) seguido do np.concatenate
    de treino+validacao. O embaralhamento e feito na LISTA de arquivos, antes de
    carregar, entao nao existe copia embaralhada do array.
    """
    entries = []
    for base_dir in base_dirs:
        for class_idx, label in enumerate(class_names):
            label_dir = os.path.join(base_dir, label)
            for fname in sorted(os.listdir(label_dir)):
                entries.append((os.path.join(label_dir, fname), class_idx))

    random.Random(seed).shuffle(entries)

    total = len(entries)
    if total == 0:
        raise ValueError(f"Nenhum volume encontrado em {base_dirs}")

    shape = nib.load(entries[0][0]).header.get_data_shape()
    print(f"Pool k-fold: {total} volumes, shape {shape}")

    images = np.lib.format.open_memmap(
        images_path, mode='w+', dtype=np.float16, shape=(total, *shape, 1)
    )
    labels = np.zeros((total, n_classes), dtype=np.float16)

    for i, (path, class_idx) in enumerate(entries):
        images[i, ..., 0] = nib.load(path).get_fdata(dtype=np.float32).astype(np.float16)
        labels[i, class_idx] = 1.0
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total} volumes escritos")

    images.flush()
    del images
    np.save(labels_path, labels)
    print(f"Cache do pool escrito em {images_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", default=None,
        help="Pasta test_N de uma execucao anterior, para retomar de onde parou. "
             "Sem isso, uma pasta nova e criada.",
    )
    args = parser.parse_args()

    os.makedirs(results_root, exist_ok=True)

    if args.results_dir:
        run_dir = args.results_dir
        if not os.path.isdir(run_dir):
            raise SystemExit(f"--results-dir nao existe: {run_dir}")
        print(f"Retomando execucao em {run_dir}")
    else:
        n = len(os.listdir(results_root))
        run_dir = os.path.join(results_root, f"test_{n + 1}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"pasta {os.path.basename(run_dir)} criada")

    images_path = os.path.join(run_dir, "_kfold_images.npy")
    labels_path = os.path.join(run_dir, "_kfold_labels.npy")

    if os.path.exists(images_path) and os.path.exists(labels_path):
        print("Cache do pool k-fold ja existe -- reaproveitando.")
    else:
        print("Gerando cache do pool k-fold (train + validation) em disco...")
        build_kfold_cache([train_dir, val_dir], images_path, labels_path)

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
        with open(os.path.join(run_dir, "kfold_summary.txt"), "w") as f:
            f.write(summary)

    for fold_idx in range(1, N_FOLDS + 1):
        print(f"\n===== FOLD {fold_idx}/{N_FOLDS} =====")

        fold_dir = os.path.join(run_dir, f"kfold_{fold_idx}")
        fold_result_path = os.path.join(fold_dir, "fold_result.json")

        if os.path.exists(fold_result_path):
            with open(fold_result_path) as f:
                fold_metrics.append(json.load(f))
            print(f"Fold {fold_idx} ja possui resultado salvo -- pulando retreino.")
            write_kfold_summary()
            continue

        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "run_fold_sfr.py"),
            "--fold-idx", str(fold_idx),
            "--n-folds", str(N_FOLDS),
            "--results-dir", run_dir,
            "--kfold-images", images_path,
            "--kfold-labels", labels_path,
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
    print(f"\nO cache do pool ({images_path}) pode ser apagado se nao for retomar esta execucao.")


if __name__ == "__main__":
    main()
