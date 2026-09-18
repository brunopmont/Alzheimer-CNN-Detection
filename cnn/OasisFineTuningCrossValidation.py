#!/usr/bin/env python
# coding: utf-8

# ### FINE-TUNING NO OASIS COM CROSS-VALIDATION (5-FOLD)
#
# Copia de PreAugmentCrossValidationDGX.py adaptada para:
#   - Carregar um modelo ja treinado no ADNI (PRETRAINED_MODEL_PATH)
#   - Fazer fine-tuning (continuar o treino) usando SOMENTE dados do OASIS
#   - Treinar e validar via 5-fold CV dentro do proprio OASIS (sem holdout externo)

import os
import shutil
import sys
import torchio as tio

# pra usar cpu, descomentar linha abaixo
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import math

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision

import gc
import random
import json
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import utils.processamento_dados as proc_dados
import utils.metricas_e_visualizacao as met_vil
import utils.modelos as modelos

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU habilitada com sucesso!")
        print("Memory Growth habilitado para a GPU")
    except RuntimeError as e:
        print(e)

mixed_precision.set_global_policy("mixed_float16")

tf.get_logger().setLevel('ERROR')

# FUNCOES

# Salvar predicoes em CSV
def save_predictions_csv(y_true, y_pred, y_probs, file_paths, output_path, class_names):
    # Converter indices para nomes de classes
    y_true_names = [class_names[idx] for idx in y_true]
    y_pred_names = [class_names[idx] for idx in y_pred]

    # Montar DataFrame
    data = {
        'file_path': file_paths,
        'true_label': y_true_names,
        'pred_label': y_pred_names,
        'correct': y_true == y_pred
    }

    # Adicionar probabilidades por classe
    for i, class_name in enumerate(class_names):
        data[f'prob_{class_name}'] = y_probs[:, i]

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"[OK] Predicoes salvas em: {output_path}")

# Nome das classes (apos binarizacao do OASIS: 0.0 -> cn, 1.0/2.0 -> ad)
class_labels = ['cn', 'ad']

# Definindo caminhos
oasis_dir = "./datasets_codes/OASIS_1_FSL_NORMALIZED/train"  # Dados OASIS (treino + validacao via K-Fold)

# >>> EDITE AQUI: caminho do modelo ja treinado no ADNI que sera usado como base do fine-tuning <<<
PRETRAINED_MODEL_PATH = "./datasets_codes/results_CBM/experimentos/1n_geometry_only/1/fold_1/best_model_fold_1.h5"

# Quantas camadas iniciais do modelo pre-treinado ficam congeladas durante o fine-tuning.
# 0 = fine-tuning completo (todas as camadas treinaveis).
FREEZE_LAYERS = 0

# Validar caminhos
print("\n=== VERIFICACAO DE CAMINHOS ===")
for path_name, path in [('oasis_dir', oasis_dir), ('PRETRAINED_MODEL_PATH', PRETRAINED_MODEL_PATH)]:
    if os.path.exists(path):
        print(f"[OK] {path_name}: {os.path.abspath(path)}")
    else:
        print(f"{path_name} nao encontrado: {os.path.abspath(path)}")
        print(f"   Diretorio atual: {os.getcwd()}")
print("================================\n")

if not os.path.exists(PRETRAINED_MODEL_PATH):
    raise FileNotFoundError(
        f"Modelo pre-treinado nao encontrado em: {os.path.abspath(PRETRAINED_MODEL_PATH)}\n"
        f"Ajuste a variavel PRETRAINED_MODEL_PATH no topo do script."
    )

experiment_title = 'oasis_fine_tuning'
results_dir = f'./datasets_codes/results_CBM/experimentos/{experiment_title}'
os.makedirs(results_dir, exist_ok=True)

results_dir = f'./datasets_codes/results_CBM/experimentos/{experiment_title}/{len(os.listdir(results_dir)) + 1}'
os.makedirs(results_dir, exist_ok=True)
print(f"pasta {results_dir} criada")

# Criar diretorio para salvar predicoes
predictions_dir = os.path.join(results_dir, 'predictions')
os.makedirs(predictions_dir, exist_ok=True)

# Guarda referencia ao modelo base usado, para rastreabilidade
with open(os.path.join(results_dir, "modelo_base_usado.txt"), "w") as f:
    f.write(os.path.abspath(PRETRAINED_MODEL_PATH))

# # Fine-Tuning

# 1. CONFIGURACAO DOS EXPERIMENTOS

# Augmentation aplicado nos folds de treino durante o fine-tuning
EXP_AUGMENTATION_FACTOR = 0 # Mantem so originais
EXP_INCLUDE_INTENSITY = False # Transformacoes de geometria + intensidade
EXP_TRANSFORM_PROB = 0 # Nunca transforma (sem efeito aqui, pois factor=0 ja nao gera copias)

print(f"--- CONFIGURACAO EXPERIMENTO (FINE-TUNING OASIS, BALANCED CV) ---")
print(f"Modelo base: {PRETRAINED_MODEL_PATH}")
print(f"Camadas congeladas: {FREEZE_LAYERS}")
print(f"Fator de Aumento Alvo: {EXP_AUGMENTATION_FACTOR}x (sobre a classe majoritaria)")
print(f"Incluir Intensidade: {EXP_INCLUDE_INTENSITY}")
print(f"Probabilidade Transform: {EXP_TRANSFORM_PROB}")
print("----------------------------------------------")

# CARREGAMENTO DOS DADOS DO OASIS
# O OASIS vem organizado em pastas '0.0', '1.0', '2.0' (CDR). Binarizamos:
# 0.0 -> cn (classe 0) | 1.0 e 2.0 -> ad (classe 1)
print("Carregando dados do OASIS para memoria...")
oasis_data = proc_dados.load_nifti_data_balanced_preallocated(
    oasis_dir, ['0.0', '1.0', '2.0'], augment=False, target_per_class=1000
)

if oasis_data is None:
    raise ValueError(f"Nao foi possivel carregar os dados em: {os.path.abspath(oasis_dir)}")

X_oasis_raw, y_oasis_raw, paths_raw, _ = oasis_data

oasis_raw_indices = np.argmax(y_oasis_raw, axis=1)
oasis_indices_bin = np.where(oasis_raw_indices == 0, 0, 1)
X_full_raw = X_oasis_raw
y_full_indices = to_categorical(oasis_indices_bin, num_classes=len(class_labels)).astype(np.float16)

print(f"Total de dados carregados: {len(X_full_raw)}")
print(f"Distribuicao das classes (0=cn, 1=ad): {np.bincount(oasis_indices_bin)}")

# GERADOR DE DADOS
def nifti_data_generator_3d(images_array, labels, batch_size):
    total_n = len(images_array)
    while True:
        indices = np.random.permutation(total_n)
        for i in range(0, total_n, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_images = np.array(images_array[batch_idx])
            batch_labels = np.array(labels[batch_idx])
            yield batch_images.astype(np.float16), batch_labels.astype(np.float16)


# ==============================================================================
# CACHE DE FOLDS EM DISCO (original + aumentado)
# ==============================================================================
# Em vez de aumentar em tempo real a cada epoca (caro), cada fold e aumentado
# 1x em disco antes do treino e reaproveitado nas epocas. cache/original guarda
# os 5 folds crus (cada um usado como validacao em algum momento); cache/aumento
# guarda a versao aumentada de cada fold (usada como treino quando o fold
# correspondente NAO e o de validacao da vez). Tudo dentro de cache_dir e
# apagado ao final da execucao; oasis_dir nunca e tocado.

def save_fold_npy(X, y, fold_dir):
    os.makedirs(fold_dir, exist_ok=True)
    for i in range(len(X)):
        np.save(os.path.join(fold_dir, f"vol_{i:04d}.npy"), X[i].astype(np.float16))
    np.save(os.path.join(fold_dir, "labels.npy"), y.astype(np.float16))

def load_fold_npy(fold_dir):
    vol_files = sorted(f for f in os.listdir(fold_dir) if f.startswith("vol_"))
    X = np.stack([np.load(os.path.join(fold_dir, f)) for f in vol_files])
    y = np.load(os.path.join(fold_dir, "labels.npy"))
    return X, y

def generate_augmented_fold(original_fold_dir, aug_fold_dir, augment_factor, include_intensity, transform_prob):
    X, y = load_fold_npy(original_fold_dir)
    composer = proc_dados.get_augmentation_pipeline(include_intensity) if augment_factor > 0 else None

    os.makedirs(aug_fold_dir, exist_ok=True)
    out_labels = []
    out_idx = 0
    for i in range(len(X)):
        img = X[i]
        label = y[i]

        # sempre inclui a copia original (1n)
        np.save(os.path.join(aug_fold_dir, f"vol_{out_idx:04d}.npy"), img.astype(np.float16))
        out_labels.append(label)
        out_idx += 1

        for _ in range(int(augment_factor)):
            if composer is not None and random.random() < transform_prob:
                img_t = img.transpose(3, 0, 1, 2)
                subject = tio.Subject(mri=tio.ScalarImage(tensor=img_t))
                transformed = composer(subject)
                img_aug = transformed.mri.data.numpy().transpose(1, 2, 3, 0)
            else:
                img_aug = img

            np.save(os.path.join(aug_fold_dir, f"vol_{out_idx:04d}.npy"), img_aug.astype(np.float16))
            out_labels.append(label)
            out_idx += 1

    np.save(os.path.join(aug_fold_dir, "labels.npy"), np.array(out_labels, dtype=np.float16))

# ==============================================================================
# 4. LOOP 5-FOLD CROSS VALIDATION (FINE-TUNING)
# ==============================================================================
K_FOLDS = 5

# Seed nova a cada execucao -> folds diferentes a cada uma das rodagens
seed_execucao = int.from_bytes(os.urandom(4), 'little')
random.seed(seed_execucao)
np.random.seed(seed_execucao)
with open(os.path.join(results_dir, "seed_usada.txt"), "w") as f:
    f.write(str(seed_execucao))
print(f"Seed desta execucao: {seed_execucao}")

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed_execucao)

fold_accuracies = []
fold_aucs = []
aggregated_true_labels = []
aggregated_pred_labels = []
fold_f1_scores = []

melhor_acuracia_global = 0.0
melhor_fold_global = -1
caminho_melhor_modelo_global = ""

BATCH_SIZE = 8
EPOCHS = 125
FINE_TUNING_LEARNING_RATE = 0.00005  # LR baixo para nao destruir os pesos pre-treinados

sample_shape = X_full_raw[0].shape
n_classes_fine_tuning = len(class_labels)

y_stratify = np.argmax(y_full_indices, axis=1)

# --- Materializa os 5 folds (original cru + aumentado) uma unica vez ---
cache_dir = os.path.join(results_dir, "cache")
original_dir = os.path.join(cache_dir, "original")
aumento_dir = os.path.join(cache_dir, "aumento")

fold_val_indices = {}  # fold_num -> indices em X_full_raw (para paths/CSV depois)

print("\nGerando cache de folds (original + aumento) em disco...")
for fold_num, (train_index, val_index) in enumerate(skf.split(X_full_raw, y_stratify), start=1):
    fold_val_indices[fold_num] = val_index

    original_fold_dir = os.path.join(original_dir, f"fold_{fold_num}")
    save_fold_npy(X_full_raw[val_index], y_full_indices[val_index], original_fold_dir)

    aug_fold_dir = os.path.join(aumento_dir, f"fold_{fold_num}")
    generate_augmented_fold(
        original_fold_dir, aug_fold_dir,
        augment_factor=EXP_AUGMENTATION_FACTOR,
        include_intensity=EXP_INCLUDE_INTENSITY,
        transform_prob=EXP_TRANSFORM_PROB
    )
    print(f"  fold_{fold_num}: original em {original_fold_dir} | aumento em {aug_fold_dir}")
print("Cache de folds pronto.\n")

for fold_idx in range(K_FOLDS):
    fold_num = fold_idx + 1
    print(f"\n{'='*40}\nINICIANDO FOLD {fold_num}/{K_FOLDS} (FINE-TUNING)\n{'='*40}")

    fold_dir = os.path.join(results_dir, f"fold_{fold_num}")
    os.makedirs(fold_dir, exist_ok=True)

    # Validacao: fold original cru da vez
    X_val_fold, y_val_fold = load_fold_npy(os.path.join(original_dir, f"fold_{fold_num}"))
    val_index = fold_val_indices[fold_num]

    # Treino: uniao dos outros 4 folds ja aumentados
    train_X_parts, train_y_parts = [], []
    for other_fold_num in range(1, K_FOLDS + 1):
        if other_fold_num == fold_num:
            continue
        X_part, y_part = load_fold_npy(os.path.join(aumento_dir, f"fold_{other_fold_num}"))
        train_X_parts.append(X_part)
        train_y_parts.append(y_part)
    X_train_fold = np.concatenate(train_X_parts, axis=0)
    y_train_fold = np.concatenate(train_y_parts, axis=0)

    steps_per_epoch = math.ceil(len(X_train_fold) / BATCH_SIZE)

    # Geradores (so shuffle + batch; o aumento ja veio pronto do disco)
    raw_train_gen = nifti_data_generator_3d(X_train_fold, y_train_fold, BATCH_SIZE)
    raw_val_gen = nifti_data_generator_3d(X_val_fold, y_val_fold, BATCH_SIZE)

    train_generator = tf.data.Dataset.from_generator(
        lambda: raw_train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *sample_shape), dtype=tf.float16),
            tf.TensorSpec(shape=(None, n_classes_fine_tuning), dtype=tf.float16)
        )
    ).prefetch(tf.data.AUTOTUNE)

    val_generator = tf.data.Dataset.from_generator(
        lambda: raw_val_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *sample_shape), dtype=tf.float16),
            tf.TensorSpec(shape=(None, n_classes_fine_tuning), dtype=tf.float16)
        )
    ).prefetch(tf.data.AUTOTUNE)

    val_steps = math.ceil(len(X_val_fold) / BATCH_SIZE)

    # --- Modelo: carrega o modelo pre-treinado no ADNI a cada fold ---
    K.clear_session()
    gc.collect()

    model = load_model(PRETRAINED_MODEL_PATH)

    if FREEZE_LAYERS > 0:
        print(f"Congelando as {FREEZE_LAYERS} primeiras camadas do modelo pre-treinado...")
        for layer in model.layers[:FREEZE_LAYERS]:
            layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNING_LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)])

    # --- Fine-tuning ---
    checkpoint_path_fold = os.path.join(fold_dir, f"best_model_fold_{fold_num}.h5")
    fold_last_path = os.path.join(fold_dir, f"last_model_{fold_num}.h5")

    callbacks_list = [
        # EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        CSVLogger(os.path.join(fold_dir, 'training_log.csv')),
        ModelCheckpoint(filepath=checkpoint_path_fold, monitor='val_loss',
                        save_best_only=True, mode='min'),
        ModelCheckpoint(fold_last_path, save_best_only=False)
    ]

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=callbacks_list
    )

    # --- Avaliacao ---
    met_vil.plot_training_history(history, fold_dir)

    del history
    del train_generator, val_generator
    for _ in range(3):
        gc.collect()

    model.load_weights(checkpoint_path_fold)

    val_pred_probs = model.predict(X_val_fold, batch_size=4)
    val_pred_labels = np.argmax(val_pred_probs, axis=1)
    val_true_labels = np.argmax(y_val_fold, axis=1)

    acc = accuracy_score(val_true_labels, val_pred_labels)
    auc_score = roc_auc_score(val_true_labels, val_pred_probs[:, 1])
    f1 = f1_score(val_true_labels, val_pred_labels, average='macro')

    fold_accuracies.append(acc)
    fold_aucs.append(auc_score)
    fold_f1_scores.append(f1)

    val_pos_probs = val_pred_probs[:, 1]

    met_vil.get_classification_report(val_true_labels, val_pred_labels, fold_dir, f'report_fold_{fold_num}')
    met_vil.plot_confusion_matrix(val_true_labels, val_pred_labels, fold_dir, f'cm_fold_{fold_num}', class_labels)
    met_vil.save_auc(val_pos_probs, val_true_labels, fold_dir, 'test')
    met_vil.plot_roc_curve(val_true_labels, val_pos_probs, fold_dir, 'validation', title='')

    # SALVAR PREDICOES DO FOLD EM CSV
    fold_paths = [paths_raw[i] for i in val_index]
    fold_predictions_csv = os.path.join(predictions_dir, f'fold_{fold_num}_predictions.csv')
    save_predictions_csv(val_true_labels, val_pred_labels, val_pred_probs, fold_paths, fold_predictions_csv, class_labels)

    aggregated_true_labels.extend(val_true_labels)
    aggregated_pred_labels.extend(val_pred_labels)

    print(f"Resultados Fold {fold_num} -> Acuracia: {acc:.4f} | AUC: {auc_score:.4f} | F1-Score: {f1:.4f}")

    if acc > melhor_acuracia_global:
        melhor_acuracia_global = acc
        melhor_fold_global = fold_num
        caminho_melhor_modelo_global = checkpoint_path_fold

    # Limpeza agressiva ao final de cada fold
    del model, X_train_fold, X_val_fold, y_train_fold, y_val_fold
    del val_pred_probs, val_pred_labels, val_true_labels, val_pos_probs

    # Limpa sessao do Keras
    K.clear_session()

    # Forca coleta de lixo em multiplas geracoes
    for _ in range(3):
        gc.collect()

    print(f"Fold {fold_num} finalizado. Memoria liberada.")

print(f"\n{'='*40}")
print(f"FINE-TUNING CROSS-VALIDATION CONCLUIDO!")
print(f"O melhor modelo foi do Fold {melhor_fold_global} com Acuracia {melhor_acuracia_global:.4f}")
print(f"{'='*40}\n")

# Apaga o cache de folds (original + aumento) ao final da execucao.
# oasis_dir (dados brutos originais) nunca e tocado aqui.
print(f"Removendo cache de folds em {cache_dir}...")
shutil.rmtree(cache_dir, ignore_errors=True)
print("Cache removido.\n")

# ==============================================================================
# 5. RESULTADOS GLOBAIS (fora do loop, depois que todos os folds finalizaram)
# ==============================================================================
print("\n=================================================")
print("--- RESULTADOS FINAIS DO FINE-TUNING CROSS VALIDATION (OASIS) ---")
print("=================================================")

mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

mean_auc = np.mean(fold_aucs)
std_auc = np.std(fold_aucs)

mean_f1 = np.mean(fold_f1_scores)
std_f1 = np.std(fold_f1_scores)

print(f"Acuracia Media: {mean_acc:.4f} +/- {std_acc:.4f}")
print(f"AUC Media:      {mean_auc:.4f} +/- {std_auc:.4f}")
print(f"F1-Score Medio: {mean_f1:.4f} +/- {std_f1:.4f}")

# Opcional: Salvar esse resumo em um arquivo de texto
summary_path = os.path.join(results_dir, "cv_metrics_summary.txt")
with open(summary_path, "w") as f:
    f.write("Resultados do Fine-Tuning por Cross-Validation (OASIS)\n")
    f.write("-------------------------------------------------------\n")
    f.write(f"Modelo base: {os.path.abspath(PRETRAINED_MODEL_PATH)}\n")
    f.write(f"Acuracia: {mean_acc:.4f} +/- {std_acc:.4f}\n")
    f.write(f"AUC:      {mean_auc:.4f} +/- {std_auc:.4f}\n")
    f.write(f"F1-Score: {mean_f1:.4f} +/- {std_f1:.4f}\n")
    f.write("\nValores Individuais por Fold:\n")
    f.write(f"Acuracias: {fold_accuracies}\n")
    f.write(f"AUCs:      {fold_aucs}\n")

aggregated_true = np.array(aggregated_true_labels)
aggregated_pred = np.array(aggregated_pred_labels)

met_vil.plot_confusion_matrix(
    aggregated_true,
    aggregated_pred,
    results_dir,
    'FINAL_CV_CONFUSION_MATRIX',
    class_labels
)

met_vil.get_classification_report(
    aggregated_true,
    aggregated_pred,
    results_dir,
    'FINAL_CV_REPORT'
)

# ==============================================================================
# 6. RESUMO FINAL DE PREDICOES
# ==============================================================================
print(f"\n{'='*50}")
print("SALVANDO RESUMO FINAL DE PREDICOES...")
print(f"{'='*50}\n")

predictions_summary = {
    'experimento': experiment_title,
    'modelo_base': os.path.abspath(PRETRAINED_MODEL_PATH),
    'data_criacao': str(pd.Timestamp.now()),
    'fine_tuning_cross_validation_oasis': {
        'total_folds': K_FOLDS,
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'mean_auc': float(mean_auc),
        'std_auc': float(std_auc),
        'mean_f1': float(mean_f1),
        'std_f1': float(std_f1),
        'melhor_fold': int(melhor_fold_global),
        'melhor_acuracia': float(melhor_acuracia_global),
        'arquivos': f'fold_1_predictions.csv a fold_{K_FOLDS}_predictions.csv'
    },
    'diretorio_predicoes': predictions_dir
}

summary_json_path = os.path.join(predictions_dir, 'predictions_summary.json')
with open(summary_json_path, 'w') as f:
    json.dump(predictions_summary, f, indent=2)

print(f"Resumo de predicoes salvo em: {summary_json_path}")
print(f"\nTodas as predicoes foram salvas em CSV:")
print(f"  - {K_FOLDS} arquivos de CV (fold_1_predictions.csv ... fold_{K_FOLDS}_predictions.csv)")
print(f"  - predictions_summary.json")
print(f"\n  Localizacao: {predictions_dir}")
print(f"\n{'='*50}")
print("PROCESSAMENTO COMPLETO!")
print(f"{'='*50}")
