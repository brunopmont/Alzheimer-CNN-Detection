import os
import torchio as tio

import numpy as np
import math
import nibabel as nib

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, models, Input, Model

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import gc
import seaborn as sns

from scipy import ndimage
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import tempfile
from math import ceil
import random
from tqdm import tqdm
from tensorflow.keras import backend as K

# ----------------------------------------------------------------------------------------

# --- FUNÇÕES DE CARREGAMENTO E AUGMENTATION (Devem vir ANTES do fluxo principal) ---

def load_nifti_data(data_dir):
    X = []
    y = []
    classes = {'cn': 0, 'ad': 1}
    
    print(f"Carregando dados de: {data_dir}")
    
    for label_name, label_idx in classes.items():
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.exists(folder_path):
            print(f"Aviso: Pasta {folder_path} não encontrada.")
            continue
            
        files = [f for f in os.listdir(folder_path) if f.endswith(('.nii', '.nii.gz'))]
        
        # tqdm aqui ajuda a ver o progresso do carregamento
        for file_name in tqdm(files, desc=f"Lendo {label_name}"):
            file_path = os.path.join(folder_path, file_name)
            try:
                img = nib.load(file_path)
                data = img.get_fdata().astype(np.float32) # Força float32 para economizar RAM
                
                X.append(data)
                y.append(label_idx)
            except Exception as e:
                print(f"Erro ao ler {file_name}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    if X.ndim == 4: 
        X = np.expand_dims(X, axis=-1)
        
    print(f"Dados carregados. Shape: {X.shape}")
    return X, y

def augment_zoom(volume):
    zoom_percentage = np.random.uniform(0.0, 0.2)
    direction = np.random.choice([-1, 1])
    factor = 1.0 + (direction * zoom_percentage)
    
    if volume.ndim == 4:
        zoomed = ndimage.zoom(volume[:,:,:,0], zoom=factor, order=1)
        zoomed = np.expand_dims(zoomed, axis=-1)
    else:
        zoomed = ndimage.zoom(volume, zoom=factor, order=1)
    
    target_shape = volume.shape
    current_shape = zoomed.shape
    
    processed = np.zeros(target_shape, dtype=volume.dtype)
    
    dx = (current_shape[0] - target_shape[0]) // 2
    dy = (current_shape[1] - target_shape[1]) // 2
    dz = (current_shape[2] - target_shape[2]) // 2
    
    if factor > 1.0: # Crop
        processed = zoomed[dx:dx+target_shape[0], dy:dy+target_shape[1], dz:dz+target_shape[2]]
    else: # Pad
        processed[-dx:-dx+current_shape[0], -dy:-dy+current_shape[1], -dz:-dz+current_shape[2]] = zoomed
        
    return processed

def augment_shift(volume):
    shifts = []
    for dim in range(3):
        limit = 0.4 * volume.shape[dim]
        shift_val = np.random.uniform(-limit, limit)
        shifts.append(shift_val)
    
    if volume.ndim == 4:
        shifts.append(0)
        
    shifted = ndimage.shift(volume, shift=shifts, order=1, mode='constant', cval=0.0)
    return shifted

def augment_rotation(volume):
    angle_xy = np.random.uniform(-5, 5)
    angle_xz = np.random.uniform(-5, 5)
    angle_yz = np.random.uniform(-5, 5)
    
    vol_aug = volume
    has_channel = False
    if vol_aug.ndim == 4:
        vol_aug = vol_aug[:,:,:,0]
        has_channel = True
        
    # reshape=False é crucial para manter performance e memória
    vol_aug = ndimage.rotate(vol_aug, angle_xy, axes=(0, 1), reshape=False, order=1, mode='constant', cval=0.0)
    vol_aug = ndimage.rotate(vol_aug, angle_xz, axes=(0, 2), reshape=False, order=1, mode='constant', cval=0.0)
    vol_aug = ndimage.rotate(vol_aug, angle_yz, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0.0)
    
    if has_channel:
        vol_aug = np.expand_dims(vol_aug, axis=-1)
        
    return vol_aug

# ----------------------------------------------------------------------------------------

# Função para carregar os caminhos e labels de uma pasta, iterando pelas subpastas cujos nomes estão em class_names
def load_nifti_paths(base_dir, class_names):
    image_paths = []
    labels = []
    
    # Caminhos das subpastas
    for label in class_names:
        label_dir = os.path.join(base_dir, label)
        for fname in os.listdir(label_dir):
            img_path = os.path.join(label_dir, fname)
            image_paths.append(img_path)
            labels.append(label)

    # Codificando os rótulos
    label_encoder = LabelEncoder()

    # Inverter a ordem das classes explicitamente
    label_encoder.classes_ = np.array(class_names)

    # Codificando os rótulos (agora 'cn' será 0 e 'ad' será 1)
    labels_array = np.array(labels)
    labels_encoded = label_encoder.transform(labels_array)

    # Transformando os rótulos para one-hot encoding
    labels_one_hot = to_categorical(labels_encoded, num_classes=len(class_names))

    # Embaralhar os dados
    image_paths, labels_one_hot = shuffle(image_paths, labels_one_hot, random_state=42)

    return image_paths, labels_one_hot, label_encoder.classes_

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Função para carregar dados de forma otimizada (em memória, mas não tanto em tempo), podendo definir o número máximo de dados a serem carregados por classe.
def load_nifti_data_balanced_preallocated(
    base_dir, 
    class_names, 
    augment=False, 
    target_per_class=1000, 
    augmentation_factor=1,       # Controle de quantidade (1x, 2x, 3x, etc.)
    include_intensity=True,      # Controle de Tipo (Geo vs Geo+Intensidade)
    transform_probability=1.0    # Controle de Mix (1.0 = substitui tudo por modificações, <1.0 = mistura original + modificação)
):
    # Garante fator mínimo de 1
    if augmentation_factor < 1:
        augmentation_factor = 1

    transform_composer = None
    if augment:        
        # 1. Transformações Espaciais (Sempre presentes se augment=True)
        spatial_transforms = tio.Compose([
            tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=10,
                isotropic=True,
                p=0.75
            ),
            tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=7.5,
                locked_borders=2,
                p=0.15
            )
        ])

        # 2. Montagem do Compositor Baseado na Flag de Intensidade
        if include_intensity:
            # Se True, adiciona artefatos físicos
            artifact_transforms = tio.OneOf({
                tio.RandomBiasField(): 0.5,
                tio.RandomGhosting(): 0.2,
                tio.RandomMotion(degrees=5, translation=5): 0.2,
                tio.RandomNoise(std=0.05): 0.2,
                tio.RandomBlur(std=(0, 1)): 0.1,
            }, p=0.5)

            transform_composer = tio.Compose([
                spatial_transforms,
                artifact_transforms,
            ])
        else:
            # Se False, usa APENAS geometria
            transform_composer = spatial_transforms

    all_paths = []
    all_labels = []
    all_transform_flags = [] 

    print("Passo 1: Coletando arquivos e definindo estratégia...")
    for label in class_names:
        label_dir = os.path.join(base_dir, label)
        count = 0
        names = os.listdir(label_dir)
        
        for fname in names:
            if count < target_per_class:
                full_path = os.path.join(label_dir, fname)
                
                flags_to_add = []
                
                if not augment:
                    # Sem augment: 1 cópia original sempre
                    flags_to_add.append(False)
                else:
                    # Gera 'augmentation_factor' entradas para esta imagem
                    for _ in range(augmentation_factor):
                        # Decide individualmente se essa cópia será transformada
                        # Se transform_probability for 1.0 (padrão), SEMPRE transforma (descarta original)
                        # Se for diferente de 1.0, tem X% de chance de manter original (criando um mix)
                        if random.random() < transform_probability:
                            flags_to_add.append(True)
                        else:
                            flags_to_add.append(False)
                
                # Preenche as listas
                for should_transform in flags_to_add:
                    all_paths.append(full_path)
                    all_transform_flags.append(should_transform)
                    
                    if label in ['cn', '0.0', '0', 0]:
                        all_labels.append(0)
                    else:
                        all_labels.append(1)
                
                count += 1
                
    print(f"Total de {len(all_paths)} imagens planejadas.")

    # Codificação de labels
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array([0, 1])
    labels_encoded = label_encoder.transform(all_labels)
    labels_one_hot = to_categorical(labels_encoded, num_classes=2).astype(np.float16) 
    
    # Shuffle com as flags
    all_paths_shuffled, labels_one_hot_shuffled, flags_shuffled = shuffle(
        all_paths, labels_one_hot, all_transform_flags, random_state=42
    )
    
    del all_labels, labels_encoded, labels_one_hot, all_transform_flags, all_paths
    gc.collect()

    if not all_paths_shuffled:
        print("Nenhuma imagem encontrada.")
        return np.array([]), np.array([]), [], label_encoder.classes_

    print("Passo 2: Detectando shape e Alocando Memória...")
    try:
        first_img_nib = nib.load(all_paths_shuffled[0])
        img_shape = first_img_nib.get_fdata(dtype=np.float16).shape
    except Exception as e:
        print(f"Erro: {e}")
        return

    total_images = len(all_paths_shuffled)
    images_final = np.empty((total_images, *img_shape, 1), dtype=np.float16)
    paths_final = [None] * total_images

    print("Passo 3: Carregando e Processando...")
    for i in range(total_images):
        img_path = all_paths_shuffled[i]
        should_transform = flags_shuffled[i] # A flag decide o destino desta imagem
        
        try:
            img_nib = nib.load(img_path)
            img_data_f16 = img_nib.get_fdata(dtype=np.float16)
            
            img_final = None

            # Aplica transformação SOMENTE se a flag permitir E houver compositor
            if should_transform and transform_composer:
                img_data_f32 = img_data_f16.astype(np.float32)
                subject = tio.Subject(
                    mri=tio.ScalarImage(tensor=img_data_f32[np.newaxis, ...], affine=img_nib.affine)
                )
                
                # Aqui o TorchIO sorteia os parâmetros aleatórios
                transformed_subject = transform_composer(subject)
                
                transformed_data_f32 = transformed_subject.mri.data.numpy().squeeze(axis=0)
                img_final = transformed_data_f32.astype(np.float16)
                
                del img_data_f32, subject, transformed_subject, transformed_data_f32
            else:
                # Mantém original
                img_final = img_data_f16
            
            images_final[i] = img_final.reshape((*img_shape, 1))
            paths_final[i] = img_path

        except Exception as e:
            print(f"Erro em {img_path}: {e}")
            images_final[i] = np.zeros((*img_shape, 1), dtype=np.float16)
            paths_final[i] = img_path
        
        if (i + 1) % 100 == 0:
            gc.collect()
            print(f"Progresso: {i + 1}/{total_images}")
            
    print("Concluído.")
    return images_final, labels_one_hot_shuffled, paths_final, label_encoder.classes_

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Função geradora para ser passada durante o treinamento
def nifti_data_generator_3d(images_array, labels, batch_size):
    total_n = len(images_array)
    if total_n == 0:
        raise ValueError("O Generator recebeu uma lista vazia de imagens! Verifique o carregamento dos dados.")

    while True:
        for i in range(0, total_n, batch_size):
            final = min(i + batch_size, total_n)
            batch_images = np.array(images_array[i: final])
            batch_labels = np.array(labels[i:final])
            
            yield batch_images, batch_labels

# Função para carregar imagens NIfTI, seus rótulos e cortar as imagens
def nifti_data_generator_3d_path(image_paths, labels, batch_size, size):
    cache_size = batch_size*size
    while True:
        for i in range(0, len(image_paths), cache_size):
            final = min(i + cache_size, len(image_paths))
            batch_paths = image_paths[i:final]
            batch_labels = labels[i:final]
            images = []

            for path in batch_paths:
                # Carregar a imagem NIfTI e garantir o formato correto
                img = nib.load(path).get_fdata(dtype=np.float16) 
                img = img[..., np.newaxis] 
                images.append(img)
            
            # Converter lista para array NumPy e garantir o shape correto
            images = np.array(images) 
            batch_labels = np.array(batch_labels)

            # Liberar memória
            gc.collect()
            
            yield images, batch_labels

def load_nifti_data_augmented_balanced(base_dir, class_names):
    # Define as transformações disponíveis (usadas apenas para as classes minoritárias)
    available_transforms = [
        tio.RandomBlur(p=1.0),
        tio.RandomNoise(p=1.0, std=(0, 0.05)),
        tio.RandomAnisotropy(p=1.0),
        tio.RandomElasticDeformation(p=1.0),
        tio.RandomBiasField(p=1.0),
        tio.RandomMotion(p=1.0),
        tio.RandomSpike(p=1.0),
        tio.RandomGhosting(p=1.0),
    ]

    all_paths = []
    all_labels = []
    
    # --- Passo 1: Coletando lista de arquivos e determinando o target ---
    print("Passo 1: Coletando lista de arquivos e calculando o target...")
    
    # 1.1 Coletar todos os caminhos e contar o tamanho máximo de classe
    max_count = 0
    class_paths_map = {}
    
    for label in class_names:
        label_dir = os.path.join(base_dir, label)
        # Filtra apenas arquivos .nii ou .nii.gz (ajuste se necessário)
        paths = [os.path.join(label_dir, fname) 
                 for fname in os.listdir(label_dir) 
                 if fname.endswith(('.nii', '.nii.gz', '.img'))]
        
        class_paths_map[label] = paths
        
        if len(paths) > max_count:
            max_count = len(paths) # max_count será 2100 no seu exemplo
            
    # 1.2 Aplicar a lógica de balanceamento/aumento
    for label in class_names:
        original_paths = class_paths_map[label]
        num_originals = len(original_paths) # Ex: 2100 (MCI) ou 700 (CN/AD)
        num_to_augment = max_count - num_originals # Ex: 0 (MCI) ou 1400 (CN/AD)
        
        # Adiciona os arquivos originais
        all_paths.extend(original_paths)
        all_labels.extend([label] * num_originals)
        
        # Adiciona cópias aumentadas para balancear as classes minoritárias
        if num_to_augment > 0:
            print(f"Classe {label}: {num_originals} originais. Adicionando {num_to_augment} aumentados.")
            
            # Repete a lista original para ter caminhos suficientes para as cópias aumentadas
            # Ex: (1400 // 700) = 2. Repetimos 2 vezes a lista original
            # Usamos ceil para garantir que o número de repetições seja suficiente
            repetitions = int(np.ceil(num_to_augment / num_originals))
            
            # Cria uma lista de caminhos originais para serem transformados
            paths_to_transform = (original_paths * repetitions)[:num_to_augment]
            
            all_paths.extend(paths_to_transform)
            all_labels.extend([label] * num_to_augment)
        else:
            print(f"Classe {label}: {num_originals} originais. Não requer aumento.")

    print(f"Total de {len(all_paths)} imagens (originais + aumentadas) preparadas.")
    total_images = len(all_paths)
    
    # --- Passo 2: Codificação das labels e Shuffle ---
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(class_names)
    labels_encoded = label_encoder.transform(all_labels)
    labels_one_hot = to_categorical(labels_encoded, num_classes=len(class_names)).astype(np.float16) 
    
    # O shuffle aqui é crucial para misturar os dados originais e aumentados
    all_paths_shuffled, labels_one_hot_shuffled = shuffle(all_paths, labels_one_hot, random_state=42)
    
    del all_labels, labels_encoded, labels_one_hot
    gc.collect()

    if not all_paths_shuffled:
        print("Nenhuma imagem encontrada.")
        return np.array([]), np.array([]), [], label_encoder.classes_

    # --- Passo 3: Alocação de Memória e Carregamento ---
    print("Passo 3: Determinando o shape da imagem e alocando memória...")
    try:
        first_img_nib = nib.load(all_paths_shuffled[0])
        img_shape = first_img_nib.get_fdata(dtype=np.float16).shape
    except Exception as e:
        print(f"Erro ao carregar a primeira imagem: {e}")
        return

    print(f"Shape detectado: {img_shape}. Alocando memória para {total_images} imagens...")
    images_final = np.empty((total_images, *img_shape, 1), dtype=np.float16)
    
    # Lista de tuplas (caminho_original, é_aumentada)
    paths_final = [None] * total_images 

    # --- Passo 4: Carregando, Normalizando e Aplicando Aumento ---
    print("Passo 4: Carregando e transformando imagens (Aumento aplicado apenas a cópias)...")
    
    for i in range(total_images):
        img_path = all_paths_shuffled[i]
        label = label_encoder.inverse_transform([np.argmax(labels_one_hot_shuffled[i])])[0]
        
        # Determina se esta cópia deve ser aumentada ou carregada como original
        # Se o caminho for um dos caminhos originais, e a classe não for a majoritária,
        # e a posição 'i' estiver além do número de originais, ela é uma cópia.
        is_augmented_copy = (img_path in class_paths_map[label]) and \
                            (total_images // len(class_names) > class_paths_map[label]) and \
                            (i >= len(class_paths_map[label]))
                            
        # Lógica simplificada: Aumento é aplicado SE a classe era minoritária
        # e estamos em uma posição de cópia.
        apply_augmentation = (label in class_paths_map and len(class_paths_map[label]) < max_count)
        
        try:
            img_nib = nib.load(img_path)
            img_data_f16 = img_nib.get_fdata(dtype=np.float16)
            
            # Aplica Aumento SE esta cópia é uma das que estamos usando para balancear
            # E se a classe minoritária for a que precisa de aumento.
            if apply_augmentation and (i >= len(class_paths_map[label])):
                
                # Seleciona aleatoriamente de 1 a 6 transformações do pool
                num_transforms = random.randint(1, 6)
                selected_transforms = random.sample(available_transforms, num_transforms)
                transform_composer = tio.Compose(selected_transforms)

                img_data_f32 = img_data_f16.astype(np.float32)
                subject = tio.Subject(
                    mri=tio.ScalarImage(tensor=img_data_f32[np.newaxis, ...], affine=img_nib.affine)
                )
                
                # Aplica a composição
                transformed_data_f32 = transform_composer(subject).mri.data.numpy().squeeze(axis=0)
                img_final = transformed_data_f32.astype(np.float16)
                
                del img_data_f32, subject, transformed_data_f32, transform_composer, selected_transforms
                paths_final[i] = (img_path, True) # Marcado como aumentada
            else:
                img_final = img_data_f16
                paths_final[i] = (img_path, False) # Marcado como original (ou da classe majoritária)

            images_final[i] = img_final.reshape((*img_shape, 1))

        except Exception as e:
            print(f"Erro em {img_path}: {e}. Inserindo array vazio.")
            images_final[i] = np.zeros((*img_shape, 1), dtype=np.float16)
            paths_final[i] = (img_path, "ERRO") 
            
        if (i + 1) % 50 == 0:
            gc.collect()
            print(f"Processado {i + 1}/{total_images}...")
            
    print("Passo 5: Carregamento concluído.")
    
    return images_final, labels_one_hot_shuffled, paths_final, label_encoder.classes_

def load_nifti_data_balanced_full_augment(base_dir, class_names, augment_originals=True):
    """
    Carrega dados NIfTI, equaliza as quantidades pelo maior (oversampling)
    e aplica augmentation em TODOS os dados (inclusive originais da classe majoritária)
    se augment_originals=True.
    """
    
    # Lista de transformações possíveis
    # Ajuste as probabilidades (p) conforme a "intensidade" de ruído que deseja
    available_transforms = [
        tio.RandomBlur(p=0.5),              # Borrão
        tio.RandomNoise(p=0.5, std=(0, 0.05)), # Ruído Gaussiano
        tio.RandomBiasField(p=0.3),         # Variação de campo magnético
        tio.RandomMotion(p=0.2),            # Simula movimento leve
        tio.RandomGhosting(p=0.2),          # Artefatos de fantasma
        tio.RandomGamma(p=0.3)              # Ajuste de contraste/gama
    ]

    print("Passo 1: Mapeamento e Balanceamento...")
    paths_by_class = {}
    
    # 1.1 Coleta caminhos e define o alvo (classe com mais arquivos)
    for label in class_names:
        label_dir = os.path.join(base_dir, label)
        valid_files = [
            os.path.join(label_dir, f) 
            for f in os.listdir(label_dir) 
            if f.endswith(('.nii', '.nii.gz'))
        ]
        paths_by_class[label] = valid_files
    
    max_samples = max(len(files) for files in paths_by_class.values())
    print(f"-> Alvo de balanceamento: {max_samples} imagens por classe.")

    # 1.2 Cria a lista de tarefas
    # Cada tarefa é uma tupla: (caminho_arquivo, label, aplicar_aug)
    loading_tasks = []

    for label in class_names:
        original_files = paths_by_class[label]
        n_originals = len(original_files)
        
        # A. Adiciona os Originais (CN, MCI, AD)
        # Se augment_originals=True, marcamos True para aplicar ruído neles também
        for path in original_files:
            loading_tasks.append((path, label, augment_originals))
        
        # B. Adiciona Cópias para as classes menores (MCI, AD) para atingir o teto
        deficit = max_samples - n_originals
        if deficit > 0:
            print(f"   Gerando {deficit} cópias sintéticas para '{label}'...")
            base_files_for_aug = random.choices(original_files, k=deficit)
            for path in base_files_for_aug:
                loading_tasks.append((path, label, True)) # Cópias SEMPRE sofrem augmentation
    
    random.shuffle(loading_tasks)
    total_images = len(loading_tasks)

    # Prepara Labels One-Hot
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    labels_encoded = label_encoder.transform([t[1] for t in loading_tasks])
    labels_one_hot = to_categorical(labels_encoded, num_classes=len(class_names)).astype(np.float16)

    # Passo 2: Alocação de Memória
    try:
        sample_path = loading_tasks[0][0]
        first_img = nib.load(sample_path).get_fdata(dtype=np.float16)
        img_shape = first_img.shape
    except Exception as e:
        print(f"Erro ao ler shape: {e}")
        return

    print(f"Alocando memória para {total_images} volumes {img_shape}...")
    images_final = np.empty((total_images, *img_shape, 1), dtype=np.float16)
    paths_final = [None] * total_images

    # Passo 3: Processamento
    print("Passo 3: Carregando e aplicando transformações...")
    
    for i, (img_path, label, apply_aug) in enumerate(loading_tasks):
        try:
            img_nib = nib.load(img_path)
            data = img_nib.get_fdata(dtype=np.float16)
            
            if apply_aug:
                # Sorteia transformações
                # Nota: mesmo originais recebem ruído aqui
                selected = random.sample(available_transforms, k=random.randint(1, 3))
                transform = tio.Compose(selected)
                
                # Torchio exige 4D float32
                subject = tio.Subject(
                    mri=tio.ScalarImage(tensor=data.astype(np.float32)[np.newaxis, ...], affine=img_nib.affine)
                )
                data = transform(subject).mri.data.numpy().squeeze(axis=0).astype(np.float16)
            
            # Garante shape correto
            if data.shape != img_shape:
                 # Resize simples se necessário (implementar se seus dados variarem muito)
                 pass

            images_final[i] = data.reshape((*img_shape, 1))
            paths_final[i] = img_path

        except Exception as e:
            print(f"Erro imagem {i}: {e}")
            images_final[i] = np.zeros((*img_shape, 1), dtype=np.float16)

        if (i + 1) % 50 == 0:
            gc.collect()
            print(f"Progresso: {i + 1}/{total_images}")

    return images_final, labels_one_hot, paths_final, label_encoder.classes_