import os

# pra usar cpu, descomentar linha abaixo
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
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
from tensorflow.keras import mixed_precision

from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, models, Input

import matplotlib.pyplot as plt
import gc
import seaborn as sns

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import tempfile
from math import ceil

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import utils.metricas_e_visualizacao as met_vil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.get_logger().setLevel('ERROR')

# Configurar para usar apenas a memória necessária da GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory Growth habilitado para a GPU")
    except RuntimeError as e:
        print(e)

# FUNÇÕES


def load_nifti_data_balanced(base_dir, class_names, target=None):
    images = []
    labels = []
    paths = []
    
    for label in class_names:
        print(f"Carregando diretório '{label}'...")
        label_dir = os.path.join(base_dir, label)

        filenames = os.listdir(label_dir)
        if target:
            filenames = filenames[:target]

        for fname in filenames:
            img_path = os.path.join(label_dir, fname)

            img = nib.load(img_path).get_fdata(dtype=np.float16)
            img = np.expand_dims(img, axis=-1)

            images.append(img)
            labels.append(label)
            paths.append(img_path)

        print(f"Diretório '{label}' carregado com {len(filenames)} imagens.")

    # Convertendo para arrays NumPy
    images = np.stack(images, axis=0)  # mais eficiente e seguro que np.array
    print(f"Shape dos dados: {images.shape}")

    # Codificando os rótulos
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(class_names)

    labels_encoded = label_encoder.transform(labels)
    labels_one_hot = to_categorical(labels_encoded, num_classes=len(class_names))

    # Embaralhar os dados
    images, labels_one_hot, paths = shuffle(images, labels_one_hot, paths, random_state=42)

    return images, labels_one_hot, paths, label_encoder.classes_

            
def nifti_data_generator_3d_indexed(full_images, full_labels, index_list, batch_size):
    while True:
        for i in range(0, len(index_list), batch_size):
            batch_idx = index_list[i:i+batch_size]
            yield full_images[batch_idx], full_labels[batch_idx]









def create_model_3d_seq(input_shape, n_classes):
    model = Sequential([        
        Input(shape=input_shape),  # Formato de entrada: (1, 145, 182, 155)

        # Camada 2
        Conv3D(4, (3, 3, 3), padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.3),  
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
        Dropout(0.3),

        # Camada 3
        Conv3D(8, (3, 3, 3), padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.3),  
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
        Dropout(0.3),

        # Camada 4
        Conv3D(16, (3, 3, 3), padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.3),  
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
        Dropout(0.3),

        # Camada de saída convolucional
        Flatten(),

        # Camadas densas
        Dense(16, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        LeakyReLU(negative_slope=0.3),  

        # Camadas densas
        Dense(8, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        LeakyReLU(negative_slope=0.3),  

        # Camada de saída
        Dense(n_classes, activation='softmax')
    ])
    
    return model

def create_model_3d_maxpool(input_shape, n_classes):
    inputs = Input(shape=input_shape)  # (D, H, W, C)

    # Camada 1
    x = layers.Conv3D(2, (3, 3, 3), padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.3)(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
    x = layers.Dropout(0.3)(x)

    # Camada 2
    x = layers.Conv3D(4, (3, 3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.3)(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
    x = layers.Dropout(0.3)(x)

    # Camada 3
    x = layers.Conv3D(8, (3, 3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.3)(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
    x = layers.Dropout(0.3)(x)

    # Camada 4
    x = layers.Conv3D(16, (3, 3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.3)(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
    x = layers.Dropout(0.3)(x)

    # Flatten e densas
    x = layers.Flatten()(x)

    x = layers.Dense(16, kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LeakyReLU(negative_slope=0.3)(x)

    x = layers.Dense(8, kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LeakyReLU(negative_slope=0.3)(x)

    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

def create_model_3d_best(input_shape, n_classes):
    model = Sequential([        
        Input(shape=input_shape),  # Formato de entrada: (1, 145, 182, 155)

        # Camada 1 - Filtro 3x3
        Conv3D(4, (3, 3, 3), padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.3),  
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
        Dropout(0.4),

        # Camada 2 - Filtro 5x5
        Conv3D(8, (3, 3, 3), padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.3),  
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
        Dropout(0.4),

        # Camada 3 - Filtro 5x5
        Conv3D(16, (3, 3, 3), padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.3),  
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
        Dropout(0.4),

        # Camada de saída convolucional
        Flatten(),

        # Camadas densas
        Dense(16, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        LeakyReLU(negative_slope=0.3),  

        # Camadas densas
        Dense(8, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        LeakyReLU(negative_slope=0.3),  

        # Camada de saída
        Dense(n_classes, activation='softmax')
    ])
    
    return model

# Definindo caminhos
dir_base = "/mnt/c/Users/Paulo Pires/Desktop/Alzheimer_cnn/3D_BRAIN_NORM"

train_dir = f'{dir_base}/train'
val_dir = f'{dir_base}/validation'
test_dir = f'{dir_base}/test'
results_dir = f'{dir_base}/results/folds'

# Criar o diretório de resultados se ele não existir
os.makedirs(results_dir, exist_ok=True)

# Nome das classes
class_names = ['cn', 'emci', 'mci', 'lmci', 'ad']

n_classes = len(class_names)

train_images, train_labels, train_paths, class_labels = load_nifti_data_balanced(train_dir, class_names, target=None)
val_images, val_labels, val_paths, _ = load_nifti_data_balanced(val_dir, class_names, target=None)

batch_size = 64

steps_per_epoch = len(train_paths) // batch_size
validation_steps = len(val_paths) // batch_size

print(f"N treino: {len(train_paths)}")
print(f"N validation: {len(val_paths)}")

full_images = np.concatenate([train_images, val_images],  axis=0)
full_labels = np.concatenate([train_labels, val_labels],  axis=0)
full_paths = train_paths + val_paths

shape = full_images[0].shape

del train_images, train_labels, train_paths, val_images, val_labels, val_paths

n = len(os.listdir(results_dir))
        
# if (n > 0):
#     if (len(os.listdir(os.path.join(results_dir, f'test_{n}'))) < 5): 
#         for item in os.listdir(os.path.join(results_dir, f"test_{n}")):
#             os.remove(os.path.join(results_dir,  f"test_{n}", item))
#         os.removedirs(os.path.join(results_dir, f'test_{n}'))
#         n -= 1

folder_name = f"test_{str(n+1)}"
results_dir = os.path.join(results_dir, folder_name)
os.makedirs(results_dir, exist_ok=True)
print(f"pasta {folder_name} criada")

epochs = 250

new_model_name_ker = (f"binary_classifier_{epochs}_epochs_batch_{batch_size}_{n_classes}_classes.keras")

# Parar caso fique {patience} épocas sem melhora
early_stopping = EarlyStopping(
    monitor='val_loss',     
    patience=15,                 
    verbose=1
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=1)

count = 1
size = len(full_images)
step = size // 5

tf.keras.backend.clear_session()

for i in range(0, size, step):
    print(f"\n\nINICIANDO FOLD {count}\n\n")

    # Compila modelo
    model = create_model_3d_maxpool(shape, n_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # initial_weights = model.get_weights()

    final = min(i + step, size)
    results_fold = f"{results_dir}/fold_{count}"
    os.makedirs(results_fold, exist_ok=True)

    # Defina o nome do arquivo para salvar o melhor modelo
    model_checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(results_fold, new_model_name_ker),    
    monitor='val_categorical_accuracy',
    save_best_only=True, 
    mode='max', 
    )

    log_path = os.path.join(results_fold, 'log_treino.csv')

    csv_log = CSVLogger(log_path, append=False)

    #print(f"\nCALLBACKS SETADOS\n")

    # val_images = full_images[i:final]
    # val_labels = full_labels[i:final]
    # val_paths = full_paths[i:final]

    # print(f"\nVAL DATA\n")

    # train_images = np.delete(full_images, np.s_[i:final], axis=0)
    # train_labels = np.delete(full_labels, np.s_[i:final], axis=0)

    val_idx = np.arange(i, final)
    train_idx = np.setdiff1d(np.arange(size), val_idx)

    train_idx = shuffle(train_idx, random_state=36)
    val_idx = shuffle(val_idx, random_state=36)

    val_paths = [full_paths[i] for i in val_idx]

    #print(f"\nTRAIN DATA\n")

    val_generator = nifti_data_generator_3d_indexed(full_images, full_labels, val_idx, batch_size)
    train_generator = nifti_data_generator_3d_indexed(full_images, full_labels, train_idx, batch_size)

    #print(f"\nGERADORES CONFIGURADOS\n")

    print(f"Iniciando treinamento do modelo {new_model_name_ker} para classes {class_names}")

    # Treinamento
    history = model.fit(
        train_generator,
        epochs=epochs,
        verbose=1,
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[model_checkpoint_callback, reduce_lr, csv_log]
    )

    # Plotando o histórico de treinamento após o treinamento
    met_vil.plot_training_history(history, results_fold)

    # Realizar predições para dados do conjunto validação
    val_pred_labels, val_true_labels, val_pred = met_vil.get_predictions(full_images[val_idx], full_labels[val_idx], batch_size, model)

    # Obter métricas da valiadação e salvá-las em um arquivo
    met_vil.get_classification_report(val_true_labels, val_pred_labels, results_fold, 'val')

    # Obter matriz de confusão
    met_vil.plot_confusion_matrix(val_true_labels, val_pred_labels, results_fold, 'val', class_names, f'fold {count}')

    # Criar pdf com predições
    val_pdf_path = os.path.join(results_fold, "validation_predictions.pdf")
    met_vil.create_pdf(val_paths, full_images[val_idx], val_true_labels, val_pred_labels, val_pred, val_pdf_path, class_names)

    model_checkpoint_callback = None
    csv_log = None

    #print(f"Memória GPU alocada: {tf.config.experimental.get_memory_info('GPU:0')['current']} bytes")

    del history, val_true_labels, val_pred_labels, val_pred, val_paths
    gc.collect()

    count += 1

   #print(f"Memória GPU alocada: {tf.config.experimental.get_memory_info('GPU:0')['current']} bytes")