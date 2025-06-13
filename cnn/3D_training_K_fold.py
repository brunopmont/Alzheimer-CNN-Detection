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

# Função para carregar imagens NIfTI, seus rótulos e cortar as imagens
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

    # Convertendo a lista de rótulos para um array NumPy
    labels_array = np.array(labels)

    # Codificando os rótulos (agora 'cn' será 0 e 'ad' será 1)
    labels_encoded = label_encoder.transform(labels_array)

    # Transformando os rótulos para one-hot encoding
    labels_one_hot = to_categorical(labels_encoded, num_classes=len(class_names))

    # Embaralhar os dados
    image_paths, labels_one_hot = shuffle(image_paths, labels_one_hot, random_state=42)

    return image_paths, labels_one_hot, label_encoder.classes_

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

def nifti_data_generator_3d(images_array, labels, batch_size):
    total_n = len(images_array)
    while True:
        for i in range(0, total_n, batch_size):
            final = min(i + batch_size, total_n)
            yield images_array[i:final], labels[i:final]
            
def nifti_data_generator_3d_indexed(full_images, full_labels, index_list, batch_size):
    while True:
        for i in range(0, len(index_list), batch_size):
            batch_idx = index_list[i:i+batch_size]
            yield full_images[batch_idx], full_labels[batch_idx]


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
                img = nib.load(path).get_fdata(dtype=np.float16)  # Shape original: 
                img = img[..., np.newaxis]       # Adicionar a dimensão do canal: 
                images.append(img)
            
            # Converter lista para array NumPy e garantir o shape correto
            images = np.array(images) 
            batch_labels = np.array(batch_labels)

            # Liberar memória
            gc.collect()
            
            yield images, batch_labels

# realizar predições e armazenar em um vetor
def get_predictions(images, labels, batch_size, best_model):
    pred = []

    for i in range(0, len(images), batch_size):
        final = min(i + batch_size, len(images))
        
        # Fazendo predição para o lote atual
        batch_pred = best_model.predict(images[i:final])
        pred.append(batch_pred)

    # Concatenando as predições e os rótulos verdadeiros
    pred = np.concatenate(pred)

    # Convertendo as predições para rótulos (a classe com maior probabilidade)
    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(pred, axis=1)
    return true_labels, pred_labels, pred

def plot_training_history(history, dir):
    plt.figure(figsize=(12, 4))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Graphic')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Graphic')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(dir, 'training_history.png'))

    #plt.show()
    plt.close('all')

def plot_confusion_matrix(y_true, y_pred, dir, subset, class_names, comp=''):
    # Calcular a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)

    # Plotando a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names,  annot_kws={"size": 14})
    plt.xlabel('Previsões')
    plt.ylabel('Valores Reais')
    plt.title(f'Matriz de Confusão{comp}')
    plt.savefig(f'{dir}/{subset}_confusion_matrix.png')
    #plt.show()
    plt.close('all')

def get_classification_report(y_true, y_pred, dir, subset):
    # Gerar relatório
    report = classification_report(y_true, y_pred)
    print(report)

    # Escrevendo o relatório em um arquivo .txt
    with open(os.path.join(dir, f"{subset}_classification_report.txt"), "w") as file:
        file.write(report)

# Função para carregar uma imagem NIfTI e extrair uma fatia específica do eixo Z
def load_nifti_image_pdf(file_path):
    img = nib.load(file_path) 
    data = img.get_fdata(dtype=np.float16)  
    slice_2d = data[2, :, :]

    # implementar lógica de retornar vetor com cada fatia como imagem única

    return slice_2d

# Função para criar o PDF
def create_pdf(y_paths, y_images, y_true_labels, y_pred_labels, y_pred, output_pdf_path, class_names):
    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    width, height = letter  # Dimensões da página no PDF

    #for image, name in zip(y_images, y_paths):
    for i in range(0, len(y_images)):
        true = ''
        pred = ''
        # Carregar a imagem NIfTI e obter a fatia 2D no eixo Z
        # nifti_image = load_nifti_image_pdf(item)
        nifti_image = y_images[i][:, :, 88, 0]

        # Converter a fatia 2D para uma imagem 8-bit (grayscale) para visualização
        img = Image.fromarray(np.uint8(nifti_image / np.max(nifti_image) * 255))  # Normalizar e converter
        img = img.convert("RGB")  # Garantir que a imagem tenha 3 canais (RGB)

        # Redimensionar a imagem para se ajustar ao tamanho da página
        img_width, img_height = img.size
        aspect_ratio = img_height / float(img_width)
        new_width = width * 0.2  # Definir largura como 80% da largura da página
        new_height = new_width * aspect_ratio
        img = img.resize((int(new_width), int(new_height)))

        # Criar um arquivo temporário para salvar a imagem
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file_path = temp_file.name
            img.save(temp_file_path)

        # configurar para printar as 7 fatias em uma página inteira, com as informações de label predito e esperado

        # Colocar a imagem no PDF usando o caminho temporário
        if i % 12 < 4:
            x = 80
        elif i % 12 < 8:
            x = width - 2.35*new_width - 80
        else:
            x = width - new_width - 80

        y = height - (new_height + 80)*((i%4)+1)

        c.drawImage(temp_file_path, x, y, width=new_width, height=new_height)

        # Escrever os rótulos
        true_label = y_true_labels[i]
        pred_label = y_pred_labels[i]

        #ver como transformar os labels de maneira inteligente
        true = class_names[true_label]
        pred = class_names[pred_label]

        # Definir a cor para os rótulos
        if true_label == pred_label:
            pred_color = (0, 1, 0)  # Verde
        else:
            pred_color = (1, 0, 0)  # Vermelho
        
        #Nome paciente (em preto)
        c.setFont("Helvetica", 12)
        c.setFillColorRGB(0, 0, 0)  # Preto
        c.drawString(x+24, y+new_height+50, f"{os.path.basename(y_paths[i])}")

        # Rótulo esperado (em preto)
        c.setFont("Helvetica", 12)
        c.setFillColorRGB(0, 0, 0)  # Preto
        c.drawString(x+26, y+new_height+35, f"Expected: {true}")

        # Rótulo predito
        c.setFont("Helvetica", 12)
        c.setFillColorRGB(*pred_color)  # Verde ou Vermelho
        c.drawString(x+26, y+new_height+20, f"Predicted: {pred}")

        # Rótulo predito
        c.setFont("Helvetica", 12)
        c.setFillColorRGB(*pred_color)  # Verde ou Vermelho
        c.drawString(x+26, y+new_height+5, f"Prob: {max(y_pred[i])*100:.2f}%")

        # Avançar para a próxima imagem
        i += 1
        
        # Adicionar uma nova página no PDF a cada 2 imagens (se necessário)
        if i % 12 == 0:  # Por exemplo, a cada 2 imagens, adicionamos uma nova página
            c.showPage()

    # Salvar o PDF
    c.save()

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

def create_model_3d(input_shape, n_classes):
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
    model = create_model_3d(shape, n_classes)
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
    plot_training_history(history, results_fold)

    # Realizar predições para dados do conjunto validação
    val_true_labels, val_pred_labels, val_pred = get_predictions(full_images[val_idx], full_labels[val_idx], batch_size, model)

    # Obter métricas da valiadação e salvá-las em um arquivo
    get_classification_report(val_true_labels, val_pred_labels, results_fold, 'val')

    # Obter matriz de confusão
    plot_confusion_matrix(val_true_labels, val_pred_labels, results_fold, 'val', class_names, f'_fold_{count}')

    # Criar pdf com predições
    val_pdf_path = os.path.join(results_fold, "validation_predictions.pdf")
    create_pdf(val_paths, full_images[val_idx], val_true_labels, val_pred_labels, val_pred, val_pdf_path, class_names)

    model_checkpoint_callback = None
    csv_log = None

    #print(f"Memória GPU alocada: {tf.config.experimental.get_memory_info('GPU:0')['current']} bytes")

    del history, val_true_labels, val_pred_labels, val_pred, val_paths
    gc.collect()

    count += 1

   #print(f"Memória GPU alocada: {tf.config.experimental.get_memory_info('GPU:0')['current']} bytes")
