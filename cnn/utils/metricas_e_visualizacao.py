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

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import tempfile
from math import ceil
import random
from tensorflow.keras import backend as K

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
    return pred_labels, true_labels, pred

def plot_training_history(history, dir, title='training_history.png'):
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
    plt.plot(history.history['binary_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Graphic')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(dir, title))

    plt.show()

def plot_confusion_matrix(y_true, y_pred, dir, subset, class_names):
    cm = confusion_matrix(y_true, y_pred)

    # Plotando a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names,  annot_kws={"size": 14})
    plt.xlabel('Previsões')
    plt.ylabel('Valores Reais')
    plt.title(f'Matriz de Confusão - {subset}')
    plt.savefig(f'{dir}/{subset}_confusion_matrix.png')
    plt.show()

def get_classification_report(y_true, y_pred, dir, subset):
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
        if i % 12 == 0:
            c.showPage()

    # Salvar o PDF
    c.save()

def plot_custom_confusion_matrix(cm, y_labels, x_labels, dir, subset):
    plt.figure(figsize=(10, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=x_labels, yticklabels=y_labels, 
                annot_kws={"size": 14})
    plt.xlabel('Previsões')
    plt.ylabel('Valores Reais')
    plt.title(f'Matriz de Confusão - {subset}')
    plt.savefig(f'{dir}/{subset}_custom_confusion_matrix.png', bbox_inches='tight')
    plt.show()

def generate_axial_pdf_reports(model, images, true_labels_onehot, class_names, output_dir, max_samples=None):
    """
    Gera um PDF SEPARADO para cada volume 3D, contendo todas as fatias axiais.
    
    Args:
        model: Modelo treinado.
        images: Array numpy (N, D, H, W, C).
        true_labels_onehot: Labels reais (N, n_classes).
        class_names: Lista de nomes das classes.
        output_dir: Diretório onde os PDFs serão salvos.
        max_samples: Limite de amostras.
    """
    
    # 1. Setup inicial
    true_indices = np.argmax(true_labels_onehot, axis=1)
    
    n_samples = len(images)
    if max_samples is not None and max_samples < n_samples:
        n_samples = max_samples
        print(f"Limitando relatórios a {n_samples} amostras.")

    # 2. Cria o diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    print(f"Gerando {n_samples} relatórios PDF em: {output_dir}")

    # 3. Loop Externo (Um PDF por volume)
    for i in range(0,n_samples,20):
        vol = images[i]
        true_index = true_indices[i]
        true_label = class_names[true_index]

        # 4. Predição (Feita uma vez por volume)
        vol_batch = np.expand_dims(vol, axis=0) 
        pred_probs_single = model.predict(vol_batch, verbose=0)
        
        pred_index = np.argmax(pred_probs_single[0])
        pred_label = class_names[pred_index]
        confidence = pred_probs_single[0][pred_index] * 100

        # 5. Define status e nome do arquivo
        is_correct = true_index == pred_index
        status = "CORRETO" if is_correct else "ERRO"
        title_color = 'green' if is_correct else 'red'
        
        # Nome descritivo para o PDF (preenchimento com zeros para ordenação)
        pdf_filename = f"Amostra_{i:04d}_{status}_Real_{true_label}_Pred_{pred_label}.pdf"
        pdf_path_full = os.path.join(output_dir, pdf_filename)
        
        try:
            # 6. Abre o arquivo PDF para o volume atual
            with PdfPages(pdf_path_full) as pdf:
                
                # Assumindo shape (D, H, W, C), o eixo Axial é o índice 2 (W)
                num_axial_slices = vol.shape[2] 
                
                # 7. Loop Interno (Uma página por fatia axial)
                for k in range(num_axial_slices):
                    # Extrai a fatia axial 'k'
                    img_slice = vol[:, :, k, 0] 

                    fig, ax = plt.subplots(figsize=(6, 6))
                    
                    # Usa .T (Transposto) e origin='lower' para orientação radiológica padrão
                    ax.imshow(img_slice.T, cmap='gray', origin='lower')
                    ax.axis('off')
                    
                    # Título para a página
                    title_text = (f"Amostra {i} [Fatia Axial {k+1}/{num_axial_slices}] [{status}]\n"
                                  f"Real: {true_label} | Pred: {pred_label} ({confidence:.2f}%)")
                    
                    ax.set_title(title_text, color=title_color, fontsize=10)
                    
                    # Salva a página atual no PDF
                    pdf.savefig(fig)
                    # Fecha a figura para liberar memória (CRÍTICO)
                    plt.close(fig) 

        except Exception as e:
            print(f"Erro ao gerar PDF {pdf_path_full}: {e}")
            plt.close('all') # Fecha todas as figuras em caso de falha

        # Log no loop externo
        if (i + 20) % 100 == 0: # Log mais frequente
            print(f"Processado {i + 10}/{n_samples} volumes (PDFs gerados)...")

    print(f"Geração de {n_samples} relatórios PDF concluída.")

def generate_axial_pdf_reports_no_prediction(images, true_labels_onehot, class_names, output_dir, max_samples=None):
    """
    Gera um PDF SEPARADO para cada volume 3D, contendo todas as fatias axiais.
    
    Args:
        model: Modelo treinado.
        images: Array numpy (N, D, H, W, C).
        true_labels_onehot: Labels reais (N, n_classes).
        class_names: Lista de nomes das classes.
        output_dir: Diretório onde os PDFs serão salvos.
        max_samples: Limite de amostras.
    """
    
    # 1. Setup inicial
    true_indices = np.argmax(true_labels_onehot, axis=1)
    
    n_samples = len(images)
    if max_samples is not None and max_samples < n_samples:
        n_samples = max_samples
        print(f"Limitando relatórios a {n_samples} amostras.")

    # 2. Cria o diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    print(f"Gerando {n_samples} relatórios PDF em: {output_dir}")

    # 3. Loop Externo (Um PDF por volume)
    for i in range(0,n_samples):
        vol = images[i]
        true_index = true_indices[i]
        true_label = class_names[true_index]
    
        # Nome descritivo para o PDF (preenchimento com zeros para ordenação)
        pdf_filename = f"Amostra_{i:04d}_Real_{true_label}_.pdf"
        pdf_path_full = os.path.join(output_dir, pdf_filename)
        
        try:
            # 6. Abre o arquivo PDF para o volume atual
            with PdfPages(pdf_path_full) as pdf:
                
                # Assumindo shape (D, H, W, C), o eixo Axial é o índice 2 (W)
                num_axial_slices = vol.shape[2] 
                
                # 7. Loop Interno (Uma página por fatia axial)
                for k in range(num_axial_slices):
                    # Extrai a fatia axial 'k'
                    img_slice = vol[:, :, k, 0] 

                    fig, ax = plt.subplots(figsize=(6, 6))
                    
                    # Usa .T (Transposto) e origin='lower' para orientação radiológica padrão
                    ax.imshow(img_slice.T, cmap='gray', origin='lower')
                    ax.axis('off')
                    
                    # Título para a página
                    title_text = (f"Amostra {i} [Fatia Axial {k+1}/{num_axial_slices}]]\n"
                                  f"Real: {true_label}")
                    
                    ax.set_title(title_text, fontsize=10)
                    
                    # Salva a página atual no PDF
                    pdf.savefig(fig)
                    # Fecha a figura para liberar memória (CRÍTICO)
                    plt.close(fig) 

        except Exception as e:
            print(f"Erro ao gerar PDF {pdf_path_full}: {e}")
            plt.close('all') # Fecha todas as figuras em caso de falha

        # Log no loop externo
        if (i + 20) % 100 == 0: # Log mais frequente
            print(f"Processado {i + 10}/{n_samples} volumes (PDFs gerados)...")

    print(f"Geração de {n_samples} relatórios PDF concluída.")

def get_predictions_binary(images, labels, batch_size, best_model):
    pred = []

    # Loop de predição em batches (para não estourar a memória)
    for i in range(0, len(images), batch_size):
        final = min(i + batch_size, len(images))
        
        # Fazendo predição para o lote atual
        # verbose=0 evita poluir o log com barras de progresso repetidas
        batch_pred = best_model.predict(images[i:final], verbose=0) 
        pred.append(batch_pred)

    # Concatenando todas as predições em um único array
    pred = np.concatenate(pred)

    pred_labels = (pred > 0.5).astype("int32").flatten()

    # 2. Tratar os Rótulos Verdadeiros (True Labels)
    # Se os labels já vierem como 0 e 1 (dimensão 1), não usamos argmax.
    # Se vierem one-hot (ex: [[0,1], [1,0]]), aí sim usamos argmax.
    if labels.ndim > 1 and labels.shape[-1] > 1:
        true_labels = np.argmax(labels, axis=1)
    else:
        true_labels = labels.astype("int32").flatten()

    return pred_labels, true_labels, pred

def plot_training_history_binary(history, dir, title='training_history.png'):
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
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Graphic')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(dir, title))

    plt.show()