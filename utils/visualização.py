import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

def unificar_tamanhos_com_padding(lista_de_imagens):
    """Função auxiliar para deixar cortes de tamanhos iguais para plotagem."""
    max_altura = 0
    max_largura = 0
    for img in lista_de_imagens:
        altura, largura = img.shape
        if altura > max_altura: max_altura = altura
        if largura > max_largura: max_largura = largura

    imagens_uniformes = []
    for img in lista_de_imagens:
        fundo = np.zeros((max_altura, max_largura))
        altura_img, largura_img = img.shape
        y_offset = (max_altura - altura_img) // 2
        x_offset = (max_largura - largura_img) // 2
        fundo[y_offset:y_offset+altura_img, x_offset:x_offset+largura_img] = img
        imagens_uniformes.append(fundo)
        
    return imagens_uniformes

def plot_views_uniforme_final(image, main_title, k=0, sag_idx=90, cor_idx=110, ax_idx=110, figsize=(15, 5), axes_pad=0.3):
    """Plota as 3 visões (Sagital, Coronal, Axial) de uma MRI."""
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=axes_pad)
    
    # Remove a dimensão do canal se existir (ex: 160, 160, 160, 1) -> (160, 160, 160)
    if image.ndim == 4:
        image = image.squeeze()

    slices_originais = [
        np.rot90(image[sag_idx, :, :], k=k),
        np.rot90(image[:, cor_idx, :], k=k),
        np.rot90(image[:, :, ax_idx], k=k)
    ]
    
    slices_uniformizadas = unificar_tamanhos_com_padding(slices_originais)
    titles = ["Sagital", "Coronal", "Axial"]

    for ax, im_slice, title in zip(grid, slices_uniformizadas, titles):
        ax.imshow(im_slice, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    fig.suptitle(main_title)
    plt.show()

def plot_custom_confusion_matrix(cm, display_labels, x_tick_labels, results_dir, filename_suffix):
    """
    Plota matriz de confusão personalizada (útil quando treino e teste têm classes diferentes).
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical')
    
    # Ajuste manual dos labels do eixo X se fornecidos
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
        
    plt.title(f'Matriz de Confusão - {filename_suffix}')
    plt.savefig(os.path.join(results_dir, f"confusion_matrix_{filename_suffix}.png"))
    plt.show()
    plt.close()