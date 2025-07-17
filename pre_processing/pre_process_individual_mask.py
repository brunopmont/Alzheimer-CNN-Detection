#!/usr/bin/env python3
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from concurrent.futures import as_completed
import os
import numpy as np
import ants
import logging
from antspynet.utilities import brain_extraction
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desativa GPUs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Supressão de logs detalhados do TensorFlow

MODALITY = 't1' #BOTE 'flair' ou 't1'

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# FUNÇÕES
# Winsorize -> reduz outliers, limitando os percentis inf e sup
def winsorize_image(image_data, lower_percentile=0, upper_percentile=99.9):
    lower_bound = np.percentile(image_data, lower_percentile)
    upper_bound = np.percentile(image_data, upper_percentile)
    winsorized_data = np.clip(image_data, lower_bound, upper_bound)
    return winsorized_data

# Normalization -> valores de voxels entre 0 e 1
def normalize_image(image_data):
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    normalized_data = (image_data - min_val) / (max_val - min_val)
    return normalized_data

# Função para processar uma única imagem
def process_image(img_path, output_dir):
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    try:
        logger.info(f"Inicio processamento: {img_path}")
        # Carrega a imagem
        image = ants.image_read(img_path)

        # Registra pra padronizar shape da imagem
        registration = ants.registration(fixed=template, moving=image, type_of_transform='Affine')
        affine_image = registration['warpedmovout']

        # Cria template pra máscara
        prob_mask = brain_extraction(affine_image, modality=MODALITY)
        logger.info(f"Template obtido.")

        # Cria a máscara
        mask = ants.get_mask(prob_mask, low_thresh=0.5)
        logger.info(f"Máscara aplicada.")

        # Máscara do cérebro e extração
        brain_masked = ants.mask_image(affine_image, mask)
        logger.info(f"Extração.")

        # Bias Field Correction
        #image = ants.from_numpy(data, origin=image.origin, spacing=image.spacing, direction=image.direction)
        image = ants.n4_bias_field_correction(brain_masked, shrink_factor=2)
        data = image.numpy()
        logger.info(f"Bias Corrigido.")

        # Winsorizing
        data = winsorize_image(data, 0, 99.9)
        logger.info(f"Winsorized.")

        # Normalização
        data = normalize_image(data)
        image = ants.from_numpy(data, origin=brain_masked.origin, spacing=brain_masked.spacing, direction=brain_masked.direction)

        logger.info(f"Imagem {img_path} processada.")

        ants.image_write(image, output_path)
        logger.info(f"Imagem salva: {os.path.basename(output_path)}")

        gc.collect()

        return image
        
    except Exception as e:
        logger.error(f"Erro ao processar a imagem {img_path}: {e}")
        return None
    
subsets = ['train', 'validation', 'test']
    
# DIRETÓRIOS
DIR_INPUT_BASE = f"/mnt/c/Users/Paulo Pires/Desktop/Alzheimer_cnn/NIFTI_RAW"
DIR_OUTPUT_BASE = "/mnt/c/Users/Paulo Pires/Desktop/Alzheimer_cnn/NIFTI_PROCESSED"
os.makedirs(DIR_OUTPUT_BASE, exist_ok=True)

template_path = "/mnt/c/Users/Paulo Pires/Desktop/Alzheimer_cnn/Alzheimer-CNN-Detection/pre_processing/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii"
template = ants.image_read(template_path)

for subset in subsets:
    DIR_SUBSET_INPUT = f"{DIR_INPUT_BASE}/{subset}"
    DIR_SUBSET_OUTPUT = f"{DIR_OUTPUT_BASE}/{subset}"
    os.makedirs(DIR_SUBSET_OUTPUT, exist_ok=True)

    for label in ['cn', 'emci', 'mci', 'lmci', 'ad']:
        DIR_INPUT = f"{DIR_SUBSET_INPUT}/{label}"
        DIR_OUTPUT = f"{DIR_SUBSET_OUTPUT}/{label}"
        os.makedirs(DIR_OUTPUT, exist_ok=True)

        # Checa o diretório de saída pra ver se alguma imagem já foi processada
        already_processed = [file for file in os.listdir(DIR_OUTPUT)]

        # Lista de caminhos para as imagens brutas
        image_paths = [os.path.join(DIR_INPUT, file) for file in os.listdir(DIR_INPUT) if file not in already_processed]

        # Início do processamento
        if __name__ == "__main__":
            start_time = datetime.now()
            logger.info(f"Início do processamento em: {start_time}")

            print(f"\n\nIMAGENS PROCESSADAS: {len(already_processed)}\nIMAGENS A PROCESSAR: {len(image_paths)}\n\n")

            # Função parcial para passar parâmetros fixos
            process_func = partial(process_image, output_dir=DIR_OUTPUT)

            # Processamento e salvamento de cada imagem usando ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=16) as executor: #max_workers define o número máximo de processos paralelos

                # dependendo do pc, é melhor fazer um proceso só, pois paralelizar pode deixar cada processo mais demorado sem hardware que aguente
                futures = [executor.submit(process_func, img_path) for img_path in image_paths]
                
                for future in as_completed(futures):
                    future.result()  # Pega o resultado para garantir que exceções sejam lançadas

            # Fim do processamento
            end_time = datetime.now()
            logger.info(f"Término do processamento em: {end_time}")
            logger.info(f"Duração total: {end_time - start_time}")