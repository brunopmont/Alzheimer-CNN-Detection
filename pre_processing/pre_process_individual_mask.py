#!/usr/bin/env python3
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from concurrent.futures import as_completed
import numpy as np
import ants
import logging
from antspynet.utilities import brain_extraction
import gc
import tensorflow as tf

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
def process_image(img_path, output_dir, template):
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
        # data = image.numpy()
        logger.info(f"Bias Corrigido.")

        # # Winsorizing
        # data = winsorize_image(data, 0, 99.9)
        # logger.info(f"Winsorized.")

        # # Normalização
        # data = normalize_image(data)
        # image = ants.from_numpy(data, origin=brain_masked.origin, spacing=brain_masked.spacing, direction=brain_masked.direction)

        logger.info(f"Imagem {img_path} processada.")

        # ants.image_write(image, output_path)
        # logger.info(f"Imagem salva: {os.path.basename(output_path)}")

        gc.collect()
        tf.keras.backend.clear_session()

        return image
        
    except Exception as e:
        logger.error(f"Erro ao processar a imagem {img_path}: {e}")
        return None

# Função que processa e salva uma imagem
def process_and_save_image(img_path, output_dir, orient, template):
    # logger.info(f"Imagem em processamento: {img_path}")
    normalized_image = process_image(img_path, output_dir, template) #processa imagem
    if normalized_image is not None:
        os.makedirs(output_dir, exist_ok=True) #cria o diretório onde será salvo caso não exista
        output_path = os.path.join(output_dir, os.path.basename(img_path)) 
        ants.image_write(normalized_image, output_path)
        logger.info(f"Imagem salva: {os.path.basename(output_path)}")
        # Liberar a memória manualmente
        normalized_image = None
        gc.collect()

# Início do processamento
if __name__ == "__main__":
    # DIRETÓRIOS
    DIR_BASE = "/mnt/c/Users/Bruno/Desktop/IANS/ADNI"
    DIR_RAW = f"{DIR_BASE}/ADNI_3_4_RAW"
    DIR_OUTPUT = f"{DIR_BASE}/ADNI_3_4_PROCESSED"
    os.makedirs(DIR_OUTPUT, exist_ok=True)
    DIR_MASK = "/mnt/c/Users/Bruno/Documents/Github/Alzheimer-CNN-Detection/pre_processing/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c"
    
    template_path = os.path.join(DIR_MASK, 'mni_icbm152_t1_tal_nlin_asym_09c.nii')
    mask_path = os.path.join(DIR_MASK, 'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii')

    template = ants.image_read(template_path) 
    mask = ants.image_read(mask_path) 

    start_time = datetime.now()
    logger.info(f"INICIO DO PROCESSAMENTO")

    for name in os.listdir(f"{DIR_RAW}"): #itera as subpastas dentre o diretório original
        for class_name in os.listdir(f"{DIR_RAW}/{name}"):
            input_path = f"{DIR_RAW}/{name}/{class_name}"
            output_path = f"{DIR_OUTPUT}/{name}/{class_name}"
            os.makedirs(output_path, exist_ok=True)

            # Caminhos das imagens
            already_processed = [file for file in os.listdir(output_path)] #checa o diretório de saída pra ver se alguma imagem já foi processada
            image_paths = [os.path.join(input_path, file) for file in os.listdir(input_path) if file not in already_processed] #carrega o endereço das imagens não processadas

            print(f"IMAGENS PROCESSADAS {name}: {len(already_processed)}")
            print(f"IMAGENS A SEREM PROCESSADAS {name}: {len(image_paths)}")

            # Função parcial para passar parâmetros fixos
            process_func = partial(process_and_save_image, output_dir=output_path, orient='IRA', template=template)

            # Processamento e salvamento de cada imagem usando ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=4) as executor: #max_workers define o número máximo de processos paralelos
                # dependendo do pc, é melhor fazer um por vez, pois paralelizar pode deixar cada processo mais demorado sem hardware que aguente
                futures = [executor.submit(process_func, img_path) for img_path in image_paths]
                
                for future in as_completed(futures):
                    future.result()  # Pega o resultado para garantir que exceções sejam lançadas

    # Fim do processamento
    end_time = datetime.now()
    logger.info(f"FIM DO PROCESSAMENTO")
    logger.info(f"Duração total: {end_time - start_time}")