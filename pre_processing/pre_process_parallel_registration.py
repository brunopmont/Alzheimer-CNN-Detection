#!/usr/bin/env python3
import os
import numpy as np
import ants
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from concurrent.futures import as_completed
import gc

# Índices de corte para o corte NIfTI
SLICE_NII_IDX0 = slice(24, 169)
SLICE_NII_IDX1 = slice(24, 206)
SLICE_NII_IDX2 = slice(6, 161)

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# FUNÇÕES
# Winsorize -> reduz outliers, limitando os percentis inf e sup
def winsorize_image(image_data, lower_percentile=0, upper_percentile=98): #reduz valores extremos
    lower_bound = np.percentile(image_data, lower_percentile)
    upper_bound = np.percentile(image_data, upper_percentile)
    winsorized_data = np.clip(image_data, lower_bound, upper_bound)
    return winsorized_data

# Normalization -> valores de voxels entre 0 e 1
def normalize_image_min(image_data): 
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    normalized_data = (image_data - min_val) / (max_val - min_val)
    return normalized_data

# Função para processar uma única imagem
def process_image(img_path, template, mask, orient):
    try:
        #logger.info(f"IMAGE_TO_READ {img_path}")
        # Carrega a imagem
        image = ants.image_read(img_path, reorient=orient)
        #logger.info(f"IMAGE_READ")

        # Registro (Registration) com as transformações para a imagem e máscara, com intuito de melhorar o corte
        registration = ants.registration(fixed=template, moving=image, type_of_transform='Translation')
        warped_image = registration['warpedmovout']
        #logger.info(f"TRANSLATION")

        registration = ants.registration(fixed=template, moving=warped_image, type_of_transform='Rigid')
        warped_image = registration['warpedmovout']
        #logger.info(f"RIGID")

        registration = ants.registration(fixed=template, moving=warped_image, type_of_transform='Affine')
        warped_image = registration['warpedmovout']
        #logger.info(f"AFFINE")

        registration = ants.registration(fixed=template, moving=warped_image, type_of_transform='SyN')
        warped_image = registration['warpedmovout']
        #logger.info(f"SYN")

        # Máscara do cérebro e extração
        brain_masked = ants.mask_image(warped_image, mask)
        data = brain_masked.numpy()

        # Aplicação do crop usando slices diretamente
        data = data[
            SLICE_NII_IDX0,
            SLICE_NII_IDX1,
            SLICE_NII_IDX2
        ]
        
        # Recria a imagem ANTs com os dados cortados
        image = ants.from_numpy(
            data,
            origin=brain_masked.origin,
            spacing=brain_masked.spacing,
            direction=brain_masked.direction
        )

        # Bias Field Correction
        image = ants.n4_bias_field_correction(image, shrink_factor=2)
        #logger.info(f"BIAS")

        # Winsorizing
        data = image.numpy()
        data = winsorize_image(data, lower_percentile=0, upper_percentile=99.9)

        # Normalização
        data = normalize_image_min(data)
        image = ants.from_numpy(data, origin=image.origin, spacing=image.spacing, direction=image.direction)

        return image
        
    except Exception as e:
        logger.error(f"Erro ao processar a imagem {os.path.basename(img_path)}: {e}")
        return None

# Função que processa e salva uma imagem
def process_and_save_image(img_path, template, mask, output_dir, orient):
    logger.info(f"Imagem em processamento: {img_path}")
    normalized_image = process_image(img_path, template, mask, orient) #processa imagem
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
    # DIR_BASE = "/mnt/c/Users/Team Taiane/Desktop/OASIS"
    DIR_BASE = "/mnt/c/Users/Paulo Pires/Desktop/Alzheimer_cnn/OASIS-1"
    DIR_RAW = f"{DIR_BASE}/OASIS_RAW"
    DIR_OUTPUT = f"{DIR_BASE}/{os.path.basename(DIR_RAW)}_PROCESSED"
    os.makedirs(DIR_OUTPUT, exist_ok=True)
    DIR_MASK = "pre_processing/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c"
    
    template_path = os.path.join(DIR_MASK, 'mni_icbm152_t1_tal_nlin_asym_09c.nii')
    mask_path = os.path.join(DIR_MASK, 'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii')

    template = ants.image_read(template_path) 
    mask = ants.image_read(mask_path) 

    start_time = datetime.now()
    logger.info(f"INICIO DO PROCESSAMENTO")

    for name in os.listdir(DIR_RAW): #itera as subpastas dentre o diretóiro original

        input_path = os.path.join(DIR_RAW, name)
        output_path = os.path.join(DIR_OUTPUT, name)
        os.makedirs(output_path, exist_ok=True)

        # Caminhos das imagens
        already_processed = [file for file in os.listdir(output_path)] #checa o diretório de saída pra ver se alguma imagem já foi processada
        image_paths = [os.path.join(input_path, file) for file in os.listdir(input_path) if file not in already_processed] #carrega o endereço das imagens não processadas

        # Função parcial para passar parâmetros fixos
        process_func = partial(process_and_save_image, template=template, mask=mask, output_dir=output_path, orient='IRA')

        # Processamento e salvamento de cada imagem usando ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=16) as executor: #max_workers define o número máximo de processos paralelos
            # dependendo do pc, é melhor fazer um por vez, pois paralelizar pode deixar cada processo mais demorado sem hardware que aguente
            futures = [executor.submit(process_func, img_path) for img_path in image_paths]
            
            for future in as_completed(futures):
                future.result()  # Pega o resultado para garantir que exceções sejam lançadas


    # Fim do processamento
    end_time = datetime.now()
    logger.info(f"FIM DO PROCESSAMENTO")
    logger.info(f"Duração total: {end_time - start_time}")