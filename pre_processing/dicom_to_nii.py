import os
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor
import sys
from datetime import datetime
import logging
from tqdm import tqdm

# CONFIGURAÇÕES DO LOG
logging.basicConfig(
    filename='conversion.log',  # Arquivo de log
    level=logging.INFO,         # Nivel de log (INFO para mensagens normais, ERROR para erros)
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# FUNÇÕES
def get_f_dir(directory):
    sub_item = os.listdir(directory)
    directory = os.path.abspath(os.path.join(directory, sub_item[0]))
    return directory

def load_dicom_series(input_folder):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(input_folder)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    return image

def find_final_dir(dir_path):
    if(os.path.isdir(dir_path)):
        files = os.listdir(dir_path)
        for file in files:
            if file.endswith(".dcm"):
                return dir_path
        path = find_final_dir(os.path.join(dir_path, file))
        return path

def save_as_nifti(image, output_file):
    sitk.WriteImage(image, output_file)

def reorient_image(image):
    # Reorienta a imagem para o sistema padrão RAS (Right, Anterior, Superior)
    return sitk.DICOMOrient(image, 'RAS')

def convert_dicom_to_nifti(input_folder, output_folder):
    logging.info(f"CONVERTENDO IMAGEM {input_folder}")

    # Formar nome de saída pelo 'I...'
    output_name = os.path.basename(input_folder) + '.nii.gz'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Carrega a série DICOM
    image = load_dicom_series(input_folder)

    # Reorienta a imagem para o padrão RAS
    image = reorient_image(image)

    # Nome do arquivo NIfTI de saída
    output_file = os.path.abspath(os.path.join(output_folder, output_name))

    # Salva no formato NIfTI
    save_as_nifti(image, output_file)

    # Mensagem de conclusão
    logging.info(f"Imagem {output_name} convertida com sucesso!")

# CONVERSÃO
if __name__ == "__main__":

    tot_images = 0
    dcm_paths = []

    DIR_BASE = os.path.abspath("/mnt/c/Users/Paulo Pires/Desktop/Alzheimer_cnn/AIBL/AIBL")
    DIR_RAW = os.path.join("/mnt/c/Users/Paulo Pires/Desktop/Alzheimer_cnn/AIBL/AIBL_raw")
    os.makedirs(DIR_RAW, exist_ok=True)

    logging.info(f"Convertendo imagens de:\n{DIR_BASE}\npara:\n{DIR_RAW}")

    for item in (os.listdir(DIR_BASE)):
        item_path = find_final_dir(os.path.join(DIR_BASE, item))
        if item_path != None:
            dcm_paths.append(item_path)

    start_time = datetime.now()
    logging.info(f"Início do processamento em: {start_time}")

    os.makedirs(DIR_RAW, exist_ok=True)

    already_converted = [os.path.basename(file) for file in os.listdir(os.path.join(DIR_RAW))]
    dicom_paths = []

    for item in dcm_paths:
        if os.path.basename(item).rsplit('.dcm', 1)[0] not in already_converted:
            dicom_paths.append(item)

    # Coletar todas as pastas DICOM
    # for sub in tqdm(os.listdir(os.path.join(DIR_BASE)), f"CARREGANDO:"):
    #     names = os.listdir(os.path.join(DIR_BASE, sub))
    #     datas = os.listdir(os.path.join(DIR_BASE, sub, names[0]))
    #     file = os.listdir(os.path.join(DIR_BASE, sub, names[0], datas[0]))
    #     file_path = os.path.join(DIR_BASE, sub, names[0], datas[0], file[0])
    #     if f"{os.path.basename(file_path)}.nii.gz" not in already_converted:
    #         dicom_folders.append(file_path)

    logging.info(f"IMAGENS PROCESSADAS: {len(already_converted)}\nIMAGENS A SEREM PROCESSADAS: {len(dicom_paths)}")

    with ProcessPoolExecutor(32) as executor:
        futures = {executor.submit(convert_dicom_to_nifti, folder, os.path.join(DIR_RAW)): folder for folder in dicom_paths}
        
        for future in futures:
            try:
                future.result()  # Relata erros
            except Exception as e:
                logging.error(f"Erro ao processar {futures[future]}: {e}")

    tot_images += len(dicom_paths)

    logging.info(f'\nForam convertidas {len(dicom_paths)} imagens!')

    # Fim do processamento
    logging.info(f'\nForam convertidas {len(dicom_paths)} imagens!')
    end_time = datetime.now()
    logging.info(f"Término do processamento em: {end_time}")
    logging.info(f"Duração total: {end_time - start_time}")