import os
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor
import sys
from datetime import datetime
import logging

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

    DIR_BASE = os.path.abspath("/mnt/c/Users/Team Taiane/Desktop/ADNI/FULL_ADNI/DICOM")
    DIR_RAW = os.path.join("/mnt/c/Users/Team Taiane/Desktop/ADNI/FULL_ADNI/NIFTI_RAW")

    os.makedirs(DIR_RAW, exist_ok=True)

    logging.info(f"Convertendo imagens de:\n{DIR_BASE}\npara:\n{DIR_RAW}")

    start_time = datetime.now()
    logging.info(f"Início do processamento em: {start_time}")

    for subfolder in os.listdir(DIR_BASE):
        input_dir = os.path.join(DIR_BASE, subfolder)
        output_dir = os.path.join(DIR_RAW, subfolder)
        for group in os.listdir(input_dir):
            logging.info(f"\nCONVERSOES DA PASTA {group}\n")

            input_folder = os.path.join(input_dir, group)
            output_folder = os.path.join(output_dir, group)

            os.makedirs(output_folder, exist_ok=True)

            # Coletar todas as pastas DICOM
            already_converted = [file for file in os.listdir(output_folder)]
            dicom_folders = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file not in already_converted]

            logging.info(f"IMAGENS PROCESSADAS: {len(already_converted)}\nIMAGENS A SEREM PROCESSADAS: {len(dicom_folders)}")

            with ProcessPoolExecutor(16) as executor:
                futures = {executor.submit(convert_dicom_to_nifti, folder, output_folder): folder for folder in dicom_folders}
                
                for future in futures:
                    try:
                        future.result()  # Relata erros
                    except Exception as e:
                        logging.error(f"Erro ao processar {futures[future]}: {e}")

            tot_images += len(dicom_folders)

            logging.info(f'\nForam convertidas {len(dicom_folders)} imagens!')

    # Fim do processamento
    logging.info(f'\nForam convertidas {len(dicom_folders)} imagens!')
    end_time = datetime.now()
    logging.info(f"Término do processamento em: {end_time}")
    logging.info(f"Duração total: {end_time - start_time}")