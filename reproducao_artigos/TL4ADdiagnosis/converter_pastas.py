import os
import numpy as np
import nibabel as nib
import glob

# Configurações
pasta_entrada = "3T_data"    # Seus .nii originais
pasta_saida = "3T_data_npy"  # Destino organizado

classes = ["AD", "CN"]

for classe in classes:
    # Caminho de entrada (tenta maiúsculo e minúsculo)
    caminho_in = os.path.join(pasta_entrada, classe)
    if not os.path.exists(caminho_in):
        caminho_in = os.path.join(pasta_entrada, classe.lower())
    
    # Caminho de saída (Sempre cria subpastas AD e CN)
    caminho_out = os.path.join(pasta_saida, classe)
    if not os.path.exists(caminho_out):
        os.makedirs(caminho_out)

    arquivos = glob.glob(os.path.join(caminho_in, "*.nii"))
    print(f"Processando {classe}: {len(arquivos)} arquivos...")

    for f in arquivos:
        try:
            img = nib.load(f)
            data = img.get_fdata()
            
            # Salva DENTRO da subpasta da classe
            nome = os.path.basename(f).replace('.nii', '.npy')
            np.save(os.path.join(caminho_out, nome), data)
        except Exception as e:
            print(f"Erro: {e}")

print("Concluído! Pastas AD e CN criadas dentro de 3T_data_npy.")