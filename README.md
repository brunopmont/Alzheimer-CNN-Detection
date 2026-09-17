# Alzheimer-CNN-Detection

Este projeto é parte da minha Iniciação Científica no IANS/UFF, onde desenvolvi e treinei um modelo de Deep Learning para detecção de Alzheimer a partir de exames de ressonância magnética.

### Tecnologias Utilizadas
* **Linguagens:** Python
* **Frameworks de DL:** TensorFlow / Keras
* **Manipulação de Dados:** NumPy, Pandas
* **Visualização:** Matplotlib

### Principais Etapas
1. **Pré-processamento:** Limpeza e normalização de imagens de ressonância magnética para otimizar o treinamento da rede.
2. **Arquitetura:** Implementação de uma CNN 3D customizada para extração de características espaciais complexas.
3. **Validação:** Aplicação de técnicas de explicabilidade para aumentar a confiança clínica do modelo.

### Resultados
* **Acurácia:** 93% no conjunto de teste.
* O modelo demonstra alta performance na distinção dos padrões neurais associados à doença.

---

## Estrutura do repositório

| Diretório | Conteúdo |
|---|---|
| `utils/` | Módulos compartilhados por todos os notebooks — carregamento de dados, métricas, arquiteturas e Grad-CAM |
| `pre_processing/` | Conversão DICOM→NIfTI, registro no template MNI, extração de cérebro e crop |
| `cnn/` | Treinamento: validação cruzada, transfer learning, treino com ruído e o experimento SFR |
| `3T_data_prediction/` | Experimentos com dados 3T — pré-treino em 1.5T e fine-tuning em 3T |
| `explainability/` | Grad-CAM e oclusão (por região anatômica e por janelas quadradas) |
| `analysis/` | Comparação entre datasets, análise de qualidade de imagem e consolidação de métricas |
| `reproducao_artigos/` | Reproduções de trabalhos de terceiros, mantidas com o código original |

## Os módulos de `utils/`

Notebooks e scripts importam daqui em vez de redefinir as funções localmente.

**`processamento_dados.py`** — carregamento e augmentation
`load_nifti_data_balanced_preallocated` e `load_nifti_data_from_multiple_sources` carregam volumes NIfTI já balanceados por classe e em one-hot, alocando a memória de uma vez (`float16`); a primeira é um atalho da segunda para um único diretório. `get_augmentation_pipeline` monta o compose do TorchIO (afim + deformação elástica, e opcionalmente artefatos de intensidade). `nifti_data_generator_3d` e `nifti_data_generator_3d_path` são os geradores para `model.fit`.

**`metricas_e_visualizacao.py`** — avaliação e relatórios
`get_predictions` (softmax) e `get_predictions_binary` (sigmoid) rodam predição em lotes; `plot_training_history`, `plot_confusion_matrix`, `plot_roc_curve`, `save_auc` e `get_classification_report` geram e salvam os artefatos de avaliação. `create_pdf` e `generate_axial_pdf_reports` exportam as fatias com rótulo real e predito para inspeção visual.

**`modelos.py`** — `create_model_3d`, a CNN 3D usada como padrão: três blocos `Conv3D → BatchNorm → LeakyReLU → AveragePooling → Dropout` (4→8→16 filtros), uma densa de 16 e saída softmax, com regularização L2.

**`gradcam.py`** — `get_gradcam_3d`, `resize_heatmap_to_volume` e `show_gradcam_slice`, as primitivas usadas pelos notebooks de explicabilidade.

Alguns notebooks mantêm versões próprias de funções com nome parecido porque o comportamento é diferente de propósito — por exemplo, a matriz de confusão do SFR usa fontes de artigo, e o `create_pdf` do pipeline 2D tem outra assinatura. Arquiteturas alternativas ficam no notebook que as usa, com nome próprio (`create_model_3d_wide`, `_double_conv`, `_small`, `_maxpool`).

## Convenção dos dados

As funções de carregamento esperam um diretório por classe, com os nomes passados em `class_names`:

```
dataset/
├── cn/    paciente_001.nii  ...
└── ad/    paciente_042.nii  ...
```

Os rótulos saem em one-hot na ordem de `class_names`. Os conjuntos usados variam por experimento: `['cn', 'ad']`, `['cn', 'mci', 'ad']` e `['cn', 'emci', 'mci', 'lmci', 'ad']` no ADNI; no OASIS as pastas são os escores CDR (`['0.0', '0.5', '1.0']`).

## Pré-processamento

`dicom_to_nii.py` converte as séries DICOM para NIfTI. Em seguida, `pre_process_parallel_registration.py` aplica, para cada volume:

1. Registro progressivo no template MNI ICBM152 (Translation → Rigid → Affine → SyN)
2. Extração do cérebro pela máscara do template
3. Crop pelos índices fixos definidos no topo do script
4. Correção de campo de viés (N4)

`pre_process_individual_mask.py` faz o mesmo, mas gera uma máscara por paciente com `antspynet.brain_extraction` em vez de usar a do template. O template e as máscaras ficam em `pre_processing/mni_icbm152_nlin_asym_09c_nifti/`.

## Usando os utils em um notebook

Os notebooks rodam a partir do próprio diretório, então o import adiciona a raiz do projeto ao `sys.path`:

```python
import os, sys

project_root = os.path.abspath('..')        # '../..' se o notebook estiver dois níveis abaixo
if project_root not in sys.path:
    sys.path.append(project_root)

import utils.processamento_dados as proc_dados
import utils.metricas_e_visualizacao as met_vil
import utils.modelos as modelos
import utils.gradcam as gcam
```

Um treino mínimo fica assim:

```python
train_images, train_labels, train_paths, class_labels = proc_dados.load_nifti_data_balanced_preallocated(
    train_dir, class_names=['cn', 'ad'], augment=True, target_per_class=1000
)

model = modelos.create_model_3d(train_images[0].shape, n_classes=2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    proc_dados.nifti_data_generator_3d(train_images, train_labels, batch_size),
    steps_per_epoch=len(train_images) // batch_size,
    epochs=epochs,
)

met_vil.plot_training_history(history, results_dir)
pred_labels, true_labels, pred = met_vil.get_predictions(val_images, val_labels, batch_size, model)
met_vil.plot_confusion_matrix(true_labels, pred_labels, results_dir, 'val', class_labels)
```

`get_predictions` retorna nesta ordem: **predito, real, probabilidades**.

## Dependências

Não há `requirements.txt`; as bibliotecas usadas ao longo do projeto são:

**Núcleo** — `tensorflow`, `numpy`, `scipy`, `scikit-learn`, `nibabel`, `torchio`, `pandas`
**Pré-processamento** — `antspyx`, `antspynet`, `SimpleITK`, `scikit-image`
**Visualização e relatórios** — `matplotlib`, `seaborn`, `reportlab`, `Pillow`, `tqdm`, `ipywidgets`

As reproduções em `reproducao_artigos/` usam PyTorch (`torch`, `torchvision`) em vez de TensorFlow.

## Reproduções de artigos

`reproducao_artigos/` contém implementações de trabalhos de terceiros, usadas como comparação. O código é mantido como no original — inclusive as convenções de nome e estrutura — e por isso **não** segue os módulos de `utils/`. Cada subdiretório tem o próprio README:

* `TL4ADdiagnosis/` — transfer learning para diagnóstico de AD, com uma abordagem geral (extração de features + SVM/KNN/RF) e uma de fine-tuning
* `AD_classification/` — CNN 3D treinada do zero com validação cruzada

---
*Desenvolvido por: Bruno Porto (Estudante de Ciência da Computação - UFF)*
