import os
import numpy as np
from sklearn.metrics import classification_report
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import tempfile

def get_predictions(images, labels_one_hot, batch_size, model):
    """Realiza predições em lote."""
    predictions = model.predict(images, batch_size=batch_size)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels_one_hot, axis=1)
    return true_labels, pred_labels, predictions

def get_classification_report(true_labels, pred_labels, results_dir, suffix):
    """Gera e salva o relatório de classificação do Scikit-Learn."""
    report = classification_report(true_labels, pred_labels)
    print(report)
    
    with open(os.path.join(results_dir, f"classification_report_{suffix}.txt"), "w") as f:
        f.write(report)

def create_pdf(paths, images, true_labels, pred_labels, raw_preds, save_path, class_names):
    """
    Gera um PDF com as imagens e suas predições (Exemplo Simplificado).
    Você deve substituir isso pela sua implementação original que usa reportlab/pillow
    se ela tiver formatação específica.
    """
    c = canvas.Canvas(save_path, pagesize=letter)
    width, height = letter
    y_position = height - 50
    
    c.drawString(30, y_position, "Relatório de Predições")
    y_position -= 30
    
    # Exemplo simples de loop
    for i in range(min(len(paths), 20)): # Limitado a 20 para exemplo
        if y_position < 100:
            c.showPage()
            y_position = height - 50
            
        res_text = f"Img: {os.path.basename(paths[i])} | Real: {class_names[true_labels[i]]} | Pred: {class_names[pred_labels[i]]}"
        c.drawString(30, y_position, res_text)
        y_position -= 20
        
    c.save()
    print(f"PDF salvo em: {save_path}")