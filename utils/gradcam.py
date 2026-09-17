import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import zoom


def get_gradcam_3d(model, volume, class_index, last_conv_layer_name):

    base_model = model.model if hasattr(model, 'model') else model

    grad_model = tf.keras.models.Model(
        [base_model.inputs],
        [base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    with tf.GradientTape() as tape:
        volume_float32 = tf.cast(volume, tf.float32)
        # O modelo espera um batch, então expandimos a dimensão
        conv_outputs, predictions = grad_model(tf.expand_dims(volume_float32, axis=0))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    # Faz o cast explícito para float32 para evitar erros de tipo
    conv_outputs = tf.cast(conv_outputs, tf.float32)
    grads = tf.cast(grads, tf.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))  # média global

    # Pondera os mapas de ativação
    conv_outputs = conv_outputs[0]
    heatmap = tf.zeros(conv_outputs.shape[:-1], dtype=tf.float32)

    for i in range(pooled_grads.shape[0]):
        heatmap += pooled_grads[i] * conv_outputs[..., i]

    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / tf.reduce_max(heatmap + 1e-6)

    return heatmap.numpy()


def resize_heatmap_to_volume(heatmap, target_shape):
    zoom_factors = (
        target_shape[0] / heatmap.shape[0],
        target_shape[1] / heatmap.shape[1],
        target_shape[2] / heatmap.shape[2]
    )
    heatmap_upscaled = zoom(heatmap, zoom_factors, order=1)  # order=1: interpolação linear
    return heatmap_upscaled


def show_gradcam_slice(volume, heatmap, slice_idx=None, axis=0, alpha=0.5, cmap='jet'):
    if slice_idx is None:
        slice_idx = volume.shape[axis] // 2

    if axis == 0:
        img_slice = volume[slice_idx, :, :, 0]
        heatmap_slice = heatmap[slice_idx, :, :]
    elif axis == 1:
        img_slice = volume[:, slice_idx, :, 0]
        heatmap_slice = heatmap[:, slice_idx, :]
    else:
        img_slice = volume[:, :, slice_idx, 0]
        heatmap_slice = heatmap[:, :, slice_idx]

    plt.imshow(img_slice, cmap='gray')
    plt.imshow(heatmap_slice, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.title(f'Slice {slice_idx} (Axis {axis})')
    plt.show()
