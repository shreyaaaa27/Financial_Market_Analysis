import tensorflow as tf
import numpy as np
import cv2
import os

# ----------------------------
# LOAD IMAGE
# ----------------------------
def load_image(img_path, size=(64,64)):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image not found")

    img = cv2.resize(img, size)
    img = img / 255.0
    img = img.reshape(1, size[0], size[1], 1)

    return img


# ----------------------------
# BUILD CNN SUBMODEL ONLY
# ----------------------------
def get_cnn_model(full_model):

    cnn_input = full_model.input[0]
    conv_layer = full_model.get_layer("conv2").output

    return tf.keras.Model(inputs=cnn_input, outputs=conv_layer)


# ----------------------------
# GRAD-CAM (FIXED FOR FUSION)
# ----------------------------
def generate_heatmap(model, img_array, seq_input):

    cnn_model = get_cnn_model(model)

    seq_input = tf.convert_to_tensor(seq_input)

    img_array = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:

        tape.watch(img_array)

        conv_out = cnn_model(img_array)

        # fake loss from CNN branch only
        loss = tf.reduce_mean(conv_out)

    grads = tape.gradient(loss, conv_out)

    pooled = tf.reduce_mean(grads, axis=(0,1,2))

    conv_out = conv_out[0]

    heatmap = tf.tensordot(conv_out, pooled, axes=1)

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


# ----------------------------
# SAVE HEATMAP
# ----------------------------
def save_heatmap(img_path, heatmap):

    img = cv2.imread(img_path)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    os.makedirs("heatmaps", exist_ok=True)

    cv2.imwrite("heatmaps/result.png", overlay)

    print("✅ Heatmap saved")


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    from tensorflow.keras.models import load_model
    from fusion_dataset import load_fusion_dataset

    print("Loading model...")

    model = load_model("fusion_model.h5", compile=False)

    print("Model loaded")

    X_img, X_seq, y = load_fusion_dataset(window=20)

    img_sample = X_img[0].reshape(1,20,4,1)
    seq_sample = X_seq[0]

    heatmap = generate_heatmap(model, img_sample, seq_sample)

    save_heatmap("candles/candle_3_HOLD.png", heatmap)