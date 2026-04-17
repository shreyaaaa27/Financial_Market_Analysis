from tensorflow.keras import layers, Model

def build_fusion_model(img_shape, seq_shape):

    # ---------------- CNN ----------------
    img_input = layers.Input(shape=img_shape)

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same', name="conv1")(img_input)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same', name="conv2")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)

    # ---------------- LSTM ----------------
    seq_input = layers.Input(shape=seq_shape)

    y = layers.LSTM(64)(seq_input)

    # ---------------- FUSION ----------------
    combined = layers.concatenate([x, y])

    z = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[img_input, seq_input], outputs=output)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


    return model