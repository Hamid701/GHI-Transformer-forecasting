import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization


def build_transformer(
    window_size,
    d_model=98,
    num_heads=7,
    dff=199,
    num_layers=4,
    dropout_rate=0.2935,
    learning_rate=0.001,
):
    """Build Transformer model with proper sequence handling"""
    # Input layer with explicit sequence dimension
    inputs = tf.keras.Input(shape=(window_size, 1))  # Changed from (window_size,)

    # Embedding layer (process each time step)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(d_model))(inputs)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Positional encoding (fixed for all batches)
    position = tf.range(window_size, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(
        tf.range(0, d_model, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / d_model)
    )
    pos_enc = tf.concat(
        [tf.sin(position * div_term), tf.cos(position * div_term)], axis=-1
    )
    x = x + pos_enc[tf.newaxis, ...]  # Add batch dimension

    # Transformer blocks
    for _ in range(num_layers):
        # Self-attention
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )(x, x)
        attn = tf.keras.layers.Dropout(dropout_rate)(attn)
        x = LayerNormalization(epsilon=1e-6)(x + attn)

        # Feed forward
        ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ]
        )
        ffn_out = ffn(x)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_out)

    # Output (using last sequence element)
    outputs = tf.keras.layers.Dense(1)(x[:, -1, :])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    return model


def build_lstm(window_size, units=128, dropout_rate=0.1702, learning_rate=0.0164):
    """Build LSTM model"""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                units, return_sequences=True, input_shape=(window_size, 1)
            ),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(units),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    return model
