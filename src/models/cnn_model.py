from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

def create_cnn_model(input_shape, learning_rate=0.001, dropout_rate=0.3, filters=32):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # First Conv Block
        layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Second Conv Block
        layers.Conv1D(filters * 2, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Flatten and Dense layers
        layers.GlobalAveragePooling1D(),
        layers.Dense(filters, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model