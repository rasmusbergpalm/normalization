import keras
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed, RepeatVector
import keras.backend as K

import babel_data
import data


def all_acc(y_true, y_pred):
    return K.cast(
        K.all(
            K.equal(
                K.max(y_true, axis=-1),
                K.cast(K.argmax(y_pred, axis=-1), K.floatx())
            ),
            axis=1
        ),
        K.floatx()
    )


print("Creating data...")
babel_data.create("train", 1000000, True)
babel_data.create("dev", 100000, False)

print("Loading data...")
train = data.ParallelReader("train-source.txt", "vocab-source.txt", "", "train-target.txt", "vocab-target.txt", "")
dev = data.ParallelReader("dev-source.txt", "vocab-source.txt", "", "dev-target.txt", "vocab-target.txt", "")

print("Building model...")
# Encoder
source = Input(shape=(None,), dtype='int32', name='source')
embedded = Embedding(output_dim=128, input_dim=train.source_vocab_size(), mask_zero=True)(source)
last_hid = Bidirectional(LSTM(units=128))(embedded)
# Decoder
repeated = RepeatVector(train.target.padded.shape[1])(last_hid)
decoder = LSTM(units=128, return_sequences=True, name="decoder1")(repeated)
decoder = LSTM(units=128, return_sequences=True, name="decoder2")(decoder)
output = TimeDistributed(Dense(units=train.target_vocab_size(), activation='softmax'))(decoder)
model = Model([source], outputs=[output])

print("Compiling model...")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[all_acc])

print("Training...")
train_sources, train_targets = train.all()
dev_sources, dev_targets = dev.all()
model.fit(train_sources, train_targets[:, :, None],
          epochs=50,
          validation_data=(dev_sources, dev_targets[:, :, None]),
          callbacks=[keras.callbacks.ModelCheckpoint("model.{epoch:02d}-{val_loss:.2f}.hdf5",
                                                     monitor='val_loss',
                                                     verbose=0,
                                                     save_best_only=True,
                                                     save_weights_only=False,
                                                     mode='auto',
                                                     period=1)]
          )
