import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed, RepeatVector

import babel_data
import data

def all_acc(y_true, y_pred):
    return K.mean(
        K.all(
            K.equal(
                K.max(y_true, axis=-1),
                K.cast(K.argmax(y_pred, axis=-1), K.floatx())
            ),
            axis=1)
    )


print("Creating data...")
babel_data.create("train", 1000000, True)
babel_data.create("dev", 100000, False)

print("Loading data...")
train = data.ParallelReader("train-source.txt", "source-vocab.txt", "", "train-target.txt", "target-vocab.txt", "")
dev = data.ParallelReader("dev-source.txt", "source-vocab.txt", "", "dev-target.txt", "target-vocab.txt", "")

print("Building model...")
# Encoder
source = Input(shape=(None,), dtype='int32', name='source')
embedded = Embedding(output_dim=128, input_dim=train.source_vocab_size(), mask_zero=True)(source)
last_hid = Bidirectional(LSTM(output_dim=128))(embedded)

# Decoder
repeated = RepeatVector(train.target.padded.shape[1])(last_hid)
decoder = LSTM(output_dim=128, return_sequences=True, name="decoder1")(repeated)
decoder = LSTM(output_dim=128, return_sequences=True, name="decoder2")(decoder)
output = TimeDistributed(Dense(output_dim=train.target_vocab_size(), activation='softmax'))(decoder)
model = Model([source], output=[output])

print("Compiling model...")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[all_acc])

print("Training...")
train_sources, train_targets = train.all()
dev_sources, dev_targets = dev.all()
model.fit(train_sources, train_targets[:, :, None],
          nb_epoch=50,
          validation_data=(dev_sources, dev_targets[:, :, None]),
          callbacks=[
              ModelCheckpoint("model.{epoch:02d}-{val_loss:.2f}.hdf5",
                              monitor='val_loss',
                              verbose=0,
                              save_best_only=True,
                              save_weights_only=False,
                              mode='auto')
          ])