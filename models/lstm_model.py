from keras import Sequential
import tensorflow as tf

from keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    TimeDistributed,
    Input,
)
from keras.optimizers import Adam
from .base_model import BaseModel, ModelConfig


class LSTMModel(BaseModel):

    def __init__(
        self,
        config: ModelConfig,
        vocab_size: int,
        num_tags: int,
        max_sequence_length: int,
    ):
        super().__init__(config)
        self.model = None
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.max_sequence_length = max_sequence_length

    def build_model(self):
        model = Sequential()

        model.add(Input(shape=(self.max_sequence_length,)))

        model.add(
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.config.embedding_dim,
                mask_zero=True,
            )
        )

        if self.config.bidirectional:
            model.add(
                Bidirectional(
                    LSTM(
                        units=self.config.lstm_units,
                        return_sequences=True,
                        dropout=self.config.dropout_rate,
                    )
                )
            )
        else:
            model.add(
                LSTM(
                    units=self.config.lstm_units,
                    return_sequences=True,
                    dropout=self.config.dropout_rate,
                )
            )
        model.add(TimeDistributed(Dense(self.num_tags, activation="softmax")))

        self.model = model
        return self.model

    def compile_model(self):
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")

        adam = Adam(learning_rate=self.config.training_config.learning_rate)
        self.model.compile(
            optimizer=adam,
            loss="sparse_categorical_crossentropy",
            metrics=[self._masked_accuracy],
        )

    def get_model(self) -> Sequential:
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model

    def _masked_accuracy(self, y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)

        y_true = tf.cast(y_true, tf.int64)
        y_pred = tf.argmax(y_pred, axis=-1)

        matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        masked_acc = tf.reduce_sum(matches * mask) / tf.reduce_sum(mask)

        return masked_acc

    def __str__(self):
        if self.config.bidirectional:
            direction = "BiLSTM"
        else:
            direction = "LSTM"

        # Convert floats to cleaner integer representations
        dropout_rate_int = int(self.config.dropout_rate * 100)
        learning_rate_int = int(self.config.training_config.learning_rate * 10000)

        return (
            f"{direction}_emb{self.config.embedding_dim}"
            f"_lstm{self.config.lstm_units}"
            f"_drop{dropout_rate_int}"
            f"_ep{self.config.training_config.epochs}"
            f"_bs{self.config.training_config.batch_size}"
            f"_pat{self.config.training_config.early_stopping_patience}"
            f"_lr{learning_rate_int}"
        )
