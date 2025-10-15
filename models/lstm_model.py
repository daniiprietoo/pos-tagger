from keras import Sequential
from keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    TimeDistributed,
    Input,
    Dropout,
)
from keras.optimizers import Adam
from .base_model import BaseModel, ModelConfig

class LSTMModel(BaseModel):

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None

    def build_model(self):
        model = Sequential()

        model.add(Input(shape=(self.config.max_sequence_length,)))

        model.add(
            Embedding(
                input_dim=self.config.vocab_size,
                output_dim=self.config.embedding_dim,
                input_length=self.config.max_sequence_length,
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
        model.add(Dropout(self.config.dropout_rate))
        model.add(TimeDistributed(Dense(self.config.num_tags, activation="softmax")))

        self.model = model
        return self.model
    
    def compile_model(self):
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")

        adam = Adam(learning_rate=self.config.training_config.learning_rate)
        self.model.compile(
            optimizer=adam,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def get_model(self) -> Sequential:
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model
