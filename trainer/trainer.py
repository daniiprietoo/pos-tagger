import os
from keras.callbacks import EarlyStopping, ModelCheckpoint


class TrainerConfig:
    def __init__(
        self,
        epochs=5,
        batch_size=32,
        early_stopping_patience=5,
        model_dir="saved_models",
        learning_rate=0.003,
        save_best_only=True,
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_dir = model_dir
        self.save_best_only = save_best_only

class Trainer:

    def __init__(self, config, model, preprocessor):
        self.config = config
        self.model = model
        self.preprocessor = preprocessor

    def callbacks(self, model_dir, language):
        """
        Define the callbacks for training
        """

        callbacks = []

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stopping)

        if model_dir:
            checkpoint = ModelCheckpoint(
                filepath=os.path.join(
                    model_dir,
                    f"{str(self.model)}_{language}.keras",
                ),
                monitor="val_loss",
                save_best_only=self.config.save_best_only,
                verbose=1,
            )
            callbacks.append(checkpoint)

        return callbacks

    def train(self, train_data, val_data, language):
        """
        Train the model
        """
        x_train, y_train = train_data
        x_val, y_val = val_data

        self.model.build_model()
        self.model.compile_model()

        history = self.model.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=self.callbacks(self.config.model_dir, language),
        )

        return history
