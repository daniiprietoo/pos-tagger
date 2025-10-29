from utils import download_datasets_english
from conllu import parse_incr
from data.preprocessor import DataPreprocessor, DataPreprocessorConfig
from trainer.trainer import TrainerConfig, Trainer
from models.lstm_model import LSTMModel
from evaluator.evaluator import Evaluator
from inference.predictor import Predictor

def create_config():
    preprocessor_config = DataPreprocessorConfig(
        max_sequence_length=100,
        padding_type="post",
        truncation_type="post",
        remove_long_sentences=True,
    )

    training_config = TrainerConfig(
        epochs=20,
        batch_size=64,
        early_stopping_patience=7,
        learning_rate=1e-4,
        model_dir="saved_models",
        save_best_only=True,
    )

    model_config = ModelConfig(
        embedding_dim=128,
        lstm_units=128,
        max_sequence_length=100,
        bidirectional=True,
        dropout_rate=0.3,
        training_config=training_config,
    )

    return preprocessor_config, model_config, training_config

def main():
    # Create configurations
    print("Creating configurations...\n")
    preprocessor_config, model_config, training_config = create_config()

    # Load data
    print("Loading data...\n")
    train_data, dev_data, test_data = load_data()

    # Initialize preprocessor
    preprocessor = DataPreprocessor(preprocessor_config)

    # Preprocess data
    print("Preprocessing data...\n")

    X_train, y_train = preprocessor.process_data_to_pad_sequences(train_data, is_train_dataset=True, model_config=model_config)
    X_dev, y_dev = preprocessor.process_data_to_pad_sequences(dev_data, is_train_dataset=False, model_config=model_config)
    X_test, y_test = preprocessor.process_data_to_pad_sequences(test_data, is_train_dataset=False, model_config=model_config)

    print("Initializing and building model...\n")
    # Initialize and build model
    model = LSTMModel(model_config)
    model.build_model()
    model.compile_model()

    print("Model summary:\n")
    print(model.get_model().summary())
    # Initialize trainer
    trainer = Trainer(training_config, model, preprocessor)

    # Train the model
    print("Training model...\n")
    trainer.train((X_train, y_train), (X_dev, y_dev))
    print("Training completed.\n")

    evaluator = Evaluator(model.get_model(), preprocessor)

    test_metrics = evaluator.evaluate(X_test, y_test, "Test")

    # Create predictor for inference
    predictor = Predictor(model.get_model(), preprocessor)

    # Example prediction
    example_sentence = "Google es un buen motor de busqueda ."
    predicted_tags = predictor.predict_sentence(example_sentence)
    print(f"\nExample prediction:")
    print(f"Sentence: {example_sentence}")
    print(f"Predicted tags: {' '.join(predicted_tags)}")


if __name__ == "__main__":
    main()
