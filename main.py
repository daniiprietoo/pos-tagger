from utils import download_datasets
from conllu import parse_incr
from models.base_model import ModelConfig
from data.preprocessor import DataPreprocessor, DataPreprocessorConfig
from trainer.trainer import TrainerConfig, Trainer
from models.lstm_model import LSTMModel
from evaluator.evaluator import Evaluator
from inference.predictor import Predictor
import keras

def create_config():
    preprocessor_config = DataPreprocessorConfig(
        max_sequence_length=100,
        padding_type="post",
        truncation_type="post",
        remove_long_sentences=True,
    )

    training_config = TrainerConfig(
        epochs=20,
        batch_size=128,
        early_stopping_patience=3,
        learning_rate=5e-4,
        model_dir="saved_models",
        save_best_only=True,
    )

    model_config = ModelConfig(
        embedding_dim=100,
        lstm_units=64,
        max_sequence_length=100,
        bidirectional=True,
        dropout_rate=0.5,
        training_config=training_config,
    )

    return preprocessor_config, model_config, training_config

def load_data():
    dev_path, train_path, test_path = download_datasets()

    dev_data = list(parse_incr(open(dev_path, "r", encoding="utf-8")))
    train_data = list(parse_incr(open(train_path, "r", encoding="utf-8")))
    test_data = list(parse_incr(open(test_path, "r", encoding="utf-8")))

    return train_data, dev_data, test_data

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
    train_data = preprocessor.select_columns(train_data)
    dev_data = preprocessor.select_columns(dev_data)
    test_data = preprocessor.select_columns(test_data)

    print("Separating words and tags...\n")
    train_sentences, train_tags = preprocessor.separate_words_and_tags(train_data)
    dev_sentences, dev_tags = preprocessor.separate_words_and_tags(dev_data)
    test_sentences, test_tags = preprocessor.separate_words_and_tags(test_data)

    print("Creating vocabulary and tag mapping...\n")
    vocabulary = preprocessor.create_text_vectorizer(train_sentences)
    tag2idx, idx2tag = preprocessor.create_tag_mapping(train_tags)
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Number of tags: {len(tag2idx)}\n")

    model_config.vocab_size = len(vocabulary) + 1
    model_config.num_tags = len(tag2idx) + 1  # +1 for padding

    # Vectorize and pad the data
    print("Vectorizing and padding data...\n")
    X_train, y_train = preprocessor.vectorize_pad_data(train_sentences, train_tags)
    X_dev, y_dev = preprocessor.vectorize_pad_data(dev_sentences, dev_tags)
    X_test, y_test = preprocessor.vectorize_pad_data(test_sentences, test_tags)

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

    # # Evaluate model
    # model = keras.models.load_model("saved_models/best_model.h5")
    # print("Evaluating model...")
    evaluator = Evaluator(model.get_model(), preprocessor)

    test_metrics = evaluator.evaluate(X_test, y_test, "Test")

    # Create predictor for inference
    predictor = Predictor(model.get_model(), preprocessor)

    # Example prediction
    example_sentence = "Google is a nice search engine . Do I like it ? Yes , I am happy to hear more about it ."
    predicted_tags = predictor.predict_sentence(example_sentence)
    print(f"\nExample prediction:")
    print(f"Sentence: {example_sentence}")
    print(f"Predicted tags: {' '.join(predicted_tags)}")


if __name__ == "__main__":
    main()
