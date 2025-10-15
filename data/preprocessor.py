from keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences
from models.base_model import ModelConfig

class DataPreprocessorConfig:
    def __init__(
        self, max_sequence_length, padding_type, truncation_type, remove_long_sentences
    ) -> None:
        self.max_sequence_length = max_sequence_length
        self.padding_type = padding_type
        self.truncation_type = truncation_type
        self.remove_long_sentences = remove_long_sentences


class DataPreprocessor:
    def __init__(self, config: DataPreprocessorConfig):
        self.text_vectorizer = None
        self.tag2idx = {}
        self.idx2tag = {}
        self.config = config


    def select_columns(self, data):
        """Extract word and upos columns from conllu data, filtering invalid tags"""
        preprocessed_data = []

        for sentence in data:
            sentence_data = []
            for token in sentence:
                word = token.get("form", "")
                tag = token.get("upos", "")

                # Skip tokens with missing or invalid tags
                if tag and tag != "_" and word and word != "_":
                    sentence_data.append([word, tag])

            # Only include sentences with at least one valid token
            if sentence_data:
                preprocessed_data.append(sentence_data)

        return preprocessed_data

    def separate_words_and_tags(self, data):
        """
        Separate words and tags.
        """

        sentences = [" ".join([word for word, tag in sentence]) for sentence in data]
        tags = [" ".join([tag for word, tag in sentence]) for sentence in data]
        return sentences, tags

    def create_text_vectorizer(self, train_sentences):
        """
        Create a text vectorizer and fit it on the data
        """
        self.text_vectorizer = TextVectorization()
        self.text_vectorizer.adapt(train_sentences)

        return self.text_vectorizer.get_vocabulary()

    def create_tag_mapping(self, train_tags):
        """
        Map tags to indices, for the training data
        """

        tag_set = set()
        for sequence in train_tags:
            # each sequence is a space-separated string of tags
            for tag in sequence.split():
                tag_set.add(tag)
        # reserve 0 for padding
        self.tag2idx = {tag: idx + 1 for idx, tag in enumerate(sorted(tag_set))}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        return self.tag2idx, self.idx2tag

    def vectorize_pad_data(self, sentences, tags):
        """
        Convert sentences and tags to vectors, pad them
        """

        if self.text_vectorizer is None:
            raise ValueError(
                "Text vectorizer not initialized. Call create_text_vectorizer first."
            )

        print(f"Sample sentences: {sentences[:3]}")
        print(f"Sample tags: {tags[:3]}")

        # Vectorize text
        vectorized_text = self.text_vectorizer(sentences)
        padded_text = pad_sequences(
            vectorized_text,
            maxlen=self.config.max_sequence_length,
            padding=self.config.padding_type,
            truncating=self.config.truncation_type,
        )

        padded_tags = None
        if tags is not None:
            # Convert tags to indices and pad
            tag_sequences = [[self.tag2idx[tag] for tag in seq.split()] for seq in tags]
            padded_tags = pad_sequences(
                tag_sequences,
                maxlen=self.config.max_sequence_length,
                padding=self.config.padding_type,
                truncating=self.config.truncation_type,
            )

        print(f"Sample vectorized sentences shape: {padded_text[:3].shape}")
        print(f"Sample vectorized tags: {padded_tags[:3]}")

        return padded_text, padded_tags

    def process_data_to_pad_sequences(self, data, is_train_dataset, model_config: ModelConfig):

        relevant_data = self.select_columns(data)

        sentences, tags = self.separate_words_and_tags(relevant_data)

        if is_train_dataset:
            vocabulary = self.create_text_vectorizer(sentences)
            self.create_tag_mapping(tags)

            model_config.vocab_size = len(vocabulary) + 1
            model_config.num_tags = len(self.tag2idx) + 1

        padded_text, padded_tags = self.vectorize_pad_data(sentences, tags)

        return padded_text, padded_tags

    def vectorize_single_sentence(self, sentence: str):
        """
        Vectorize and pad a single sentence for inference
        """
        if self.text_vectorizer is None:
            raise ValueError(
                "Text vectorizer not initialized. Call create_text_vectorizer first."
            )

        vectorized_text = self.text_vectorizer([sentence])
        padded_text = pad_sequences(
            vectorized_text,
            maxlen=self.config.max_sequence_length,
            padding=self.config.padding_type,
            truncating=self.config.truncation_type,
        )

        return padded_text, len(sentence.split())  # Return original length for accurate tag extraction
    
