from keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences
from models.base_model import BaseModel


class DataPreprocessorConfig:
    def __init__(
        self,
        padding_type,
        truncation_type,
        remove_long_sentences,
        max_sequence_length=100,
    ) -> None:
        self.padding_type = padding_type
        self.truncation_type = truncation_type
        self.remove_long_sentences = remove_long_sentences
        self.max_sequence_length = max_sequence_length


class DataPreprocessor:
    def __init__(self, config: DataPreprocessorConfig):
        self.text_vectorizer = None
        self.tag2idx = {}
        self.idx2tag = {}
        self.config = config
        self.vocab_size = 0
        self.num_tags = 0

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
        self.text_vectorizer = TextVectorization(
            max_tokens=None,  # Use all tokens
            standardize=None,  # No standardization (preserves case and punctuation)
            split="whitespace",  # Split on whitespace only
            output_mode="int",
            output_sequence_length=None,
        )
        self.text_vectorizer.adapt(train_sentences)

        return self.text_vectorizer.get_vocabulary()

    def create_tag_mapping(self, train_tags):
        """
        Map tags to indices from the training data, including an UNK tag.
        """
        tag_set = set()
        for sequence in train_tags:
            for tag in sequence.split():
                tag_set.add(tag)

        # Reserve 0 for padding
        self.tag2idx = {tag: idx + 1 for idx, tag in enumerate(sorted(tag_set))}
        # Add a specific index for unknown tags
        self.tag2idx["[UNK]"] = len(self.tag2idx) + 1

        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        return self.tag2idx, self.idx2tag

    def vectorize_pad_data(self, sentences, tags):
        """
        Convert sentences and tags to vectors, pad them, and handle unknown tags.
        """
        if self.text_vectorizer is None:
            raise ValueError(
                "Text vectorizer not initialized. Call create_text_vectorizer first."
            )

        # print(f"Sample sentence: {sentences[0]}")
        # print(f"Sample tags: {tags[0]}")

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
            # Use .get() to handle tags that are in dev/test but not in train
            unk_tag_idx = self.tag2idx["[UNK]"]
            tag_sequences = [
                [self.tag2idx.get(tag, unk_tag_idx) for tag in seq.split()]
                for seq in tags
            ]
            padded_tags = pad_sequences(
                tag_sequences,
                maxlen=self.config.max_sequence_length,
                padding=self.config.padding_type,
                truncating=self.config.truncation_type,
            )

        # print(f"Sample vectorized sentence shape: {padded_text[0].shape}")
        # print(f"Sample vectorized sentence: {padded_text[0]}")
        # if padded_tags is not None:
            # print(f"Sample vectorized tags: {padded_tags[0]}")

        return padded_text, padded_tags

    def process_data_to_pad_sequences(self, data, is_train_dataset):

        relevant_data = self.select_columns(data)

        sentences, tags = self.separate_words_and_tags(relevant_data)

        if is_train_dataset:
            vocabulary = self.create_text_vectorizer(sentences)
            self.create_tag_mapping(tags)

            self.vocab_size = len(vocabulary) + 1
            # The number of tags is the number of known tags + padding + UNK tag
            self.num_tags = len(self.tag2idx) + 1

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

        return padded_text, len(sentence.split())
