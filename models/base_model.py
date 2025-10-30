from abc import ABC, abstractmethod
from keras.models import Sequential
from trainer.trainer import TrainerConfig

class ModelConfig:

    def __init__(
        self,
        training_config: TrainerConfig,
        embedding_dim: int = 100,
        lstm_units: int = 128,
        dropout_rate: float = 0.5,
        bidirectional: bool = True,
        lstm_layers: int = 1
    ) -> None:
        self.embedding_dim: int = embedding_dim
        self.lstm_units: int = lstm_units
        self.dropout_rate: float = dropout_rate
        self.bidirectional: bool = bidirectional
        self.training_config: TrainerConfig = training_config
        self.lstm_layers: int = lstm_layers

class BaseModel(ABC):

    def __init__(self, config: ModelConfig):
        self.config = config
        self.vocab_size = 0  # Will be set after vocabulary creation
        self.num_tags = 0  # Will be set after tag mapping creation

    @abstractmethod
    def build_model(self) -> Sequential:
        pass

    @abstractmethod
    def compile_model(self):
        pass

    @abstractmethod
    def get_model(self) -> Sequential:
        pass
