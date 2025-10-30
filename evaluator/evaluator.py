import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from models.base_model import BaseModel
from data.preprocessor import DataPreprocessor

class Evaluator:
    """Handles model evaluation"""

    def __init__(self, model: BaseModel, preprocessor: DataPreprocessor):
        self.model = model.get_model()
        self.preprocessor = preprocessor

    def evaluate(self, x_data, y_data, dataset_name: str = ""):

        predictions = self.model.predict(x_data, verbose='0')
        predicted_classes = np.argmax(predictions, axis=-1)

        # Flatten for evaluation (excluding padding)
        y_true_flat = []
        y_pred_flat = []

        for i in range(len(y_data)):
            for j in range(len(y_data[i])):
                if y_data[i][j] != 0:  # Exclude padding
                    y_true_flat.append(y_data[i][j])
                    y_pred_flat.append(predicted_classes[i][j])

        # Calculate metrics
        accuracy = accuracy_score(y_true_flat, y_pred_flat)

        # Convert indices back to tag names for detailed report
        y_true_tags = [self.preprocessor.idx2tag.get(idx, "[UNK]") for idx in y_true_flat]
        y_pred_tags = [self.preprocessor.idx2tag.get(idx, "[UNK]") for idx in y_pred_flat]

        print(f"\n{dataset_name} Set Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_true_tags, y_pred_tags, zero_division=0))

        return {"accuracy": accuracy, "y_true": y_true_tags, "y_pred": y_pred_tags}
