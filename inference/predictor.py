import numpy as np

class Predictor:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict_sentence(self, sentence: str):
         """Predict tags for a given sentence."""
         padded_sequence, original_length = self.preprocessor.vectorize_single_sentence(sentence)

         # Get prediction
         predictions = self.model.predict(padded_sequence, verbose=0)
         predicted_classes = np.argmax(predictions, axis=-1)

         # Convert indices to tags, using the true token length
         idx2tag = self.preprocessor.idx2tag
        
         predicted_tags = [
            idx2tag.get(predicted_classes[0][j], "[UNK]") # Using UNK for unknown tags
            for j in range(original_length)
         ]

         return predicted_tags
    
    