import torch
import numpy as np
import torch_directml
from Model_Training.the_model import get_student_model


class EEGPredictor:
    def __init__(self, model_path, input_size=18, num_classes=19):
        self.device = self._get_device()
        self.model = self._load_model(model_path, input_size, num_classes)
        self.model.to(self.device)
        self.model.eval()

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")  # For Mac M1/M2/M3
        elif torch_directml.is_available():
            return torch_directml.device()  # For Windows DirectML
        else:
            return torch.device("cpu")  # Default to CPU

    def _load_model(self, model_path, input_size, num_classes):
        model = get_student_model(input_size=input_size, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
        return model

    def predict(self, input_data):
        """
        Make predictions on input data.

        Args:
            input_data: Can be either:
                - A numpy array of shape (18,) for single prediction
                - A numpy array of shape (n, 18) for batch prediction
                - A list of 18 numbers

        Returns:
            predicted_class: The predicted class index
            confidence: Confidence score for the prediction
        """
        # Convert input to tensor if it's not already
        if isinstance(input_data, list):
            input_data = np.array(input_data)

        if input_data.ndim == 1:
            # Add batch dimension if single sample
            input_data = input_data.reshape(1, -1)

        # Convert to tensor
        input_tensor = torch.FloatTensor(input_data).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            confidence_scores = torch.max(probabilities, dim=1).values

        # Convert to numpy for easier handling
        predictions = predicted_classes.cpu().numpy()
        confidences = confidence_scores.cpu().numpy()

        if len(predictions) == 1:
            return predictions[0], confidences[0]
        return predictions, confidences


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = EEGPredictor(
        model_path='./models/first_model.pth',
        input_size=18,
        num_classes=19
    )

    # Example of single prediction
    single_sample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                     1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    predicted_class, confidence = predictor.predict(single_sample)
    print(f"Single prediction: Class {predicted_class} with confidence {confidence:.2f}")

    # Example of batch prediction
    batch_samples = np.random.rand(5, 18)  # 5 samples with 18 features each
    predictions, confidences = predictor.predict(batch_samples)
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        print(f"Sample {i + 1}: Class {pred} with confidence {conf:.2f}")