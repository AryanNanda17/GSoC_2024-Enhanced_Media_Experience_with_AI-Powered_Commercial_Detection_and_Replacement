import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image

class FeatureExtractor:
    """
    A class to process video frames and classify them using a TFLite model.

    Parameters
    ----------
    model_path : str
        Path to the TFLite model file.
    video_path : str
        Path to the video file.
    """

    def __init__(self, model_path, video_path):
        """
        Initializes the VideoProcessor with model and video paths.
        
        Loads the InceptionV3 model, the TFLite model, and PCA parameters.
        """
        self.pca_mean_path = '../pca/mean.npy'
        self.pca_eigenvals_path = '../pca/eigenvals.npy'
        self.pca_eigenvecs_path = '../pca/eigenvecs.npy'
        
        self.model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.pca_mean, self.pca_eigenvals, self.pca_eigenvecs = self.load_pca()

        self.video_path = video_path

    def extract_features_from_frame(self, frame):
        """
        Extracts features from a single frame using the InceptionV3 model.

        Parameters
        ----------
        frame : numpy.ndarray
            The video frame from which to extract features.

        Returns
        -------
        numpy.ndarray
            Flattened features extracted from the frame.
        """
        img = cv2.resize(frame, (299, 299))
        img = img[:, :, ::-1]
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.model.predict(x)
        return features.flatten()

    def process_video(self):
        """
        Processes the video file and performs inference on each chunk of frames.

        Returns
        -------
        list of tuples
            Each tuple contains (start_frame, end_frame, predicted_label).
        """
        cap = cv2.VideoCapture(self.video_path)
        chunk_size = 150
        frame_count = 0
        all_final_features = []
        prediction_ranges = []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval_frames = int(fps * (250 / 1000))

        while True:
            frame_features_list = []
            frames_in_chunk = 0
            start_frame = frame_count

            while frames_in_chunk < chunk_size:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % interval_frames == 0:
                    features = self.extract_features_from_frame(frame)
                    frame_features_list.append(features)
                    frames_in_chunk += 1

                frame_count += 1

            if frames_in_chunk < chunk_size:
                break

            features = np.array(frame_features_list)
            reduced_features = self.apply_pca(features)
            final_features = self.quantize_features(reduced_features)
            final_features = final_features / 255
            final_features = np.reshape(final_features, (1, 150, 1024))
            predicted_label = self.evaluate_model(final_features)
            all_final_features.append(predicted_label)
            
            end_frame = frame_count - 1
            prediction_ranges.append((start_frame, end_frame, predicted_label))
            print(prediction_ranges)
            print(f"Inferencing on frames {start_frame} to {end_frame} done. Prediction: {predicted_label}")

        cap.release()
        return prediction_ranges

    def load_pca(self):
        """
        Loads PCA parameters from file.

        Returns
        -------
        tuple of numpy.ndarray
            PCA mean, eigenvalues, and eigenvectors.
        """
        pca_mean = np.load(self.pca_mean_path)[:, 0]
        pca_eigenvals = np.load(self.pca_eigenvals_path)[:1024, 0]
        pca_eigenvecs = np.load(self.pca_eigenvecs_path).T[:, :1024]
        return pca_mean, pca_eigenvals, pca_eigenvecs

    def apply_pca(self, features):
        """
        Applies PCA transformation to the features.

        Parameters
        ----------
        features : numpy.ndarray
            Features to be transformed.

        Returns
        -------
        numpy.ndarray
            Transformed features.
        """
        features = features - self.pca_mean
        features = features.dot(self.pca_eigenvecs)
        features /= np.sqrt(self.pca_eigenvals + 1e-4)
        return features

    def quantize_features(self, features, num_bits=8):
        """
        Quantizes the features to a specified number of bits.

        Parameters
        ----------
        features : numpy.ndarray
            Features to be quantized.
        num_bits : int
            Number of bits for quantization.

        Returns
        -------
        numpy.ndarray
            Quantized features.
        """
        num_bins = 2 ** num_bits
        min_val = np.min(features)
        max_val = np.max(features)
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        quantized = np.digitize(features, bin_edges[:-1])
        quantized = np.clip(quantized, 0, num_bins - 1)
        return quantized

    def run_inference(self, input_data):
        """
        Runs inference on the provided input data using the TFLite model.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data for the model.

        Returns
        -------
        numpy.ndarray
            Model output.
        """
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

    def evaluate_model(self, X_test, threshold=0.5):
        """
        Evaluates the model on the test data and applies a threshold.

        Parameters
        ----------
        X_test : numpy.ndarray
            Test data.
        threshold : float
            Threshold for classification.

        Returns
        -------
        int
            Predicted label after thresholding.
        """
        input_data = np.expand_dims(X_test[0], axis=0).astype(np.float32)
        predicted_output = self.run_inference(input_data)
        predicted_label = (predicted_output >= threshold).astype(int)[0][0]
        return predicted_label