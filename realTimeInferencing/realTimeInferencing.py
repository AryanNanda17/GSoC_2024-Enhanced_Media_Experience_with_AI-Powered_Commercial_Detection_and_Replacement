import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
import time
import queue
import threading

# Initialize queues for frames and predictions, and an event to signal when a label is available.
q = queue.Queue()
predicted_label = queue.Queue()
label_event = threading.Event()
end_frame = 0

def extract_features_from_frame(frame, model):
    """
    Extract features from a single frame using the InceptionV3 model.

    Parameters:
    - frame (numpy array): The image frame from which to extract features.
    - model (Keras Model): The pre-trained InceptionV3 model for feature extraction.

    Returns:
    - numpy array: Flattened feature vector.
    """
    img = cv2.resize(frame, (299, 299))
    img = img[:, :, ::-1]  # Convert BGR to RGB
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def load_pca(pca_mean_path, pca_eigenvals_path, pca_eigenvecs_path):
    """
    Load PCA parameters from saved numpy files.

    Parameters:
    - pca_mean_path (str): Path to the PCA mean vector file.
    - pca_eigenvals_path (str): Path to the PCA eigenvalues file.
    - pca_eigenvecs_path (str): Path to the PCA eigenvectors file.

    Returns:
    - tuple: PCA mean, eigenvalues, and eigenvectors.
    """
    pca_mean = np.load(pca_mean_path)[:, 0]
    pca_eigenvals = np.load(pca_eigenvals_path)[:1024, 0]
    pca_eigenvecs = np.load(pca_eigenvecs_path).T[:, :1024]
    return pca_mean, pca_eigenvals, pca_eigenvecs

def apply_pca(features, pca_mean, pca_eigenvals, pca_eigenvecs):
    """
    Apply PCA to reduce the dimensionality of the feature vectors.

    Parameters:
    - features (numpy array): The feature vector to be reduced.
    - pca_mean (numpy array): The PCA mean vector.
    - pca_eigenvals (numpy array): The PCA eigenvalues.
    - pca_eigenvecs (numpy array): The PCA eigenvectors.

    Returns:
    - numpy array: The PCA-reduced feature vector.
    """
    features = features - pca_mean
    features = features.dot(pca_eigenvecs)
    features /= np.sqrt(pca_eigenvals + 1e-4)
    return features

def quantize_features(features, num_bits=8):
    """
    Quantize the PCA-reduced features to a lower bit representation.

    Parameters:
    - features (numpy array): The PCA-reduced feature vector.
    - num_bits (int): The number of bits for quantization. Default is 8.

    Returns:
    - numpy array: The quantized feature vector.
    """
    num_bins = 2 ** num_bits
    min_val = np.min(features)
    max_val = np.max(features)
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    quantized = np.digitize(features, bin_edges[:-1])
    quantized = np.clip(quantized, 0, num_bins - 1)
    return quantized

def run_inference(interpreter, input_data):
    """
    Run inference using a TensorFlow Lite interpreter.

    Parameters:
    - interpreter (tf.lite.Interpreter): The TensorFlow Lite interpreter.
    - input_data (numpy array): The input data for the model.

    Returns:
    - numpy array: The output from the model after inference.
    """
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    return output_data

def evaluate_model(interpreter, X_test, threshold=0.5):
    """
    Evaluate the model on the test data to predict the label.

    Parameters:
    - interpreter (tf.lite.Interpreter): The TensorFlow Lite interpreter.
    - X_test (numpy array): The test data for model evaluation.
    - threshold (float): The threshold for binary classification. Default is 0.5.

    Returns:
    - int: The predicted label (0 or 1).
    """
    input_data = np.expand_dims(X_test[0], axis=0).astype(np.float32)
    predicted_output = run_inference(interpreter, input_data)
    predicted_label = (predicted_output >= threshold).astype(int)[0][0]
    return predicted_label

def display_chunk_results(target_fps=40):
    """
    Display the video frames with classification results.

    Parameters:
    - target_fps (int): The target frames per second for display. Default is 40.
    """
    frame_delay = 1 / target_fps  # Calculate delay between frames
    prev_time = time.time()
    frameCount = 0
    flag = 0
    global end_frame
    end_frame1 = end_frame
    while True:
        if not label_event.is_set():
            continue
        if q.empty():
            break
        frameCount += 1
        if flag == 0 or frameCount >= end_frame1:
            current_label = predicted_label.get()
            flag = 1
            end_frame1 = end_frame
        frame = q.get()
        if current_label == 1:
            # Apply effect based on prediction
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame[:, :] = 0  # Set frame to black
        current_time = time.time()
        elapsed_time = current_time - prev_time
        sleep_time = frame_delay - elapsed_time
        if current_label == 0:
            cv2.putText(frame, f'Label: Content', (frame.shape[1] - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif current_label == 1:
            text = 'Commercial'
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            x_position = (frame.shape[1] - text_width) // 2
            y_position = (frame.shape[0] + text_height) // 2
            cv2.putText(frame, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Frame', frame)
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        prev_time = time.time()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def process_video(args):
    """
    Process the input video, extract features, apply PCA, quantize, and classify frames.

    Parameters:
    - args (argparse.Namespace): Parsed command-line arguments.
    """
    global predicted_label, end_frame
    model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    
    # Load PCA components
    pca_mean_path = './pca/mean.npy'
    pca_eigenvals_path = './pca/eigenvals.npy'
    pca_eigenvecs_path = './pca/eigenvecs.npy'
    pca_mean, pca_eigenvals, pca_eigenvecs = load_pca(pca_mean_path, pca_eigenvals_path, pca_eigenvecs_path)

    cap = cv2.VideoCapture(args.video_path)
    chunk_size = 150
    frame_count = 0
    
    # Determine frame interval for feature extraction
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
            q.put(frame)
            # Extract features at regular intervals
            if frame_count % interval_frames == 0:
                features = extract_features_from_frame(frame, model)
                frame_features_list.append(features)
                frames_in_chunk += 1

            frame_count += 1
        
        if frames_in_chunk < chunk_size:
            break
        
        # Apply PCA and quantization to extracted features
        features = np.array(frame_features_list)
        reduced_features = apply_pca(features, pca_mean, pca_eigenvals, pca_eigenvecs)
        final_features = quantize_features(reduced_features)
        final_features = final_features / 255
        final_features = np.reshape(final_features, (1, 150, 1024))
        
        # Classify and store the predicted label
        predicted_label.put(evaluate_model(interpreter, final_features))
        end_frame = frame_count - 1
        label_event.set()
        print(f"Inferencing on frames {start_frame} to {end_frame} done. Prediction: {predicted_label.queue[-1]}")
    
    cap.release()
    label_event.set()

def main():
    """
        Main function to parse command-line arguments, start video processing and display results.
    """
    parser = argparse.ArgumentParser(description='Process a video and classify frames.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the TFLite model file')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file')
    args = parser.parse_args()
    # Start thread for video processing
    thread = threading.Thread(target=process_video, args=(args,))
    thread.start()
    # Start displaying the video 
    display_chunk_results()
    # Wait for the thread to finish
    thread.join()

if __name__ == '__main__':
    main()