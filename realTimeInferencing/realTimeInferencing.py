import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
import time
import queue
import threading

q = queue.Queue()
predicted_label = queue.Queue()
label_event = threading.Event()
end_frame = 0

def extract_features_from_frame(frame, model):
    img = cv2.resize(frame, (299, 299))
    img = img[:, :, ::-1]  
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def load_pca(pca_mean_path, pca_eigenvals_path, pca_eigenvecs_path):
    pca_mean = np.load(pca_mean_path)[:, 0]
    pca_eigenvals = np.load(pca_eigenvals_path)[:1024, 0]
    pca_eigenvecs = np.load(pca_eigenvecs_path).T[:, :1024]
    return pca_mean, pca_eigenvals, pca_eigenvecs

def apply_pca(features, pca_mean, pca_eigenvals, pca_eigenvecs):
    features = features - pca_mean
    features = features.dot(pca_eigenvecs)
    features /= np.sqrt(pca_eigenvals + 1e-4)
    return features

def quantize_features(features, num_bits=8):
    num_bins = 2 ** num_bits
    min_val = np.min(features)
    max_val = np.max(features)
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    quantized = np.digitize(features, bin_edges[:-1])
    quantized = np.clip(quantized, 0, num_bins - 1)
    return quantized

def run_inference(interpreter, input_data):
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    return output_data

def evaluate_model(interpreter, X_test, threshold=0.5):
    input_data = np.expand_dims(X_test[0], axis=0).astype(np.float32)
    predicted_output = run_inference(interpreter, input_data)
    predicted_label = (predicted_output >= threshold).astype(int)[0][0]
    return predicted_label

def display_chunk_results(target_fps=40):
    frame_delay = 1 / target_fps  
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame[:, :] = 0  
        current_time = time.time()
        elapsed_time = current_time - prev_time
        sleep_time = frame_delay - elapsed_time
        cv2.putText(frame, f'Label: {current_label}', (frame.shape[1] - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Frame', frame)
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        prev_time = time.time()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def process_video(args):
    global predicted_label, end_frame
    model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    pca_mean_path = './pca/mean.npy'
    pca_eigenvals_path = './pca/eigenvals.npy'
    pca_eigenvecs_path = './pca/eigenvecs.npy'
    pca_mean, pca_eigenvals, pca_eigenvecs = load_pca(pca_mean_path, pca_eigenvals_path, pca_eigenvecs_path)

    cap = cv2.VideoCapture(args.video_path)
    chunk_size = 150
    frame_count = 0
    
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
            # print(q.qsize())
            if frame_count % interval_frames == 0:
                features = extract_features_from_frame(frame, model)
                frame_features_list.append(features)
                frames_in_chunk += 1

            frame_count += 1
        
        if frames_in_chunk < chunk_size:
            break
        
        features = np.array(frame_features_list)
        reduced_features = apply_pca(features, pca_mean, pca_eigenvals, pca_eigenvecs)
        final_features = quantize_features(reduced_features)
        final_features = final_features / 255
        final_features = np.reshape(final_features, (1, 150, 1024))
        predicted_label.put(evaluate_model(interpreter, final_features))
        end_frame = frame_count - 1
        label_event.set()
        print(f"Inferencing on frames {start_frame} to {end_frame} done. Prediction: {predicted_label.queue[-1]}")
    
    cap.release()
    label_event.set()

def main():
    parser = argparse.ArgumentParser(description='Process a video and classify frames.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the TFLite model file')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file')
    args = parser.parse_args()
    thread = threading.Thread(target=process_video, args=(args,))
    thread.start()
    display_chunk_results()
    thread.join()

if __name__ == "__main__":
    main()