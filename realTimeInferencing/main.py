import argparse
import cv2
import time
from FeatureExtraction import FeatureExtractor

def main():
    """
    Main function to process video and display predictions with FPS.
    """
    parser = argparse.ArgumentParser(description='Process a video and classify frames.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the TFLite model file')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file')
    args = parser.parse_args()

    ptime1 = time.time()
    video_processor = FeatureExtractor(model_path=args.model_path, video_path=args.video_path)
    prediction_ranges = video_processor.process_video()
    ctime1 = time.time()
    
    feature_extraction_time = ctime1 - ptime1
    print(f'Total time taken for feature extraction is {feature_extraction_time:.2f} seconds')
    for start_frame, end_frame, label in prediction_ranges:
        print(f"Frames {start_frame} to {end_frame}: Label = {label}")
    cap = cv2.VideoCapture(args.video_path)
    frame_count = 0
    current_label = None
    label_idx = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        fps = int(1 / (current_time - prev_time))
        prev_time = current_time
        
        if label_idx < len(prediction_ranges):
            start_frame, end_frame, label = prediction_ranges[label_idx]
            if frame_count >= start_frame and frame_count <= end_frame:
                current_label = label
            elif frame_count > end_frame:
                label_idx += 1
                if label_idx < len(prediction_ranges):
                    start_frame, end_frame, label = prediction_ranges[label_idx]
                    current_label = label
        
        cv2.putText(frame, f'FPS: {fps}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if current_label is not None:
            cv2.putText(frame, f'Label: {current_label}', (frame.shape[1] - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()