import torch
import numpy as np
import cv2
from torchvision import transforms
from vasnet_model import VASNet
import h5py
import os

def extract_features(video_path):
    """Extract features from video using GoogleNet/Inception"""
    # Load the video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to 224x224 (GoogleNet input size)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    
    # Convert frames to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frames = np.array(frames)
    features = []
    
    # Process frames in batches
    batch_size = 32
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        batch_features = []
        for frame in batch:
            frame_tensor = transform(frame).unsqueeze(0)
            # Here you would normally pass through GoogleNet
            # For now, we'll use a placeholder 1024-dimensional feature
            feature = np.random.randn(1024)  # Placeholder
            batch_features.append(feature)
        features.extend(batch_features)
    
    return np.array(features)

def summarize_video(video_path, model_path):
    """Generate summary for a video"""
    # Load the model
    model = VASNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Extract features
    print("Extracting features from video...")
    features = extract_features(video_path)
    
    # Prepare input
    seq = torch.from_numpy(features).float().unsqueeze(0)
    
    # Generate summary
    print("Generating summary...")
    with torch.no_grad():
        scores, _ = model(seq, seq.shape[1])
    
    # Convert scores to frame indices
    summary_indices = np.where(scores.numpy() > 0.5)[1]  # Threshold of 0.5
    
    return summary_indices

def create_summary_video(video_path, summary_indices, output_path):
    """Create a summary video from the selected frames"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in summary_indices:
            out.write(frame)
            
        frame_idx += 1
    
    cap.release()
    out.release()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate video summary")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--output", type=str, required=True, help="Path to output summary video")
    
    args = parser.parse_args()
    
    # Generate summary
    summary_indices = summarize_video(args.video, args.model)
    
    # Create summary video
    create_summary_video(args.video, summary_indices, args.output)
    
    print(f"Summary video saved to {args.output}") 