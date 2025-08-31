import numpy as np
import cv2
from scipy.io import loadmat
from tqdm import tqdm
import os
import pandas as pd
import glob

def detect_plane_spatial_voting_simple(x, y, ts, p, time_window=0.02, min_votes=150):
    """Simple spatial voting method with minimum vote threshold"""
    ts_sec = ts / 1e6
    width, height = 304, 240
    duration = ts_sec.max() - ts_sec.min()
    num_frames = int(duration / time_window) + 1
    
    centers = []
    
    for frame_idx in range(num_frames):
        start = ts_sec.min() + frame_idx * time_window
        end = start + time_window
        
        mask = (ts_sec >= start) & (ts_sec < end)
        fx = x[mask]
        fy = y[mask]
        fp = p[mask]
        
        if len(fx) < 5:
            centers.append(None)
            continue
        
        vote_map = np.zeros((height, width), dtype=np.float32)
        vote_radius = 25
        
        for i in range(len(fx)):
            cx = int(fx[i])
            cy = int(fy[i])
            
            y_min = max(0, cy - vote_radius)
            y_max = min(height, cy + vote_radius)
            x_min = max(0, cx - vote_radius)
            x_max = min(width, cx + vote_radius)
            
            for yy in range(y_min, y_max):
                for xx in range(x_min, x_max):
                    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
                    if dist <= vote_radius:
                        weight = np.exp(-(dist**2) / (2 * (vote_radius/3)**2))
                        vote_map[yy, xx] += weight
        
        max_votes = np.max(vote_map)
        
        if max_votes >= min_votes:
            max_loc = np.unravel_index(np.argmax(vote_map), vote_map.shape)
            max_y, max_x = max_loc
            
            region_size = 30
            y1 = max(0, max_y - region_size)
            y2 = min(height, max_y + region_size)
            x1 = max(0, max_x - region_size)
            x2 = min(width, max_x + region_size)
            
            region_votes = vote_map[y1:y2, x1:x2]
            if np.sum(region_votes) > 0:
                y_indices, x_indices = np.mgrid[y1:y2, x1:x2]
                total_weight = np.sum(region_votes)
                
                center_y = int(np.sum(y_indices * region_votes) / total_weight)
                center_x = int(np.sum(x_indices * region_votes) / total_weight)
                
                dist_to_center = np.sqrt((fx - center_x)**2 + (fy - center_y)**2)
                nearby_mask = dist_to_center <= vote_radius
                nearby_pol = fp[nearby_mask]
                
                if len(nearby_pol) > 0:
                    pos_events = np.sum(nearby_pol > 0)
                    neg_events = np.sum(nearby_pol <= 0)
                    total_events = len(nearby_pol)
                    
                    if pos_events > neg_events:
                        dom_pol = 1
                        pol_conf = pos_events / total_events
                    elif neg_events > pos_events:
                        dom_pol = -1
                        pol_conf = neg_events / total_events
                    else:
                        dom_pol = 0
                        pol_conf = 0.5
                    
                    pol_ratio = pos_events / total_events if total_events > 0 else 0.5
                else:
                    dom_pol = 0
                    pol_conf = 0.0
                    pol_ratio = 0.5
                    pos_events = 0
                    neg_events = 0
                    total_events = 0
                
                centers.append({
                    'x': center_x,
                    'y': center_y,
                    'votes': float(max_votes),
                    'frame_idx': frame_idx,
                    'timestamp': start + time_window/2,
                    'dominant_polarity': dom_pol,
                    'polarity_confidence': pol_conf,
                    'polarity_ratio': pol_ratio,
                    'positive_events': pos_events,
                    'negative_events': neg_events,
                    'total_nearby_events': total_events
                })
            else:
                centers.append(None)
        else:
            centers.append(None)
    
    return centers

def save_all_frames(x, y, ts, p, centers, output_dir, basename, time_window=0.02):
    """Save ALL frames for a single .mat file"""
    ts_sec = ts / 1e6
    width, height = 304, 240
    
    saved_count = 0
    detection_count = 0
    
    for frame_idx, center in enumerate(centers):
        start = ts_sec.min() + frame_idx * time_window
        end = start + time_window
        
        mask = (ts_sec >= start) & (ts_sec < end)
        fx = x[mask]
        fy = y[mask]
        fp = p[mask]
        
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(len(fx)):
            xx = int(fx[i])
            yy = int(fy[i])
            pol = fp[i]
            
            if 0 <= xx < width and 0 <= yy < height:
                if pol > 0:
                    frame[yy, xx, 2] = 255  # Red
                else:
                    frame[yy, xx, 0] = 255  # Blue
        
        has_detection = center is not None
        if has_detection:
            detection_count += 1
            cx = center['x']
            cy = center['y']
            votes = center['votes']
            
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 2)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 2)
            
            cv2.putText(frame, f"DETECTED", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Center: ({cx}, {cy})", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Votes: {votes:.1f}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            pol = center['dominant_polarity']
            pol_conf = center['polarity_confidence']
            pol_text = f"Pol: {pol} ({pol_conf:.2f})"
            cv2.putText(frame, pol_text, (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.putText(frame, f"NO DETECTION", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            cv2.putText(frame, f"Votes: < threshold", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        cv2.putText(frame, f"File: {basename}", (10, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Time: {start:.2f}s", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Events: {len(fx)}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        status = "DETECTED" if has_detection else "NO_DETECT"
        if has_detection:
            filename = f"{basename}_frame_{frame_idx:06d}_{status}_t{start:.3f}s_votes{votes:.0f}.png"
        else:
            filename = f"{basename}_frame_{frame_idx:06d}_{status}_t{start:.3f}s.png"
        
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        saved_count += 1
    
    return saved_count, detection_count

def save_annotations_csv(centers, output_dir, basename, time_window=0.02):
    """Save center annotations to CSV file"""
    annotations = []
    
    for frame_idx, center in enumerate(centers):
        if center is not None:
            annotations.append({
                'timestamp': center['timestamp'],
                'x': center['x'],
                'y': center['y'],
                'frame_idx': frame_idx,
                'votes': center['votes'],
                'dominant_polarity': center['dominant_polarity'],
                'polarity_confidence': center['polarity_confidence'],
                'polarity_ratio': center['polarity_ratio'],
                'positive_events': center['positive_events'],
                'negative_events': center['negative_events'],
                'total_nearby_events': center['total_nearby_events']
            })
    
    if annotations:
        df = pd.DataFrame(annotations)
        csv_filename = f"{basename}_annotations.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_filepath, index=False)
        return len(annotations)
    else:
        df = pd.DataFrame(columns=[
            'timestamp', 'x', 'y', 'frame_idx', 'votes', 
            'dominant_polarity', 'polarity_confidence', 'polarity_ratio',
            'positive_events', 'negative_events', 'total_nearby_events'
        ])
        csv_filename = f"{basename}_annotations.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_filepath, index=False)
        return 0

def process_single_mat_file(mat_filepath, output_frames_dir, output_annotations_dir, 
                           min_votes=150, time_window=0.02):
    """Process a single .mat file"""
    basename = os.path.splitext(os.path.basename(mat_filepath))[0]
    
    print(f"Processing: {basename}")
    
    try:
        data = loadmat(mat_filepath)
        events = data['TD']
        
        x = events['x'][0][0].flatten().astype(int)
        y = events['y'][0][0].flatten().astype(int)
        ts = events['ts'][0][0].flatten()
        p = events['p'][0][0].flatten()
        
        sort_idx = np.argsort(ts)
        x = x[sort_idx]
        y = y[sort_idx]
        ts = ts[sort_idx]
        p = p[sort_idx]
        
        ts = ts - ts[0]
        
        centers = detect_plane_spatial_voting_simple(
            x, y, ts, p, time_window=time_window, min_votes=min_votes
        )
        
        saved_frames, detection_count = save_all_frames(
            x, y, ts, p, centers,
            output_frames_dir, basename, time_window
        )
        
        annotation_count = save_annotations_csv(
            centers, output_annotations_dir, basename, time_window
        )
        
        return {
            'file': basename,
            'total_events': len(x),
            'total_frames': len(centers),
            'saved_frames': saved_frames,
            'detection_count': detection_count,
            'annotation_count': annotation_count,
            'detection_rate': detection_count / len(centers) * 100 if centers else 0,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        print(f"Error processing {basename}: {str(e)}")
        return {
            'file': basename,
            'success': False,
            'error': str(e)
        }

def process_dataset_directory(input_dir, output_dir, min_votes=150, time_window=0.02):
    splits = ['train', 'val', 'test']
    
    for split in splits:
        frames_dir = os.path.join(output_dir, 'frames', split)
        os.makedirs(frames_dir, exist_ok=True)
        
        annotations_dir = os.path.join(output_dir, 'annotations', split)
        os.makedirs(annotations_dir, exist_ok=True)
    
    total_results = []
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"PROCESSING {split.upper()} SPLIT")
        print(f"{'='*60}")
        
        input_split_dir = os.path.join(input_dir, split)
        mat_files = glob.glob(os.path.join(input_split_dir, "*.mat"))
        
        if not mat_files:
            print(f"No .mat files found in {input_split_dir}")
            continue
        
        print(f"Found {len(mat_files)} .mat files in {split} split")
        
        output_frames_dir = os.path.join(output_dir, 'frames', split)
        output_annotations_dir = os.path.join(output_dir, 'annotations', split)
        
        split_results = []
        
        for mat_file in tqdm(mat_files, desc=f"Processing {split}"):
            result = process_single_mat_file(
                mat_file, output_frames_dir, output_annotations_dir,
                min_votes, time_window
            )
            split_results.append(result)
            total_results.append({**result, 'split': split})
        
        successful_files = [r for r in split_results if r['success']]
        failed_files = [r for r in split_results if not r['success']]
        
        if successful_files:
            total_frames = sum(r['total_frames'] for r in successful_files)
            total_detections = sum(r['detection_count'] for r in successful_files)
            total_annotations = sum(r['annotation_count'] for r in successful_files)
            avg_detection_rate = np.mean([r['detection_rate'] for r in successful_files])
            
            print(f"\n{split.upper()} SPLIT SUMMARY:")
            print(f"Successful files: {len(successful_files)}")
            print(f"Failed files: {len(failed_files)}")
            print(f"Total frames: {total_frames}")
            print(f"Total detections: {total_detections}")
            print(f"Total annotations: {total_annotations}")
            print(f"Average detection rate: {avg_detection_rate:.1f}%")
        
        if failed_files:
            print(f"\nFailed files in {split}:")
            for r in failed_files:
                print(f"  {r['file']}: {r['error']}")
    
    summary_df = pd.DataFrame(total_results)
    summary_path = os.path.join(output_dir, 'processing_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    successful_results = [r for r in total_results if r['success']]
    if successful_results:
        print(f"Total successful files: {len(successful_results)}")
        print(f"Total failed files: {len(total_results) - len(successful_results)}")
        
        for split in splits:
            split_results = [r for r in successful_results if r['split'] == split]
            if split_results:
                total_frames = sum(r['total_frames'] for r in split_results)
                total_detections = sum(r['detection_count'] for r in split_results)
                avg_rate = np.mean([r['detection_rate'] for r in split_results])
                print(f"{split}: {len(split_results)} files, {total_frames} frames, {total_detections} detections ({avg_rate:.1f}%)")
    
    print(f"\nProcessing summary saved to: {summary_path}")
    print(f"Frames saved to: {os.path.join(output_dir, 'frames')}")
    print(f"Annotations saved to: {os.path.join(output_dir, 'annotations')}")

def main():
    """Main function"""
    
    input_dir = "/Users/banika/Desktop/airplane_SNN_tracker/split_dataset"
    output_dir = "/Users/banika/Desktop/airplane_SNN_tracker/processed_dataset"
    min_votes = 125
    time_window = 0.02
    
    print("=" * 60)
    print("BATCH SPATIAL VOTING PLANE DETECTION")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Minimum vote threshold: {min_votes}")
    print(f"Time window: {time_window * 1000}ms")
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    required_splits = ['train', 'val', 'test']
    missing_splits = []
    for split in required_splits:
        split_dir = os.path.join(input_dir, split)
        if not os.path.exists(split_dir):
            missing_splits.append(split)
    
    if missing_splits:
        print(f"Warning: Missing split directories: {missing_splits}")
        print("Will only process existing splits...")
    
    try:
        response = input("Continue with processing? (y/n): ").strip().lower()
        if response != 'y':
            print("Processing cancelled.")
            return
    except KeyboardInterrupt:
        print("\nProcessing cancelled.")
        return
    
    process_dataset_directory(input_dir, output_dir, min_votes, time_window)

if __name__ == "__main__":
    main()