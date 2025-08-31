import os
import glob
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import scipy.io as sio

class BoundingBoxGenerator:
    def __init__(self, dataset_dir, annotations_dir, output_dir, visualization_dir):
        self.dataset_dir = dataset_dir
        self.annotations_dir = annotations_dir
        self.output_dir = output_dir
        self.visualization_dir = visualization_dir
        
        # create dirs
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # image size
        self.width = 304
        self.height = 240
        self.time_window = 0.02
        
        print(f"dataset: {self.dataset_dir}")
        print(f"annotations: {self.annotations_dir}")
        print(f"output: {self.output_dir}")
        print(f"vis: {self.visualization_dir}")
    

    def _clamp_coords(self, x1, y1, x2, y2):
        # keep coords in bounds
        x1 = max(0, min(self.width - 1, x1))
        y1 = max(0, min(self.height - 1, y1))
        x2 = max(x1 + 1, min(self.width - 1, x2))
        y2 = max(y1 + 1, min(self.height - 1, y2))
        return int(x1), int(y1), int(x2), int(y2)
    

    def load_events_from_mat(self, mat_file):
        # load events from mat file
        data = sio.loadmat(mat_file)
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
        
        ts = (ts - ts[0]) / 1e6
        
        events = np.column_stack([x, y, ts, p])
        return torch.from_numpy(events).float()
    

    def event2spikes_density(self, events):
        # convert events to spike density
        if events.shape[0] == 0:
            return torch.zeros(1, 2, self.height, self.width)
        
        duration = events[:, 2].max()
        num_frames = int(duration / self.time_window) + 1
        
        spikes = torch.zeros(num_frames, 2, self.height, self.width, dtype=torch.float32)
        
        x = events[:, 0].long()
        y = events[:, 1].long()
        ts = events[:, 2]
        p = events[:, 3]
        
        frame_idx = torch.clamp((ts / self.time_window).long(), 0, num_frames - 1)
        
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x = x[valid]
        y = y[valid]
        frame_idx = frame_idx[valid]
        p = p[valid]
        
        pos_mask = p > 0
        neg_mask = p <= 0
        
        if pos_mask.sum() > 0:
            pos_frames = frame_idx[pos_mask]
            pos_y = y[pos_mask]
            pos_x = x[pos_mask]
            
            for i in range(len(pos_frames)):
                spikes[pos_frames[i], 0, pos_y[i], pos_x[i]] += 1
                
        if neg_mask.sum() > 0:
            neg_frames = frame_idx[neg_mask]
            neg_y = y[neg_mask]
            neg_x = x[neg_mask]
            
            for i in range(len(neg_frames)):
                spikes[neg_frames[i], 1, neg_y[i], neg_x[i]] += 1
        
        return spikes
    

    def generate_bbox_from_spikes(self, spikes, center_x, center_y, confidence=0.8):
        # make bbox using spike density around gt center
        density = torch.sum(spikes, dim=0)
        total_spikes = torch.sum(density).item()
        max_density = torch.max(density).item()
        
        cx, cy = int(center_x), int(center_y)
        cx = max(0, min(self.width - 1, cx))
        cy = max(0, min(self.height - 1, cy))
        
        spike_vals = density[density > 0]
        if len(spike_vals) > 0:
            threshold = max_density * (0.1 if total_spikes < 50 else 0.2)
        else:
            threshold = 0.1
        
        mask = density > threshold
        
        if mask.any():
            x_left = self._find_extent(density, cx, cy, 'left', threshold, confidence)
            x_right = self._find_extent(density, cx, cy, 'right', threshold, confidence)
            y_up = self._find_extent(density, cx, cy, 'up', threshold, confidence)
            y_down = self._find_extent(density, cx, cy, 'down', threshold, confidence)
            
            x1 = center_x - x_left
            x2 = center_x + x_right
            y1 = center_y - y_up
            y2 = center_y + y_down
            
            min_w, min_h = self._get_min_size(total_spikes, confidence)
            
            if x2 - x1 < min_w:
                expand = (min_w - (x2 - x1)) / 2
                x1 -= expand
                x2 += expand
                
            if y2 - y1 < min_h:
                expand = (min_h - (y2 - y1)) / 2
                y1 -= expand
                y2 += expand
            
            bbox = self._clamp_coords(x1, y1, x2, y2)
        else:
            bbox = self._fallback_bbox(center_x, center_y, total_spikes, confidence)
            
        return bbox
    

    def _find_extent(self, density, cx, cy, direction, threshold, confidence=0.8):
        # find extent of spikes in direction from center
        max_extent = 40
        extent = 5
        window = 8 if confidence > 0.7 else 5
        
        if direction == 'left':
            for i in range(1, min(max_extent, cx + 1)):
                x = cx - i
                if x >= 0:
                    y_start = max(0, cy - window)
                    y_end = min(density.shape[0], cy + window + 1)
                    max_val = density[y_start:y_end, x].max().item()
                    if max_val > threshold:
                        extent = i + 3
                    else:
                        if confidence > 0.7 and i < 15:
                            continue
                        else:
                            break
        elif direction == 'right':
            for i in range(1, min(max_extent, density.shape[1] - cx)):
                x = cx + i
                if x < density.shape[1]:
                    y_start = max(0, cy - window)
                    y_end = min(density.shape[0], cy + window + 1)
                    max_val = density[y_start:y_end, x].max().item()
                    if max_val > threshold:
                        extent = i + 3
                    else:
                        if confidence > 0.7 and i < 15:
                            continue
                        else:
                            break
        elif direction == 'up':
            for i in range(1, min(max_extent, cy + 1)):
                y = cy - i
                if y >= 0:
                    x_start = max(0, cx - window)
                    x_end = min(density.shape[1], cx + window + 1)
                    max_val = density[y, x_start:x_end].max().item()
                    if max_val > threshold:
                        extent = i + 3
                    else:
                        if confidence > 0.7 and i < 15:
                            continue
                        else:
                            break
        elif direction == 'down':
            for i in range(1, min(max_extent, density.shape[0] - cy)):
                y = cy + i
                if y < density.shape[0]:
                    x_start = max(0, cx - window)
                    x_end = min(density.shape[1], cx + window + 1)
                    max_val = density[y, x_start:x_end].max().item()
                    if max_val > threshold:
                        extent = i + 3
                    else:
                        if confidence > 0.7 and i < 15:
                            continue
                        else:
                            break
        
        return min(extent, max_extent)
    

    def _get_min_size(self, total_spikes, confidence):
        # get min box size based on spike activity
        sizes = {
            True: {5: (6, 6), 15: (8, 8), 50: (10, 10), float('inf'): (14, 14)},
            False: {5: (8, 8), 15: (10, 10), 50: (12, 12), float('inf'): (16, 16)}
        }
        
        size_map = sizes[confidence > 0.7]
        for threshold, size in size_map.items():
            if total_spikes < threshold:
                return size
    

    def _fallback_bbox(self, center_x, center_y, total_spikes, confidence=0.8):
        # create fallback bbox around center
        min_w, min_h = self._get_min_size(total_spikes, confidence)
        half_w, half_h = min_w / 2, min_h / 2
        
        x1 = center_x - half_w
        y1 = center_y - half_h
        x2 = center_x + half_w
        y2 = center_y + half_h
        
        if x1 < 0:
            x1, x2 = 0, min(self.width - 1, min_w)
        elif x2 >= self.width:
            x2, x1 = self.width - 1, max(0, self.width - 1 - min_w)
        
        if y1 < 0:
            y1, y2 = 0, min(self.height - 1, min_h)
        elif y2 >= self.height:
            y2, y1 = self.height - 1, max(0, self.height - 1 - min_h)
        
        return self._clamp_coords(x1, y1, x2, y2)
    

    def visualize_frame_with_bbox(self, frame_spikes, center_x, center_y, bbox, frame_idx, filename, save_path, timestamp):
        # show frame with generated bbox
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        pos_events = frame_spikes[0].numpy()
        neg_events = frame_spikes[1].numpy()
        
        spike_max = max(pos_events.max(), neg_events.max())
        spike_min = 0.0
        
        im1 = axes[0].imshow(pos_events, cmap='Reds', vmin=spike_min, vmax=spike_max)
        axes[0].set_title(f'positive events\nframe {frame_idx}')
        axes[0].set_xlabel('width (pixels)')
        axes[0].set_ylabel('height (pixels)')
        plt.colorbar(im1, ax=axes[0], label='event density')
        
        axes[0].plot(center_x, center_y, 'bo', markersize=8, label='gt center')
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='blue', facecolor='none', label='generated bbox')
            axes[0].add_patch(rect)
        axes[0].legend()
        
        im2 = axes[1].imshow(neg_events, cmap='Blues', vmin=spike_min, vmax=spike_max)
        axes[1].set_title(f'negative events\nframe {frame_idx}')
        axes[1].set_xlabel('width (pixels)')
        axes[1].set_ylabel('height (pixels)')
        plt.colorbar(im2, ax=axes[1], label='event density')
        
        axes[1].plot(center_x, center_y, 'ro', markersize=8, label='gt center')
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none', label='generated bbox')
            axes[1].add_patch(rect)
        axes[1].legend()
        
        combined = np.zeros((pos_events.shape[0], pos_events.shape[1], 3))
        if spike_max > 0:
            combined[:, :, 0] = pos_events / spike_max
            combined[:, :, 2] = neg_events / spike_max
        axes[2].imshow(combined)
        axes[2].set_title(f'combined view\nframe {frame_idx}')
        axes[2].set_xlabel('width (pixels)')
        axes[2].set_ylabel('height (pixels)')
        
        axes[2].plot(center_x, center_y, 'wo', markersize=8, label='gt center')
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='white', facecolor='none', label='generated bbox')
            axes[2].add_patch(rect)
            axes[2].text(x1+5, y1-5, f'bbox: ({x1},{y1})-({x2},{y2})', 
                        color='white', fontsize=8, weight='bold')
            axes[2].text(center_x+5, center_y+15, f'gt: ({center_x:.1f},{center_y:.1f})', 
                        color='yellow', fontsize=8, weight='bold')
        axes[2].legend()
        
        pos_count = pos_events.sum()
        neg_count = neg_events.sum()
        bbox_info = f' | bbox: ({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]}) wxh: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}' if bbox else ' | no bbox'
        fig.suptitle(f'{filename} | frame {frame_idx} | t={timestamp:.3f}s | '
                    f'pos: {pos_count:.0f} | neg: {neg_count:.0f}{bbox_info}', 
                    fontsize=10, weight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    
    def process_single_file(self, mat_file, split):
        # process single mat file and generate bboxes using gt centers
        print(f"\nprocessing: {os.path.basename(mat_file)}")
        
        events = self.load_events_from_mat(mat_file)
        print(f"loaded {len(events)} events")
        
        spikes = self.event2spikes_density(events)
        print(f"generated spike sequence: {spikes.shape}")
        
        filename = os.path.basename(mat_file)
        filename_no_ext = os.path.splitext(filename)[0]
        csv_file = os.path.join(self.annotations_dir, split, f"{filename_no_ext}_annotations.csv")
        
        if not os.path.exists(csv_file):
            print(f"warning: annotation file not found: {csv_file}")
            return None
        
        df = pd.read_csv(csv_file)
        print(f"loaded {len(df)} annotations from csv")
        
        df['bbox_x1'] = 0
        df['bbox_y1'] = 0
        df['bbox_x2'] = 0
        df['bbox_y2'] = 0
        df['bbox_width'] = 0
        df['bbox_height'] = 0
        df['bbox_area'] = 0
        
        file_viz_dir = os.path.join(self.visualization_dir, split, filename_no_ext)
        os.makedirs(file_viz_dir, exist_ok=True)
        
        bboxes_generated = 0
        frames_visualized = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="generating bboxes"):
            timestamp = row['timestamp']
            gt_x = row['x']
            gt_y = row['y']
            polarity = row['dominant_polarity']
            
            frame_idx = int(timestamp / self.time_window)
            
            if frame_idx < spikes.shape[0]:
                frame_spikes = spikes[frame_idx]
                
                confidence = 0.8
                bbox = self.generate_bbox_from_spikes(frame_spikes, gt_x, gt_y, confidence)
                
                x1, y1, x2, y2 = bbox
                
                df.at[idx, 'bbox_x1'] = x1
                df.at[idx, 'bbox_y1'] = y1
                df.at[idx, 'bbox_x2'] = x2
                df.at[idx, 'bbox_y2'] = y2
                df.at[idx, 'bbox_width'] = x2 - x1
                df.at[idx, 'bbox_height'] = y2 - y1
                df.at[idx, 'bbox_area'] = (x2 - x1) * (y2 - y1)
                
                bboxes_generated += 1
                
                if frames_visualized < 50 or idx % 10 == 0:
                    viz_filename = f"frame_{frame_idx:04d}_ann_{idx:03d}_t{timestamp:.3f}.png"
                    viz_path = os.path.join(file_viz_dir, viz_filename)
                    
                    self.visualize_frame_with_bbox(
                        frame_spikes, gt_x, gt_y, bbox, 
                        frame_idx, filename_no_ext, viz_path, timestamp
                    )
                    frames_visualized += 1
            else:
                print(f"warning: frame {frame_idx} out of range for {filename} (timestamp: {timestamp:.3f}s)")
        
        output_split_dir = os.path.join(self.output_dir, split)
        os.makedirs(output_split_dir, exist_ok=True)
        output_file = os.path.join(output_split_dir, f"{filename_no_ext}_annotations_with_bbox.csv")
        df.to_csv(output_file, index=False)
        
        print(f"generated {bboxes_generated} bounding boxes")
        print(f"visualized {frames_visualized} frames")
        print(f"saved to: {output_file}")
        
        return df
    

    def process_all_files(self, splits=['train', 'val']):
        # process all files in dataset
        total_files = 0
        total_bboxes = 0
        
        for split in splits:
            print(f"\n{'='*50}")
            print(f"processing {split} split")
            print('='*50)
            
            split_dir = os.path.join(self.dataset_dir, split)
            if not os.path.exists(split_dir):
                print(f"warning: split directory {split_dir} does not exist")
                continue
            
            mat_files = glob.glob(os.path.join(split_dir, "*.mat"))
            print(f"found {len(mat_files)} .mat files in {split}")
            
            for mat_file in mat_files:
                try:
                    df = self.process_single_file(mat_file, split)
                    if df is not None:
                        total_files += 1
                        total_bboxes += len(df)
                except Exception as e:
                    print(f"error processing {mat_file}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"\nprocessed {total_files} files")
        print(f"generated {total_bboxes} bounding boxes")
        print(f"output directory: {self.output_dir}")
        print(f"visualization directory: {self.visualization_dir}")


def main():
    # main function to generate bboxes using gt centers
    
    config = {
        'dataset_dir': '/Users/banika/Desktop/airplane_SNN_tracker/split_dataset',
        'annotations_dir': '/Users/banika/Desktop/airplane_SNN_tracker/processed_dataset/annotations',
        'output_dir': '/Users/banika/Desktop/airplane_SNN_tracker/annotations_with_bbox',
        'visualization_dir': '/Users/banika/Desktop/airplane_SNN_tracker/bbox_visualizations'
    }
    
    print(f"dataset dir: {config['dataset_dir']}")
    print(f"annotations dir: {config['annotations_dir']}")
    print(f"output dir: {config['output_dir']}")
    print(f"visualization dir: {config['visualization_dir']}")
    
    for key, path in config.items():
        if key not in ['output_dir', 'visualization_dir']:  
            if not os.path.exists(path):
                print(f"error: {key} directory does not exist: {path}")
                return
    
    bbox_generator = BoundingBoxGenerator(
        dataset_dir=config['dataset_dir'],
        annotations_dir=config['annotations_dir'],
        output_dir=config['output_dir'],
        visualization_dir=config['visualization_dir']
    )
    
    print("\nstarting bbox generation...")
    bbox_generator.process_all_files(splits=['test'])
    


if __name__ == "__main__":
    main()