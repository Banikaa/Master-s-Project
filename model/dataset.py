import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gc



class SequenceDataset(Dataset):
    def __init__(self, split, sequence_length=None, spike_mode='density'):
        self.recordings = []
        self.time_window = 0.005 # this will activate the interpolation
        self.original_gt_window = 0.02
        self.split = split
        self.sequence_length = sequence_length
        self.spike_mode = spike_mode

        self.dir_rec = '/Users/banika/Desktop/airplane_SNN_tracker/split_dataset'
        self.dir_gt = '/Users/banika/Desktop/airplane_SNN_tracker/annotations_with_bbox'
        self.width = 304
        self.height = 240

        # create the annotations directory
        annotations_dir = os.path.join(self.dir_gt, self.split)
        if not os.path.exists(annotations_dir):
            raise ValueError(f"Annotations directory {annotations_dir} does not exist.")
        self.annotations_dir = annotations_dir

        split_dir = os.path.join(self.dir_rec, self.split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist.")
        self.split_dir = split_dir

        self.mat_files = glob.glob(os.path.join(self.split_dir, "*.mat"))
        
        # process all files to store as recordings
        self.process_all_files()


    def process_all_files(self):
        for mat_file in self.mat_files:
            try:
                # load events and convert to spikes
                events = self.load_events(mat_file)
                spike_sequence = self.event2spikes(events)
                
                # load original annotations
                original_annotations = self.load_annotations(mat_file)
                
                # create interpolated labels for each 10ms frame
                num_frames = spike_sequence.shape[0]
                frame_labels = self._create_interpolated_frame_labels(original_annotations, num_frames)
                
                # store entire recording as one sample
                recording_data = {
                    'spike_sequence': spike_sequence,
                    'frame_labels': frame_labels,
                    'original_annotations': original_annotations,
                    'filename': os.path.basename(mat_file),
                    'duration': num_frames * self.time_window
                }
                
                self.recordings.append(recording_data)
                    
            except Exception as e:
                print(f"Error processing {mat_file}: {e}")
                continue


    def load_annotations(self, mat_file):
        filename = os.path.basename(mat_file)
        filename_no_ext = os.path.splitext(filename)[0]
        csv_file = os.path.join(self.annotations_dir, f"{filename_no_ext}_annotations_with_bbox.csv")

        df = pd.read_csv(csv_file)

        # load all annotation data including bounding boxes
        timestamps = df['timestamp'].values
        x_centers = df['x'].values
        y_centers = df['y'].values
        polarities = df['dominant_polarity'].values
        bbox_x1 = df['bbox_x1'].values
        bbox_y1 = df['bbox_y1'].values
        bbox_x2 = df['bbox_x2'].values
        bbox_y2 = df['bbox_y2'].values

        # stack all annotation data
        annotations = np.column_stack([timestamps, x_centers, y_centers, polarities, 
                                     bbox_x1, bbox_y1, bbox_x2, bbox_y2])
        return torch.from_numpy(annotations).float()


    def _create_interpolated_frame_labels(self, annotations, num_frames):
        frame_labels = []

        for frame_idx in range(num_frames):
            frame_start = frame_idx * self.time_window
            frame_end = frame_start + self.time_window

            if annotations.shape[0] > 0:
                # check if this 10ms frame has direct gt
                time_mask = (annotations[:, 0] >= frame_start) & (annotations[:, 0] < frame_end)
                direct_annotations = annotations[time_mask]

                if direct_annotations.shape[0] > 0:
                    # direct gt available - use it directly
                    center_x = direct_annotations[:, 1].mean()
                    center_y = direct_annotations[:, 2].mean()
                    bbox_x1 = direct_annotations[:, 4].mean()
                    bbox_y1 = direct_annotations[:, 5].mean()
                    bbox_x2 = direct_annotations[:, 6].mean()
                    bbox_y2 = direct_annotations[:, 7].mean()
                    
                    label = {
                        'has_plane': True,
                        'center': torch.tensor([center_x, center_y], dtype=torch.float32),
                        'bbox': torch.tensor([bbox_x1, bbox_y1, bbox_x2, bbox_y2], dtype=torch.float32),
                        'confidence_target': 1.0,
                        'interpolated': False
                    }
                else:
                    # no direct gt - try to interpolate from neighboring frames
                    interpolated_label = self._interpolate_gt_for_frame(annotations, frame_start, frame_end)
                    if interpolated_label is not None:
                        interpolated_label['interpolated'] = True
                        label = interpolated_label
                    else:
                        # no plane in this frame
                        label = {
                            'has_plane': False,
                            'center': torch.tensor([0.0, 0.0], dtype=torch.float32),
                            'bbox': torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                            'confidence_target': 0.0,
                            'interpolated': False
                        }
            else:
                # no annotations at all
                label = {
                    'has_plane': False,
                    'center': torch.tensor([0.0, 0.0], dtype=torch.float32),
                    'bbox': torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                    'confidence_target': 0.0,
                    'interpolated': False
                }

            frame_labels.append(label)

        return frame_labels


    def _interpolate_gt_for_frame(self, annotations, frame_start, frame_end):
        frame_center = (frame_start + frame_end) / 2
        
        # find the closest annotations before and after this frame
        timestamps = annotations[:, 0]
        
        # find annotations before this frame
        before_mask = timestamps < frame_start
        if before_mask.sum() > 0:
            before_times = timestamps[before_mask]
            closest_before_idx = torch.argmax(before_times)
            before_annotation = annotations[before_mask][closest_before_idx]
        else:
            before_annotation = None
            
        # find annotations after this frame
        after_mask = timestamps >= frame_end
        if after_mask.sum() > 0:
            after_times = timestamps[after_mask]
            closest_after_idx = torch.argmin(after_times)
            after_annotation = annotations[after_mask][closest_after_idx]
        else:
            after_annotation = None
        
        # interpolate if we have both before and after
        if before_annotation is not None and after_annotation is not None:
            # check if the time gap is reasonable for interpolation
            time_gap = after_annotation[0] - before_annotation[0]
            if time_gap <= 0.04:
                # calculate interpolation weight
                t_before = before_annotation[0].item()
                t_after = after_annotation[0].item()
                weight = (frame_center - t_before) / (t_after - t_before)
                weight = torch.clamp(torch.tensor(weight), 0, 1)
                
                # interpolate center
                center_x = (1 - weight) * before_annotation[1] + weight * after_annotation[1]
                center_y = (1 - weight) * before_annotation[2] + weight * after_annotation[2]
                
                # interpolate bbox
                bbox_x1 = (1 - weight) * before_annotation[4] + weight * after_annotation[4]
                bbox_y1 = (1 - weight) * before_annotation[5] + weight * after_annotation[5]
                bbox_x2 = (1 - weight) * before_annotation[6] + weight * after_annotation[6]
                bbox_y2 = (1 - weight) * before_annotation[7] + weight * after_annotation[7]
                
                return {
                    'has_plane': True,
                    'center': torch.tensor([center_x, center_y], dtype=torch.float32),
                    'bbox': torch.tensor([bbox_x1, bbox_y1, bbox_x2, bbox_y2], dtype=torch.float32),
                    'confidence_target': 0.8
                }
        
        # if we only have one neighbor, use it if it's close enough
        elif before_annotation is not None:
            time_diff = frame_center - before_annotation[0]
            if time_diff <= 0.015:
                return {
                    'has_plane': True,
                    'center': torch.tensor([before_annotation[1], before_annotation[2]], dtype=torch.float32),
                    'bbox': torch.tensor([before_annotation[4], before_annotation[5], 
                                        before_annotation[6], before_annotation[7]], dtype=torch.float32),
                    'confidence_target': 0.7
                }
        elif after_annotation is not None:
            time_diff = after_annotation[0] - frame_center
            if time_diff <= 0.015:
                return {
                    'has_plane': True,
                    'center': torch.tensor([after_annotation[1], after_annotation[2]], dtype=torch.float32),
                    'bbox': torch.tensor([after_annotation[4], after_annotation[5], 
                                        after_annotation[6], after_annotation[7]], dtype=torch.float32),
                    'confidence_target': 0.7
                }
        return None
    

    def __len__(self):
        return len(self.recordings)


    def __getitem__(self, idx):
        recording = self.recordings[idx]
        spike_sequence = recording['spike_sequence']
        frame_labels = recording['frame_labels']
        
        # if sequence_length is specified, sample a subsequence
        if self.sequence_length is not None and spike_sequence.shape[0] > self.sequence_length:
            # random start point for training, fixed for validation
            if self.split == 'train':
                start_idx = torch.randint(0, spike_sequence.shape[0] - self.sequence_length + 1, (1,)).item()
            else:
                start_idx = 0
            
            end_idx = start_idx + self.sequence_length
            spike_sequence = spike_sequence[start_idx:end_idx]
            frame_labels = frame_labels[start_idx:end_idx]
        
        # convert frame labels to tensors
        has_plane_seq = torch.tensor([label['has_plane'] for label in frame_labels], dtype=torch.bool)
        center_seq = torch.stack([label['center'] for label in frame_labels])
        bbox_seq = torch.stack([label['bbox'] for label in frame_labels])
        confidence_seq = torch.tensor([label['confidence_target'] for label in frame_labels], dtype=torch.float32)
        interpolated_seq = torch.tensor([label['interpolated'] for label in frame_labels], dtype=torch.bool)
        
        return {
            'spike_sequence': spike_sequence,
            'has_plane_sequence': has_plane_seq,
            'center_sequence': center_seq,
            'bbox_sequence': bbox_seq,
            'confidence_sequence': confidence_seq,
            'interpolated_sequence': interpolated_seq,
            'filename': recording['filename'],
            'duration': recording['duration']
        }


    def load_events(self, mat_file):
        data = sio.loadmat(mat_file)
        events = data['TD']
        x_coords = events['x'][0][0].flatten().astype(int)
        y_coords = events['y'][0][0].flatten().astype(int)
        timestamps = events['ts'][0][0].flatten()
        polarities = events['p'][0][0].flatten()
        
        sort_idx = np.argsort(timestamps)
        x_coords = x_coords[sort_idx]
        y_coords = y_coords[sort_idx]
        timestamps = timestamps[sort_idx]
        polarities = polarities[sort_idx]
        
        timestamps = timestamps - timestamps[0]
        timestamps_sec = timestamps / 1e6
        
        events = np.column_stack([x_coords, y_coords, timestamps_sec, polarities])
        return torch.from_numpy(events).float()


    def event2spikes(self, events):
        if events.shape[0] == 0:
            return torch.zeros(1, 2, self.height, self.width)
        
        total_duration = events[:, 2].max()
        num_frames = int(total_duration / self.time_window) + 1
        
        # initialize spike sequence based on mode
        if self.spike_mode == 'binary':
            spike_sequence = torch.zeros(num_frames, 2, self.height, self.width)
        else:
            spike_sequence = torch.zeros(num_frames, 2, self.height, self.width, dtype=torch.float32)
        
        x_coords = events[:, 0].long()
        y_coords = events[:, 1].long()
        timestamps = events[:, 2]
        polarities = events[:, 3]
        
        # calculate which frame each event belongs to
        frame_indices = torch.clamp((timestamps / self.time_window).long(), 0, num_frames - 1)
        
        # filter valid coordinates
        valid_mask = (x_coords >= 0) & (x_coords < self.width) & (y_coords >= 0) & (y_coords < self.height)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        frame_indices = frame_indices[valid_mask]
        polarities = polarities[valid_mask]
        
        # process positive and negative events
        pos_mask = polarities > 0
        neg_mask = polarities <= 0
        
        if self.spike_mode == 'binary':
            if pos_mask.sum() > 0:
                spike_sequence[frame_indices[pos_mask], 0, y_coords[pos_mask], x_coords[pos_mask]] = 1
            if neg_mask.sum() > 0:
                spike_sequence[frame_indices[neg_mask], 1, y_coords[neg_mask], x_coords[neg_mask]] = 1
                
        elif self.spike_mode == 'density':
            if pos_mask.sum() > 0:
                pos_frame_idx = frame_indices[pos_mask]
                pos_y = y_coords[pos_mask]
                pos_x = x_coords[pos_mask]
                
                for i in range(len(pos_frame_idx)):
                    spike_sequence[pos_frame_idx[i], 0, pos_y[i], pos_x[i]] += 1
                    
            if neg_mask.sum() > 0:
                neg_frame_idx = frame_indices[neg_mask]
                neg_y = y_coords[neg_mask]
                neg_x = x_coords[neg_mask]
                
                for i in range(len(neg_frame_idx)):
                    spike_sequence[neg_frame_idx[i], 1, neg_y[i], neg_x[i]] += 1

        return spike_sequence


    def get_spike_statistics(self):
        print(f"analyzing spike statistics for {len(self.recordings)} recordings...")
        
        total_events = 0
        total_frames = 0
        max_events_per_pixel = 0
        max_events_per_frame = 0
        events_per_frame_list = []
        
        total_interpolated_frames = 0
        total_direct_gt_frames = 0
        
        for recording in self.recordings:
            spike_seq = recording['spike_sequence']
            frame_labels = recording['frame_labels']
            
            for frame_idx in range(spike_seq.shape[0]):
                frame_events = spike_seq[frame_idx].sum().item()
                events_per_frame_list.append(frame_events)
                max_events_per_frame = max(max_events_per_frame, frame_events)
                
                # check max events per pixel
                frame_max = spike_seq[frame_idx].max().item()
                max_events_per_pixel = max(max_events_per_pixel, frame_max)
                
                total_events += frame_events
                total_frames += 1
                
                # count interpolation statistics
                if frame_labels[frame_idx]['has_plane']:
                    if frame_labels[frame_idx]['interpolated']:
                        total_interpolated_frames += 1
                    else:
                        total_direct_gt_frames += 1
        
        avg_events_per_frame = total_events / total_frames if total_frames > 0 else 0
        
        print(f"spike mode: {self.spike_mode}")
        print(f"total frames: {total_frames}")
        print(f"total events: {total_events}")
        print(f"average events per frame: {avg_events_per_frame:.2f}")
        print(f"max events per frame: {max_events_per_frame}")
        print(f"max events per pixel: {max_events_per_pixel}")
        
        print(f"\ninterpolation statistics:")
        print(f"direct gt frames: {total_direct_gt_frames}")
        print(f"interpolated gt frames: {total_interpolated_frames}")
        total_plane_frames = total_direct_gt_frames + total_interpolated_frames
        if total_plane_frames > 0:
            interpolation_ratio = total_interpolated_frames / total_plane_frames
            print(f"interpolation ratio: {interpolation_ratio:.1%}")
        
        if events_per_frame_list:
            import numpy as np
            events_array = np.array(events_per_frame_list)
            print(f"events per frame - median: {np.median(events_array):.2f}, "
                  f"std: {np.std(events_array):.2f}, "
                  f"95th percentile: {np.percentile(events_array, 95):.2f}")
        
        return {
            'total_events': total_events,
            'total_frames': total_frames,
            'avg_events_per_frame': avg_events_per_frame,
            'max_events_per_frame': max_events_per_frame,
            'max_events_per_pixel': max_events_per_pixel,
            'events_per_frame_list': events_per_frame_list,
            'direct_gt_frames': total_direct_gt_frames,
            'interpolated_frames': total_interpolated_frames
        }


def get_dataloader(split='train', batch_size=1, shuffle=None, sequence_length=None, spike_mode='density'):
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = SequenceDataset(split=split, sequence_length=sequence_length, spike_mode=spike_mode)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    
    return dataloader


def visualize_interpolated_recording(dataloader, save_dir='./visualization_10ms', recording_idx=0, max_frames=500):
    # create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # get the recording
    dataset = dataloader.dataset
    if recording_idx >= len(dataset):
        print(f"Error: Recording index {recording_idx} out of range. Dataset has {len(dataset)} recordings.")
        return
    
    sample = dataset[recording_idx]
    spike_sequence = sample['spike_sequence']
    has_plane_seq = sample['has_plane_sequence']
    center_seq = sample['center_sequence']
    bbox_seq = sample['bbox_sequence']
    confidence_seq = sample['confidence_sequence']
    interpolated_seq = sample['interpolated_sequence']
    filename = sample['filename']
    duration = sample['duration']
    spike_mode = dataset.spike_mode
    
    print(f"visualizing interpolated recording: {filename}")
    print(f"duration: {duration:.3f}s, frames: {spike_sequence.shape[0]}")
    print(f"total frames with plane: {has_plane_seq.sum().item()}/{len(has_plane_seq)}")
    print(f"direct gt frames: {(has_plane_seq & ~interpolated_seq).sum().item()}")
    print(f"interpolated gt frames: {(has_plane_seq & interpolated_seq).sum().item()}")
    print(f"spike mode: {spike_mode}")
    
    # calculate colormap limits for consistent visualization
    if spike_mode != 'binary':
        spike_max = spike_sequence.max().item()
        spike_min = spike_sequence.min().item()
        print(f"spike value range: [{spike_min:.3f}, {spike_max:.3f}]")
    else:
        spike_max = 1.0
        spike_min = 0.0
    
    # process each frame individually
    total_frames = min(spike_sequence.shape[0], max_frames)
    frames_saved = 0
    print(f"processing {total_frames} frames...")
    
    for frame_idx in range(total_frames):
        frame_spikes = spike_sequence[frame_idx]
        has_plane = has_plane_seq[frame_idx].item()
        center = center_seq[frame_idx]
        bbox = bbox_seq[frame_idx]
        confidence = confidence_seq[frame_idx].item()
        is_interpolated = interpolated_seq[frame_idx].item()
        
        # skip frames without gt
        if not has_plane:
            continue
        
        frames_saved += 1
        
        # determine gt type for visualization
        if has_plane:
            if is_interpolated:
                gt_type = "interpolated"
                annotation_alpha = 0.7
                center_color = 'orange'
                bbox_color = 'orange'
                center_size = 6
                bbox_linewidth = 1.5
            else:
                gt_type = "direct_gt"
                annotation_alpha = 1.0
                center_color = 'lime'
                bbox_color = 'cyan'
                center_size = 8
                bbox_linewidth = 2.0
        else:
            gt_type = "no_plane"
            annotation_alpha = 0.0
        
        # create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # frame 1: positive events
        pos_events = frame_spikes[0].numpy()
        im1 = axes[0].imshow(pos_events, cmap='Reds', vmin=spike_min, vmax=spike_max)
        axes[0].set_title(f'positive events\nframe {frame_idx}, t={frame_idx*0.01:.3f}s')
        axes[0].set_xlabel('width (pixels)')
        axes[0].set_ylabel('height (pixels)')
        
        # add colorbar for non-binary modes
        if spike_mode != 'binary':
            plt.colorbar(im1, ax=axes[0], label='event count/density')
        
        # add annotation if plane is present
        if has_plane:
            # add center point
            circle = patches.Circle((center[0], center[1]), radius=10, 
                                  linewidth=bbox_linewidth, edgecolor=center_color, 
                                  facecolor='none', alpha=annotation_alpha)
            axes[0].add_patch(circle)
            axes[0].plot(center[0], center[1], 'o', color=center_color, 
                        markersize=center_size, alpha=annotation_alpha)
            
            # add bounding box
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox_width, bbox_height,
                                   linewidth=bbox_linewidth, edgecolor=bbox_color, 
                                   facecolor='none', alpha=annotation_alpha)
            axes[0].add_patch(rect)
        
        # frame 2: negative events
        neg_events = frame_spikes[1].numpy()
        im2 = axes[1].imshow(neg_events, cmap='Blues', vmin=spike_min, vmax=spike_max)
        axes[1].set_title(f'negative events\nframe {frame_idx}, t={frame_idx*0.01:.3f}s')
        axes[1].set_xlabel('width (pixels)')
        axes[1].set_ylabel('height (pixels)')
        
        # add colorbar for non-binary modes
        if spike_mode != 'binary':
            plt.colorbar(im2, ax=axes[1], label='event count/density')
        
        # add annotation if plane is present
        if has_plane:
            # add center point
            circle = patches.Circle((center[0], center[1]), radius=10, 
                                  linewidth=bbox_linewidth, edgecolor=center_color,
                                  facecolor='none', alpha=annotation_alpha)
            axes[1].add_patch(circle)
            axes[1].plot(center[0], center[1], 'o', color=center_color, 
                        markersize=center_size, alpha=annotation_alpha)
            
            # add bounding box
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox_width, bbox_height,
                                   linewidth=bbox_linewidth, edgecolor=bbox_color, 
                                   facecolor='none', alpha=annotation_alpha)
            axes[1].add_patch(rect)
        
        # frame 3: combined view with enhanced gt type visualization
        if spike_mode == 'binary':
            combined = np.zeros((pos_events.shape[0], pos_events.shape[1], 3))
            combined[:, :, 0] = pos_events  # Red channel for positive
            combined[:, :, 2] = neg_events  # Blue channel for negative
            axes[2].imshow(combined)
        else:
            # for non-binary modes, show as separate intensity maps
            combined = np.zeros((pos_events.shape[0], pos_events.shape[1], 3))
            # normalize to [0,1] for rgb display
            if spike_max > 0:
                combined[:, :, 0] = pos_events / spike_max
                combined[:, :, 2] = neg_events / spike_max
            axes[2].imshow(combined)
        
        axes[2].set_title(f'combined view - {gt_type}\nconf: {confidence:.1f}')
        axes[2].set_xlabel('width (pixels)')
        axes[2].set_ylabel('height (pixels)')
        
        # add annotation if plane is present with enhanced styling
        if has_plane:
            # add center point
            circle = patches.Circle((center[0], center[1]), radius=12, 
                                  linewidth=bbox_linewidth+1, edgecolor='white',
                                  facecolor='none', alpha=annotation_alpha)
            axes[2].add_patch(circle)
            axes[2].plot(center[0], center[1], 'o', color=center_color, 
                        markersize=center_size+2, alpha=annotation_alpha)
            
            # add bounding box
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox_width, bbox_height,
                                   linewidth=bbox_linewidth+1, edgecolor='yellow', 
                                   facecolor='none', alpha=annotation_alpha)
            axes[2].add_patch(rect)
            
            # add text annotation
            text_y_offset = -15 if gt_type == "direct_gt" else -25
            axes[2].text(center[0]+15, center[1], f'c:({center[0]:.1f}, {center[1]:.1f})', 
                        color='white', fontsize=9, weight='bold', alpha=annotation_alpha)
            axes[2].text(bbox[0], bbox[1]+text_y_offset, 
                        f'{gt_type}\nbbox: {bbox_width:.0f}x{bbox_height:.0f}', 
                        color='yellow', fontsize=8, weight='bold', alpha=annotation_alpha)
        
        # add event count information
        pos_count = pos_events.sum()
        neg_count = neg_events.sum()
        
        # figure title
        title_color = 'green' if gt_type == "direct_gt" else 'orange' if gt_type == "interpolated" else 'gray'
        fig.suptitle(f'recording: {filename} | frame {frame_idx}/{spike_sequence.shape[0]-1} | '
                    f'gt: {gt_type} | conf: {confidence:.1f} | '
                    f'pos events: {pos_count:.1f} | neg events: {neg_count:.1f}', 
                    fontsize=12, weight='bold', color=title_color)
        
        plt.tight_layout()
            
        # save frame
        gt_suffix = "direct" if gt_type == "direct_gt" else "interp"
        frame_filename = f"{filename[:-4]}_gt_{frames_saved:03d}_frame_{frame_idx:04d}_{gt_suffix}_10ms.png"
        save_path = os.path.join(save_dir, frame_filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # garbage collection every 10 frames
        if frames_saved % 10 == 0:
            gc.collect()
        
        # print frame info occasionally
        if frames_saved % 3 == 1 or gt_type == "direct_gt":
            bbox_info = f"bbox=({bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f})"
            print(f"saved gt {frames_saved:3d}: frame {frame_idx:3d} | {gt_type:12s} | "
                f"center=({center[0]:6.1f}, {center[1]:6.1f}) | {bbox_info} | "
                f"conf={confidence:.1f} | events: +{pos_count:6.1f} -{neg_count:6.1f}")

    print(f"\ngt visualization complete! {frames_saved} gt frames saved to: {save_dir}")
    print(f"files saved with pattern: {filename[:-4]}_gt_XXX_frame_XXXX_[direct/interp]_10ms.png")
    print(f"skipped {total_frames - frames_saved} empty frames")
    
    # summary statistics
    plane_frames = has_plane_seq[:total_frames]
    interpolated_frames = interpolated_seq[:total_frames]
    direct_frames = plane_frames & ~interpolated_frames
    interp_only_frames = plane_frames & interpolated_frames
    
    print(f"\nFrame Summary (first {total_frames} frames):")
    print(f"   Direct GT frames: {direct_frames.sum().item()} (bright green/cyan)")
    print(f"   Interpolated frames: {interp_only_frames.sum().item()} (dimmed orange)")
    print(f"   No plane frames: {(~plane_frames).sum().item()} (no annotations)")
    print(f"   Total plane frames: {plane_frames.sum().item()}")


if __name__ == "__main__":    
    spike_modes = ['binary']
    
    for mode_idx, mode in enumerate(spike_modes):
        print(f"\ntesting spike mode: {mode}")
        
        # create dataloader
        train_loader = get_dataloader(split='test', batch_size=1, shuffle=False, spike_mode=mode)
        
        print(f"train loader created with {len(train_loader.dataset)} recordings.")
        
        # get statistics including interpolation info
        stats = train_loader.dataset.get_spike_statistics()
        
        # get a sample to verify interpolation worked
        if len(train_loader.dataset) > 0:
            sample = train_loader.dataset[0]
            print(f"\nsample verification:")
            print(f"spike sequence shape: {sample['spike_sequence'].shape}")
            print(f"frames with plane: {sample['has_plane_sequence'].sum().item()}")
            print(f"interpolated frames: {sample['interpolated_sequence'].sum().item()}")
            
            # show examples of interpolated vs direct gt
            plane_frames = sample['has_plane_sequence'].nonzero().flatten()
            if len(plane_frames) > 0:
                print(f"\ngt examples (first 5 plane frames):")
                for i, frame_idx in enumerate(plane_frames[:5]):
                    bbox = sample['bbox_sequence'][frame_idx]
                    center = sample['center_sequence'][frame_idx]
                    confidence = sample['confidence_sequence'][frame_idx]
                    interpolated = sample['interpolated_sequence'][frame_idx]
                    gt_type = "interpolated" if interpolated else "direct_gt"
                    print(f"  frame {frame_idx:3d} ({gt_type}): "
                          f"center=({center[0]:6.1f}, {center[1]:6.1f}), "
                          f"conf={confidence:.1f}")
            
            # create visualizations
            print(f"\ncreating visualizations for spike mode: {mode}")
            visualize_interpolated_recording(
                train_loader, 
                save_dir=f'./visualization_10ms_{mode}', 
                recording_idx=0, 
                max_frames=500
            )
        else:
            print("no recordings found in dataset!")
    
    print(f"\ntesting complete!")