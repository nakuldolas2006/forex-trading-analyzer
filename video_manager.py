import cv2
import numpy as np
import os
import threading
import time
from datetime import datetime
from typing import List, Optional, Dict
import json

class VideoManager:
    """Handles video recording, storage, and playback"""
    
    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = output_dir
        self.is_recording = False
        self.current_writer = None
        self.current_filename = None
        self.recording_lock = threading.Lock()
        self.frame_buffer = []
        self.max_buffer_size = 300  # 30 seconds at 10 FPS
        
        # Video settings
        self.fps = 10
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.quality = 80  # Compression quality
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Metadata storage
        self.recordings_metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load recordings metadata from file"""
        metadata_file = os.path.join(self.output_dir, "metadata.json")
        try:
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save recordings metadata to file"""
        metadata_file = os.path.join(self.output_dir, "metadata.json")
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.recordings_metadata, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def start_recording(self, filename: Optional[str] = None) -> bool:
        """Start video recording"""
        try:
            with self.recording_lock:
                if self.is_recording:
                    return False
                
                # Generate filename if not provided
                if not filename:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"recording_{timestamp}.mp4"
                
                self.current_filename = filename
                filepath = os.path.join(self.output_dir, filename)
                
                # Initialize video writer (will be set up when first frame arrives)
                self.current_writer = None
                self.is_recording = True
                self.frame_buffer = []
                
                # Add to metadata
                self.recordings_metadata[filename] = {
                    'start_time': datetime.now(),
                    'end_time': None,
                    'duration': 0,
                    'frame_count': 0,
                    'file_size': 0,
                    'filepath': filepath
                }
                
                return True
                
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False
    
    def stop_recording(self) -> bool:
        """Stop video recording"""
        try:
            with self.recording_lock:
                if not self.is_recording:
                    return False
                
                self.is_recording = False
                
                # Finalize video file
                if self.current_writer:
                    self.current_writer.release()
                    self.current_writer = None
                
                # Update metadata
                if self.current_filename in self.recordings_metadata:
                    metadata = self.recordings_metadata[self.current_filename]
                    metadata['end_time'] = datetime.now()
                    
                    if metadata['start_time']:
                        duration = metadata['end_time'] - metadata['start_time']
                        metadata['duration'] = duration.total_seconds()
                    
                    # Get file size
                    filepath = metadata['filepath']
                    if os.path.exists(filepath):
                        metadata['file_size'] = os.path.getsize(filepath)
                
                self._save_metadata()
                self.current_filename = None
                
                return True
                
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return False
    
    def add_frame(self, frame: np.ndarray) -> bool:
        """Add a frame to the current recording"""
        if not self.is_recording or frame is None:
            return False
        
        try:
            with self.recording_lock:
                # Initialize video writer on first frame
                if self.current_writer is None and self.current_filename:
                    height, width = frame.shape[:2]
                    filepath = os.path.join(self.output_dir, self.current_filename)
                    
                    self.current_writer = cv2.VideoWriter(
                        filepath, self.codec, self.fps, (width, height)
                    )
                    
                    if not self.current_writer.isOpened():
                        print("Failed to open video writer")
                        return False
                
                # Write frame
                if self.current_writer:
                    self.current_writer.write(frame)
                    
                    # Update frame count in metadata
                    if self.current_filename in self.recordings_metadata:
                        self.recordings_metadata[self.current_filename]['frame_count'] += 1
                
                # Add to buffer for real-time playback
                self.frame_buffer.append(frame.copy())
                if len(self.frame_buffer) > self.max_buffer_size:
                    self.frame_buffer.pop(0)
                
                return True
                
        except Exception as e:
            print(f"Error adding frame: {e}")
            return False
    
    def get_recordings(self) -> List[str]:
        """Get list of available recordings"""
        try:
            recordings = []
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.mp4') and filename in self.recordings_metadata:
                    recordings.append(filename)
            return sorted(recordings, reverse=True)  # Most recent first
        except Exception as e:
            print(f"Error getting recordings: {e}")
            return []
    
    def get_recording_info(self, filename: str) -> Optional[Dict]:
        """Get information about a specific recording"""
        return self.recordings_metadata.get(filename)
    
    def play_recording(self, filename: str) -> bool:
        """Play a recorded video (returns frames for display)"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            if not os.path.exists(filepath):
                return False
            
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return False
            
            # This would be used in a GUI to display frames
            # For now, just validate the video can be opened
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            return frame_count > 0
            
        except Exception as e:
            print(f"Error playing recording: {e}")
            return False
    
    def get_video_frames(self, filename: str, start_frame: int = 0, count: int = 30) -> List[np.ndarray]:
        """Get specific frames from a recording"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            if not os.path.exists(filepath):
                return []
            
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return []
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames = []
            for _ in range(count):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame.copy())
            
            cap.release()
            return frames
            
        except Exception as e:
            print(f"Error getting video frames: {e}")
            return []
    
    def create_highlight_reel(self, trades: List[Dict], duration_per_trade: int = 10) -> Optional[str]:
        """Create a highlight reel of trading moments"""
        try:
            if not trades:
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            highlight_filename = f"highlights_{timestamp}.mp4"
            highlight_path = os.path.join(self.output_dir, highlight_filename)
            
            # Find recordings that contain the trade timestamps
            trade_recordings = {}
            for trade in trades:
                trade_time = trade.get('timestamp')
                if not trade_time:
                    continue
                
                # Find recording that contains this timestamp
                for recording_name, metadata in self.recordings_metadata.items():
                    start_time = metadata.get('start_time')
                    end_time = metadata.get('end_time')
                    
                    if start_time and end_time and start_time <= trade_time <= end_time:
                        if recording_name not in trade_recordings:
                            trade_recordings[recording_name] = []
                        trade_recordings[recording_name].append(trade)
                        break
            
            if not trade_recordings:
                return None
            
            # Create highlight video
            first_recording = list(trade_recordings.keys())[0]
            first_cap = cv2.VideoCapture(os.path.join(self.output_dir, first_recording))
            
            if not first_cap.isOpened():
                return None
            
            # Get video properties
            width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            first_cap.release()
            
            # Create highlight video writer
            highlight_writer = cv2.VideoWriter(
                highlight_path, self.codec, self.fps, (width, height)
            )
            
            if not highlight_writer.isOpened():
                return None
            
            # Add segments for each trade
            for recording_name, recording_trades in trade_recordings.items():
                cap = cv2.VideoCapture(os.path.join(self.output_dir, recording_name))
                
                for trade in recording_trades:
                    # Calculate frame position for trade timestamp
                    # This is simplified - in production, you'd need precise timing
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frames_to_extract = min(duration_per_trade * self.fps, total_frames)
                    
                    # Extract frames around trade time
                    for _ in range(frames_to_extract):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Add trade information overlay
                        self._add_trade_overlay(frame, trade)
                        highlight_writer.write(frame)
                
                cap.release()
            
            highlight_writer.release()
            
            # Add to metadata
            self.recordings_metadata[highlight_filename] = {
                'start_time': datetime.now(),
                'end_time': datetime.now(),
                'duration': len(trades) * duration_per_trade,
                'frame_count': len(trades) * duration_per_trade * self.fps,
                'file_size': os.path.getsize(highlight_path) if os.path.exists(highlight_path) else 0,
                'filepath': highlight_path,
                'type': 'highlight_reel',
                'trade_count': len(trades)
            }
            
            self._save_metadata()
            return highlight_filename
            
        except Exception as e:
            print(f"Error creating highlight reel: {e}")
            return None
    
    def _add_trade_overlay(self, frame: np.ndarray, trade: Dict):
        """Add trade information overlay to frame"""
        try:
            # Add trade info text
            text = f"Trade: {trade.get('action', 'N/A')} | P&L: ${trade.get('profit_loss', 0):.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            confidence = trade.get('confidence', 0)
            conf_text = f"Confidence: {confidence:.1%}"
            cv2.putText(frame, conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
        except Exception as e:
            print(f"Error adding trade overlay: {e}")
    
    def download_recording(self, filename: str) -> Optional[str]:
        """Prepare recording for download"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                return filepath
            return None
        except Exception as e:
            print(f"Error preparing download: {e}")
            return None
    
    def delete_recording(self, filename: str) -> bool:
        """Delete a recording and its metadata"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            
            # Delete file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Remove from metadata
            if filename in self.recordings_metadata:
                del self.recordings_metadata[filename]
                self._save_metadata()
            
            return True
            
        except Exception as e:
            print(f"Error deleting recording: {e}")
            return False
    
    def get_storage_info(self) -> Dict:
        """Get storage usage information"""
        try:
            total_size = 0
            file_count = 0
            
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.mp4'):
                    filepath = os.path.join(self.output_dir, filename)
                    total_size += os.path.getsize(filepath)
                    file_count += 1
            
            return {
                'total_files': file_count,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'recording_active': self.is_recording
            }
            
        except Exception as e:
            print(f"Error getting storage info: {e}")
            return {'total_files': 0, 'total_size_bytes': 0, 'total_size_mb': 0, 'recording_active': False}
