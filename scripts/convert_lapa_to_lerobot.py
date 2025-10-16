# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script for converting needle grasping dataset to LeRobot format.

Usage:
python convert_needle_grasping_data.py /path/to/needle/grasping/data \
    [--repo_id REPO_ID] [--task_prompt TASK_PROMPT] [--image_shape IMAGE_SHAPE]

The script expects data in the format:
needleGrasping_resized_93x1280x704_16fps/episode_name/
where each episode contains:
- episode_name_frame_XXXX.png (image frames)
- episode_name_frame_XXXX.json (kinematics data as 16-element arrays)

The resulting dataset will get saved to the specified output directory.
"""

import argparse
import glob
import json
import os
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image


def load_codebook(codebook_path: str):
    """Load the codebook from JSON file."""
    with open(codebook_path, 'r') as f:
        codebook_data = json.load(f)
    # Convert to numpy array - should be shape [8, 32]
    codebook = np.array(codebook_data, dtype=np.float32)
    print(f"Loaded codebook with shape: {codebook.shape}")
    return codebook


def convert_indices_to_actions(indices: list, codebook: np.ndarray):
    """Convert list of indices to action vectors using codebook lookup."""
    # indices should be 16 integers from 1-8
    # codebook should be shape [8, 32]
    action_vectors = []
    for idx in indices:
        # Convert from 1-based to 0-based indexing
        codebook_idx = idx 
        if codebook_idx < 0 or codebook_idx >= len(codebook):
            raise ValueError(f"Index {idx} is out of range for codebook of size {len(codebook)}")
        action_vectors.append(codebook[codebook_idx])
    
    # Concatenate all 16 vectors to get 512-dimensional action
    action = np.concatenate(action_vectors)
    return action


def _assert_shape(arr: np.ndarray, expected_shape: tuple[int | None, ...]):
    """Asserts that the shape of an array matches the expected shape."""
    assert len(arr.shape) == len(expected_shape), (arr.shape, expected_shape)
    for dim, expected_dim in zip(arr.shape, expected_shape):
        if expected_dim is not None:
            assert dim == expected_dim, (arr.shape, expected_shape)


def load_frame_data(episode_dir: str, codebook: np.ndarray):
    """Load frame images and kinematics from an episode directory."""
    episode_name = os.path.basename(episode_dir)
    
    # Find all PNG and JSON files
    png_files = glob.glob(os.path.join(episode_dir, f"*frame_*.png"))
    json_files = glob.glob(os.path.join(episode_dir, f"*frame_*.json"))
    
    # Sort files by frame number
    def get_frame_number(filename):
        # Extract frame number from filename like "episode_frame_0024.png"
        match = re.search(r'frame_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    png_files.sort(key=get_frame_number)
    json_files.sort(key=get_frame_number)
    
    if len(png_files) != len(json_files):
        warnings.warn(f"Mismatch in number of PNG ({len(png_files)}) and JSON ({len(json_files)}) files in {episode_dir}")
        # Use the minimum length
        min_length = min(len(png_files), len(json_files))
        png_files = png_files[:min_length]
        json_files = json_files[:min_length]
    
    frames_data = []
    for png_file, json_file in zip(png_files, json_files):
        # Load image
        try:
            image = Image.open(png_file)
            image_array = np.array(image)
        except Exception as e:
            warnings.warn(f"Error loading image {png_file}: {e}")
            continue
            
        # Load kinematics indices and convert to action vectors
        try:
            with open(json_file, 'r') as f:
                indices = json.load(f)
                # Convert indices to action vectors using codebook
                action = convert_indices_to_actions(indices, codebook)
        except Exception as e:
            warnings.warn(f"Error loading kinematics {json_file}: {e}")
            continue
            
        frames_data.append({
            'image': image_array,
            'action': action
        })
    
    return frames_data


def find_needle_grasping_episodes(data_dir: str):
    """Find all needle grasping episodes in the directory structure."""
    episodes = []
    
    # Find all episode directories
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) : #and "needleGrasping" in item:
            # Check if it contains PNG and JSON files
            png_files = glob.glob(os.path.join(item_path, "*.png"))
            json_files = glob.glob(os.path.join(item_path, "*.json"))
            
            if png_files and json_files:
                episodes.append({
                    "episode_id": item,
                    "path": item_path
                })
    
    return episodes


class NeedleGraspingFeatureDict:
    """Feature dictionary for needle grasping dataset."""
    action_key: str = "action"
    image_key: str = "observation.images.top"
    state_key: str = "observation.state"

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (224, 224, 3),
        state_shape: tuple[int, ...] = (512,),  # Use action as state for simplicity
        actions_shape: tuple[int, ...] = (512,),  # 16 indices * 32-dim vectors = 512
    ):
        self.image_shape = image_shape
        self.state_shape = state_shape
        self.actions_shape = actions_shape

    @property
    def features(self):
        features_dict = {
            self.image_key: {
                "dtype": "video",
                "shape": self.image_shape,
                "names": ["height", "width", "channels"],
            },
            self.state_key: {
                "dtype": "float32",
                "shape": self.state_shape,
                "names": ["state"],
            },
            self.action_key: {
                "dtype": "float32",
                "shape": self.actions_shape,
                "names": ["action"],
            },
        }
        return features_dict

    def __call__(self, image, state, action) -> dict:
        frame_data = {}
        img_h, img_w, _ = self.image_shape
        
        # Resize image to target shape
        frame_data[self.image_key] = resize_with_pad(image, img_h, img_w)
        frame_data[self.state_key] = state
        frame_data[self.action_key] = action
        
        return frame_data


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL."""
    # If input is a single image, add batch dimension
    if len(images.shape) == 3:
        images = images[np.newaxis, ...]
    
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images.squeeze(0) if images.shape[0] == 1 else images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    
    result = resized.reshape(*original_shape[:-3], *resized.shape[-3:])
    return result.squeeze(0) if result.shape[0] == 1 else result


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> np.ndarray:
    """Resize an image with padding using PIL."""
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.array(image)

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return np.array(zero_image)


def create_lerobot_dataset(
    output_path: str,
    features: dict,
    robot_type: str = "surgical_robot",
    fps: int = 16,
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
):
    """Create a LeRobot dataset with specified configurations."""
    if os.path.isdir(output_path):
        raise Exception(f"Output path {output_path} already exists.")

    return LeRobotDataset.create(
        repo_id=output_path,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


def main(
    data_dir: str,
    repo_id: str,
    task_prompt: str,
    feature_builder,
    codebook_path: str,
    **dataset_config_kwargs,
):
    """
    Main function to convert needle grasping data to LeRobot format.
    """
    final_output_path = Path(repo_id)
    if final_output_path.exists():
        try:
            shutil.rmtree(final_output_path)
        except Exception as e:
            raise Exception(f"Error removing {final_output_path}: {e}. Please ensure that you have write permissions.")

    # Load the codebook
    print(f"Loading codebook from {codebook_path}")
    codebook = load_codebook(codebook_path)

    robot_type = dataset_config_kwargs.pop("robot_type", "surgical_robot")
    fps = dataset_config_kwargs.pop("fps", 16)
    image_writer_threads = dataset_config_kwargs.pop("image_writer_threads", 10)
    image_writer_processes = dataset_config_kwargs.pop("image_writer_processes", 5)

    dataset = create_lerobot_dataset(
        output_path=final_output_path,
        features=feature_builder.features,
        robot_type=robot_type,
        fps=fps,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    # Find all needle grasping episodes
    episodes = find_needle_grasping_episodes(data_dir)
    if not episodes:
        warnings.warn(f"No needle grasping episodes found in {data_dir}")
        return

    print(f"Found {len(episodes)} episodes to process")
    
    # Process each episode
    for episode_info in tqdm.tqdm(episodes):
        episode_path = episode_info["path"]
        episode_id = episode_info["episode_id"]
        print(f"Processing episode: {episode_id}")
        
        try:
            # Load frame data for this episode
            frames_data = load_frame_data(episode_path, codebook)
            
            if not frames_data:
                warnings.warn(f"No valid frames found in episode {episode_id}")
                continue
                
            print(f"Processing {len(frames_data)} frames for episode {episode_id}")
            
            # Process each frame
            for i, frame_data in enumerate(frames_data):
                try:
                    image = frame_data['image']
                    action = frame_data['action']
                    # For simplicity, use current action as state (you may want to modify this)
                    state = action.copy()
                    
                    # Create frame dictionary
                    frame_dict = feature_builder(
                        image=image,
                        state=state,
                        action=action
                    )
                    
                    # Add task to the frame
                    dataset.add_frame(frame_dict, task=task_prompt)
                    
                except Exception as e:
                    warnings.warn(f"Error processing frame {i} in episode {episode_id}: {e}")
                    continue
            
            # Save episode
            dataset.save_episode()
            
        except Exception as e:
            warnings.warn(f"Error processing episode {episode_id}: {e}")
            continue

    print(f"Saving dataset to {final_output_path}")

# python convert_lapa_to_lerobot.py --data_dir /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_121x1280x704_24fps_singleview/cam_high_images_infer_lapa/ --repo_id /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_121x1280x704_24fps_singleview_lerobot_offset1 --task_prompt "The left arm of the surgical robot is picking up a needle over a red rubber pad and handing it over to the right arm." --image_shape 224,224,3 --codebook_path vae.4000_codebooks.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert needle grasping dataset to LeRobot format")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default='/home/projects/healthcareeng_monai/datasets/BSA/needleGrasping_resized_93x1280x704_16fps',
        help="Root directory containing the needle grasping episode directories"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="/home/projects/healthcareeng_monai/datasets/BSA/needleGrasping_resized_93x1280x704_16fps_lerobot_offset1",
        help="Directory to save the dataset under",
    )
    parser.add_argument(
        "--task_prompt",
        type=str,
        default="The left arm of the surgical robot is picking up a needle over a red rubber pad and handing it over to the right arm.",
        help="Prompt description of the task",
    )
    parser.add_argument(
        "--image_shape",
        type=lambda s: tuple(map(int, s.split(","))),
        default=(224, 224, 3),
        help="Shape of the image data as a comma-separated string, e.g., '224,224,3'",
    )
    parser.add_argument(
        "--codebook_path",
        type=str,
        default='codebook_1.json',
        help="Path to the codebook.json file containing the [8, 32] codebook",
    )

    args = parser.parse_args()

    # Instantiate the feature builder
    # Action has 512 dimensions (16 indices * 32-dim vectors), use same for state
    feature_builder = NeedleGraspingFeatureDict(
        image_shape=args.image_shape,
        state_shape=(512,),  # Use same as action
        actions_shape=(512,),  # 16 indices * 32-dim codebook vectors = 512
    )
    
    main(
        args.data_dir,
        args.repo_id,
        args.task_prompt,
        feature_builder=feature_builder,
        codebook_path=args.codebook_path
    ) 