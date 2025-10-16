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
Script for converting suturing dataset to LeRobot format.

Usage:
python convert_dataset.py /path/to/suturing/data \
    [--repo_id REPO_ID] [--task_prompt TASK_PROMPT] [--image_shape IMAGE_SHAPE]

The script expects data in the format:
suturing_all/tissue_X/1_needle_pickup*/episode_timestamp/
where each episode contains:
- kinematics/ (zarr format with robot state and action data)
- endo_psm1/ (left endoscopic camera images)
- endo_psm2/ (right endoscopic camera images)
- left/ (left wrist camera images)
- right/ (right wrist camera images)

The resulting dataset will get saved to the $LEROBOT_HOME directory.
"""

import argparse
import glob
import os
import re
import shutil
import warnings

import h5py
import numpy as np
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

import zarr
import numpy as np
import simplejpeg
import numcodecs
from numcodecs import register_codec
from pathlib import Path

def _assert_shape(arr: np.ndarray, expected_shape: tuple[int | None, ...]):
    """Asserts that the shape of an array matches the expected shape."""
    assert len(arr.shape) == len(expected_shape), (arr.shape, expected_shape)
    for dim, expected_dim in zip(arr.shape, expected_shape):
        if expected_dim is not None:
            assert dim == expected_dim, (arr.shape, expected_shape)


class JpegCodec(numcodecs.abc.Codec):
    """Codec for JPEG compression.
    Encodes image chunks as JPEGs. Assumes that chunks are uint8 with shape (1, H, W, 3).
    """
    codec_id = "pi_jpeg"

    def __init__(self, quality: int = 95):
        super().__init__()
        self.quality = quality

    def encode(self, buf):
        _assert_shape(buf, (1, None, None, 3))
        assert buf.dtype == "uint8"
        return simplejpeg.encode_jpeg(buf[0], quality=self.quality)

    def decode(self, buf, out=None):
        img = simplejpeg.decode_jpeg(buf, buffer=out)
        return img[np.newaxis, ...]

register_codec(JpegCodec)

def read_kinematics(kin_path: str):
    """Reads kinematics data from a Zarr file."""
    kin = zarr.open(kin_path, mode='r')
    kin_arr = kin[:]
    kin_dict = {field: kin_arr[field] for field in kin_arr.dtype.names}
    # kin_dict = {field: kin[field][:] for field in kin.dtype.names}
    return kin_dict

def load_image_sequence(image_dir: str):
    """Load a sequence of images from a directory with numbered files."""
    images = []
    
    # Get all image files in the directory and sort them numerically
    image_files = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.0.0.0') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # Extract the numeric part for sorting
            try:
                if filename.endswith('.0.0.0'):
                    numeric_part = int(filename.split('.')[0])
                else:
                    numeric_part = int(filename.split('.')[0])
                image_files.append((numeric_part, filename))
            except ValueError:
                continue
    
    # Sort by numeric part
    image_files.sort(key=lambda x: x[0])
    
    # Load images in order
    for _, filename in image_files:
        file_path = os.path.join(image_dir, filename)
        try:
            img = Image.open(file_path)
            images.append(np.array(img))
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            continue
    
    return images

def extract_state_action_from_kinematics(kin_dict: dict, step: int):
    """Extract state and action from kinematics data at a specific step."""
    # Extract PSM1 and PSM2 joint states as current state
    psm1_js = np.array([kin_dict[f'psm1_js[{i}]'][step] for i in range(6)])
    psm2_js = np.array([kin_dict[f'psm2_js[{i}]'][step] for i in range(6)])
    
    # Extract jaw positions
    psm1_jaw = kin_dict['psm1_jaw'][step]
    psm2_jaw = kin_dict['psm2_jaw'][step]
    
    # Combine into state vector (12 joint states + 2 jaw positions)
    state = np.concatenate([psm1_js, psm2_js, [psm1_jaw, psm2_jaw]])
    
    # Extract PSM poses and jaw positions as actions
    if step + 1 < len(kin_dict['psm1_pose.position.x']):
        # PSM1 pose (position + orientation)
        psm1_pos = np.array([
            kin_dict['psm1_pose.position.x'][step + 1],
            kin_dict['psm1_pose.position.y'][step + 1],
            kin_dict['psm1_pose.position.z'][step + 1]
        ])
        psm1_orient = np.array([
            kin_dict['psm1_pose.orientation.x'][step + 1],
            kin_dict['psm1_pose.orientation.y'][step + 1],
            kin_dict['psm1_pose.orientation.z'][step + 1],
            kin_dict['psm1_pose.orientation.w'][step + 1]
        ])
        
        # PSM2 pose (position + orientation)
        psm2_pos = np.array([
            kin_dict['psm2_pose.position.x'][step + 1],
            kin_dict['psm2_pose.position.y'][step + 1],
            kin_dict['psm2_pose.position.z'][step + 1]
        ])
        psm2_orient = np.array([
            kin_dict['psm2_pose.orientation.x'][step + 1],
            kin_dict['psm2_pose.orientation.y'][step + 1],
            kin_dict['psm2_pose.orientation.z'][step + 1],
            kin_dict['psm2_pose.orientation.w'][step + 1]
        ])
        
        # Jaw positions
        psm1_jaw_action = kin_dict['psm1_jaw'][step + 1]
        psm2_jaw_action = kin_dict['psm2_jaw'][step + 1]
        
        # Combine into action vector (PSM1 pose + PSM2 pose + jaw positions)
        # Total: 3 + 4 + 3 + 4 + 1 + 1 = 16 dimensions
        action = np.concatenate([psm1_pos, psm1_orient, psm2_pos, psm2_orient, [psm1_jaw_action, psm2_jaw_action]])
    else:
        # For the last step, use current poses as action
        psm1_pos = np.array([
            kin_dict['psm1_pose.position.x'][step],
            kin_dict['psm1_pose.position.y'][step],
            kin_dict['psm1_pose.position.z'][step]
        ])
        psm1_orient = np.array([
            kin_dict['psm1_pose.orientation.x'][step],
            kin_dict['psm1_pose.orientation.y'][step],
            kin_dict['psm1_pose.orientation.z'][step],
            kin_dict['psm1_pose.orientation.w'][step]
        ])
        
        psm2_pos = np.array([
            kin_dict['psm2_pose.position.x'][step],
            kin_dict['psm2_pose.position.y'][step],
            kin_dict['psm2_pose.position.z'][step]
        ])
        psm2_orient = np.array([
            kin_dict['psm2_pose.orientation.x'][step],
            kin_dict['psm2_pose.orientation.y'][step],
            kin_dict['psm2_pose.orientation.z'][step],
            kin_dict['psm2_pose.orientation.w'][step]
        ])
        
        action = np.concatenate([psm1_pos, psm1_orient, psm2_pos, psm2_orient, [psm1_jaw, psm2_jaw]])
    
    return state.astype(np.float32), action.astype(np.float32)

def find_suturing_episodes(data_dir: str):
    """Find all suturing episodes in the directory structure."""
    episodes = []
    
    # Pattern: suturing_all/tissue_X/1_needle_pickup*/episode_timestamp/
    suturing_dir = os.path.join(data_dir, "suturing_all")
    if not os.path.exists(suturing_dir):
        suturing_dir = os.path.join(data_dir, "suturing_eval/")
        if not os.path.exists(suturing_dir):
            warnings.warn(f"Suturing directory {suturing_dir} does not exist")
            return episodes
        else:
            warnings.warn(f"Eval dataset generation")
    
    # Find all tissue directories
    for tissue_dir in os.listdir(suturing_dir):
        tissue_path = os.path.join(suturing_dir, tissue_dir)
        if not os.path.isdir(tissue_path) or not tissue_dir.startswith("tissue_"):
            continue
            
        # Find all needle pickup directories
        for pickup_dir in os.listdir(tissue_path):
            pickup_path = os.path.join(tissue_path, pickup_dir)
            if not os.path.isdir(pickup_path) or not pickup_dir.startswith("1_needle_pickup"):
                continue
                
            # Find all episode directories
            for episode_dir in os.listdir(pickup_path):
                episode_path = os.path.join(pickup_path, episode_dir)
                if not os.path.isdir(episode_path):
                    continue
                    
                # Check if required directories exist
                required_dirs = ["kinematics", "endo_psm1", "endo_psm2", "left", "right"]
                if all(os.path.exists(os.path.join(episode_path, d)) for d in required_dirs):
                    episodes.append({
                        "tissue_type": tissue_dir,
                        "pickup_type": pickup_dir,
                        "episode_id": episode_dir,
                        "path": episode_path
                    })
    
    return episodes

class BaseFeatureDict:
    action_key: str = "action"
    left_endo_image_key: str = "observation.images.left_endo"
    right_endo_image_key: str = "observation.images.right_endo"
    left_wrist_image_key: str = "observation.images.left_wrist"
    right_wrist_image_key: str = "observation.images.right_wrist"
    state_key: str = "observation.state"

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (224, 224, 3),
        state_shape: tuple[int, ...] = (7,),
        actions_shape: tuple[int, ...] = (6,),
    ):
        self.image_shape = image_shape
        self.state_shape = state_shape
        self.actions_shape = actions_shape

    @property
    def features(self):
        features_dict = {
            self.left_endo_image_key: {
                "dtype": "video",
                "shape": self.image_shape,
                "names": ["height", "width", "channels"],
            },
            self.right_endo_image_key: {
                "dtype": "video",
                "shape": self.image_shape,
                "names": ["height", "width", "channels"],
            },
            self.left_wrist_image_key: {
                "dtype": "video",
                "shape": self.image_shape,
                "names": ["height", "width", "channels"],
            },
            self.right_wrist_image_key: {
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

    def __call__(self, left_endo_img, right_endo_img, left_wrist_img, right_wrist_img, state, action, seg=None, depth=None) -> dict:
        frame_data = {}
        img_h, img_w, _ = self.image_shape
        current_features = self.features  # Access property to ensure it's evaluated

        # Assign mandatory fields
        frame_data[self.left_endo_image_key] = resize_with_pad(left_endo_img, img_h, img_w)
        frame_data[self.right_endo_image_key] = resize_with_pad(right_endo_img, img_h, img_w)
        frame_data[self.left_wrist_image_key] = resize_with_pad(left_wrist_img, img_h, img_w)
        frame_data[self.right_wrist_image_key] = resize_with_pad(right_wrist_img, img_h, img_w)
        frame_data[self.state_key] = state
        frame_data[self.action_key] = action
        return frame_data



def create_lerobot_dataset(
    output_path: str,
    features: dict,
    robot_type: str = "dvrk",
    fps: int = 30,
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
):
    """
    Creates a LeRobot dataset with specified configurations.

    This function initializes a LeRobot dataset with the given parameters,
    defining the structure and features of the dataset.

    Parameters:
    - output_path: The path where the dataset will be saved.
    - features: A dictionary defining the features of the dataset.
    - robot_type: The type of robot.
    - fps: Frames per second for the dataset.
    - image_writer_threads: Number of threads for image writing.
    - image_writer_processes: Number of processes for image writing.

    Returns:
    - An instance of LeRobotDataset configured with the specified parameters.
    """

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


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


def create_lerobot_dataset(
    output_path: str,
    features: dict,
    robot_type: str = "panda",
    fps: int = 30,
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
):
    """
    Creates a LeRobot dataset with specified configurations.

    This function initializes a LeRobot dataset with the given parameters,
    defining the structure and features of the dataset.

    Parameters:
    - output_path: The path where the dataset will be saved.
    - features: A dictionary defining the features of the dataset.
    - robot_type: The type of robot.
    - fps: Frames per second for the dataset.
    - image_writer_threads: Number of threads for image writing.
    - image_writer_processes: Number of processes for image writing.

    Returns:
    - An instance of LeRobotDataset configured with the specified parameters.
    """

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
    **dataset_config_kwargs,
):
    """
    Main function to convert suturing data to LeRobot format.

    This function processes suturing data in the specified directory structure,
    extracts relevant data from zarr files and images, and saves it in the LeRobot format.

    Parameters:
    - data_dir: Root directory containing the suturing data.
    - repo_id: Identifier for the dataset repository.
    - task_prompt: Description of the task for which the dataset is used.
    - include_depth: Whether to include depth images in the dataset.
    - include_seg: Whether to include segmentation images in the dataset.
    - run_compute_stats: Whether to run compute stats.
    - dataset_config_kwargs: Additional keyword arguments for dataset configuration.
    - feature_builder: An instance of a feature dictionary builder class (e.g., GR00TN1FeatureDict).
    """
    final_output_path = Path(repo_id)
    if final_output_path.exists():
        try:
            shutil.rmtree(final_output_path)
        except Exception as e:
            raise Exception(f"Error removing {final_output_path}: {e}. Please ensure that you have write permissions.")

    robot_type = dataset_config_kwargs.pop("robot_type", "dvrk")
    fps = dataset_config_kwargs.pop("fps", 30)
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

    # Find all suturing episodes
    episodes = find_suturing_episodes(data_dir)
    if not episodes:
        warnings.warn(f"No suturing episodes found in {data_dir}")
        return

    print(f"Found {len(episodes)} episodes to process")
    
    # Process each episode
    for episode_info in tqdm.tqdm(episodes):
        episode_path = episode_info["path"]
        print(f"Processing episode: {episode_info['tissue_type']}/{episode_info['pickup_type']}/{episode_info['episode_id']}")
        
        # Load kinematics data
        kin_path = os.path.join(episode_path, "kinematics")
        kin_dict = read_kinematics(kin_path)
        
        # Load image sequences
        left_endo_images = load_image_sequence(os.path.join(episode_path, "endo_psm1"))
        right_endo_images = load_image_sequence(os.path.join(episode_path, "endo_psm2"))
        left_wrist_images = load_image_sequence(os.path.join(episode_path, "left"))
        right_wrist_images = load_image_sequence(os.path.join(episode_path, "right"))
        
        # Check that all sequences have the same length
        num_steps = len(kin_dict['timestamp'])
        image_lengths = [len(left_endo_images), len(right_endo_images), len(left_wrist_images), len(right_wrist_images)]
        
        if not all(length >= num_steps for length in image_lengths):
            warnings.warn(f"Image sequence lengths {image_lengths} are shorter than kinematics steps {num_steps} in {episode_path}")
            # Use the minimum length
            num_steps = min(num_steps, min(image_lengths))
        
        print(f"Processing {num_steps} steps for episode {episode_info['episode_id']}")
        
        # Process each step
        for step in range(num_steps):
            try:
                # Extract state and action from kinematics
                state, action = extract_state_action_from_kinematics(kin_dict, step)
                
                # Get images for this step
                left_endo_img = left_endo_images[step]
                right_endo_img = right_endo_images[step]
                left_wrist_img = left_wrist_images[step]
                right_wrist_img = right_wrist_images[step]
                
                # Create frame dictionary
                frame_dict = feature_builder(
                    left_endo_img=left_endo_img,
                    right_endo_img=right_endo_img,
                    left_wrist_img=left_wrist_img,
                    right_wrist_img=right_wrist_img,
                    state=state,
                    action=action
                )
                # Add task to the frame
                episode_task = f"{task_prompt}"
                dataset.add_frame(frame_dict, task=episode_task)
                
            except Exception as e:
                warnings.warn(f"Error processing step {step} in episode {episode_info['episode_id']}: {e}")
                continue
        
        # Save episode
        dataset.save_episode()

    print(f"Saving dataset to {final_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 files to LeRobot format")
    parser.add_argument("--data_dir", type=str, default='/home/projects/healthcareeng_monai/datasets/JHU_data/suturing_eval')
    parser.add_argument(
        "--repo_id",
        type=str,
        default="/home/projects/healthcareeng_monai/JHU_needle_grasping_test",
        help="Directory to save the dataset under (relative to LEROBOT_HOME)",
    )
    parser.add_argument(
        "--task_prompt",
        type=str,
        default="Perform robotic suturing with needle pickup.",
        help="Prompt description of the task",
    )
    parser.add_argument(
        "--image_shape",
        type=lambda s: tuple(map(int, s.split(","))),
        default=(224, 224, 3),
        help="Shape of the image data as a comma-separated string, e.g., '224,224,3'",
    )

    args = parser.parse_args()

    # Instantiate the feature builder based on args
    # For suturing data: state has 14 dimensions (12 joint states + 2 jaw positions)
    # Action has 16 dimensions (PSM1 pose: 3+4, PSM2 pose: 3+4, jaw positions: 2)
    feature_builder = BaseFeatureDict(
        image_shape=args.image_shape,
        state_shape=(14,),  # 12 joint states + 2 jaw positions
        actions_shape=(16,),  # PSM1 pose (7) + PSM2 pose (7) + jaw positions (2)
    )
    main(
        args.data_dir,
        args.repo_id,
        args.task_prompt,
        feature_builder=feature_builder
    )