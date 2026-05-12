"""LIBERO_Decomposed_Oversample — adds NaVILA-style stop-signal oversampling.

Same per-subtask split as LIBERO_Decomposed, but appends duplicated terminal
frames so the VLA can learn the binary stop signal `s_t` precisely:
  - gripper subtasks:     repeat the last frame x8
  - non-gripper subtasks: repeat the last 3 frames, each x4 (12 extra frames)
All appended frames have is_last/is_terminal=True, reward=1.0.

`is_terminal` is still a plain bool — no progress signal is emitted. The
language_instruction is already subtask-only (the high-level prefix at line
51 is commented out), so this builder targets decomposed_dataset/libero_sub_oversample.

Intermediate variant; superseded by LIBERO_Decomposed_Progress, which
additionally emits the float progress signal `p_t`.
"""

from typing import Iterator, Tuple, Any

import os
import h5py
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import json
from copy import deepcopy
from LIBERO_Decomposed_Oversample.conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields decomposed RLDS episodes from LIBERO from a list of per-episode paths."""
    
    def _parse_example(path_tuple):
        episode_idx, original_rlds_dir, task_suite, process_path = path_tuple
        episode_id = str(episode_idx)
        ds = tfds.load(task_suite, data_dir=original_rlds_dir, split="train")
        episode = list(tfds.as_numpy(ds))[episode_idx]
        steps = list(episode["steps"])
        print()
        print(task_suite, episode_idx)
        print()

        chunks_json_path = next((p for p in [
            os.path.join(process_path, task_suite, f"episode_{episode_id}", "chunks_summary.json"),
            os.path.join(process_path, task_suite, f"episode_{episode_id}_llm", "chunks_summary.json")
        ] if os.path.exists(p)), None)

        with open(chunks_json_path, "r") as f:
            episode_data = json.load(f)

        language_instruction_high_level = episode_data["metadata"]["language_instruction"]
        file_path = episode_data["metadata"]["file_path"]

        results = []
        for j, chunk in enumerate(episode_data["chunks"]):
            if 'subtask' not in chunk:
                continue

            start, end = chunk["start"], chunk["end"]
            if end >= len(steps): continue
            sub_steps = steps[start:end + 1]

            new_steps = []
            for k, step in enumerate(sub_steps):
                new_step = {k_: step[k_] for k_ in step}
                # new_step["language_instruction"] = f"Task: {language_instruction_high_level}. The current subtask: {chunk['subtask'].lower()}"
                new_step["language_instruction"] = chunk['subtask'].lower()
                
                new_step["is_first"] = (k == 0)
                new_step["is_last"] = (k == len(sub_steps) - 1)
                new_step["is_terminal"] = new_step["is_last"]
                new_step["reward"] = 1.0 if new_step["is_last"] else 0.0
                new_steps.append(new_step)
            
            # IMPORTANT, this must be no less than NUM_ACTIONS_CHUNK (8)
            # if len(new_steps) < 8:
            #     continue 

            # Oversampling logic for better stop prediction
            language_instruction_subtask = chunk['subtask'].lower()

            if "gripper" in language_instruction_subtask:
                repeat_steps = [deepcopy(new_steps[-1])] * 8
            else:
                repeat_base = new_steps[-3:] if len(new_steps) >= 3 else new_steps[-1:]
                repeat_steps = []
                for s in repeat_base:
                    repeat_steps.extend([deepcopy(s)] * 4)

            for s in repeat_steps:
                s["is_last"] = True
                s["is_terminal"] = True
                s["reward"] = 1.0

            new_steps.extend(repeat_steps)

            metadata = {"file_path": f"{file_path}___{task_suite}_{episode_id}_subtask_{j}"} 
            results.append((f"{task_suite}_{episode_id}_subtask_{j}", {
                "steps": new_steps,
                "episode_metadata": metadata,
            }))

        return results

    for path_tuple in paths:
        parsed_results = _parse_example(path_tuple)
        if parsed_results is not None:
            for item in parsed_results:
                yield item


class LIBERODecomposedOversample(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      "1.0.0": "Initial release combining all 4 LIBERO suites with subtask decomposition.",
    }
    N_WORKERS = 4              # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 8    # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot EEF state (6D pose, 2D gripper).',
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint angles.',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot EEF action.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))


    # conda activate /hdd2/kai/openvla-oft/env
    # cd /hdd2/kai/openvla-oft/rlds_dataset_builder/LIBERO_Decomposed_Oversample
    # CUDA_VISIBLE_DEVICES="-1" tfds build --data_dir=/hdd2/kai/openvla-oft/decomposed_dataset/libero_oversample --overwrite
    # CUDA_VISIBLE_DEVICES="-1" tfds build --data_dir=/hdd2/kai/openvla-oft/decomposed_dataset/libero_sub_oversample --overwrite

    def _split_paths(self):
        """Define filepaths for data splits."""
        original_rlds_dir = "/hdd2/kai/openvla-oft/LIBERO/libero/libero/modified_libero_rlds"
        task_suites = ["libero_spatial_no_noops", "libero_object_no_noops", "libero_goal_no_noops", "libero_10_no_noops"]
        process_path = "/hdd2/kai/openvla-oft/vlm_response/process_traj/libero"

        episode_configs = []
        for task_suite in task_suites:
            summary_file = os.path.join(process_path, task_suite, "all_episodes_summary.json")
            if not os.path.exists(summary_file):
                continue
            with open(summary_file, "r") as f:
                summary = json.load(f)
            valid_episode_ids = [int(eid) for eid in summary["episodes"].keys()]
            for i in valid_episode_ids:
                # if i > 5: continue
                episode_configs.append((i, original_rlds_dir, task_suite, process_path))

        return {"train": episode_configs}