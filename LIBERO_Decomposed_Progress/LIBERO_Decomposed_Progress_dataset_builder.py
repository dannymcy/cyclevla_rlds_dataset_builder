"""LIBERO_Decomposed_Progress — official CycleVLA subtask-finetuning builder.

Produces the supervision targets for the paper's extended 9-dim action
    a_t = [Δx, Δy, Δz, Δu, Δv, Δw, γ, s, p]
where `s` is the binary stop signal and `p ∈ [0.1, 1.0]` is the per-step
subtask progress (discretised in 0.1 bins).

Encoding:
  - `p_t` is carried in the `is_terminal` field, dtype widened to float32
    (schema bump v1.1.0). Body steps get progress 0.1-0.9; the NaVILA-style
    oversampled tail (gripper: last frame x8; non-gripper: last 3 frames x4)
    is pinned to 1.0 and also flags is_last=True / reward=1.0, which is what
    the VLA learns as `s_t`.
  - language_instruction is hard-coded to the subtask string, so this builder
    only ever produces decomposed_dataset/libero_sub_progress.

This is the variant whose checkpoints live in
checkpoints/libero/libero_sub_decomposed_progress_A100/.
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
from LIBERO_Decomposed_Progress.conversion_utils import MultiThreadedDatasetBuilder


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

        # `_manual` is the recovery path for episodes that Stage 2's FSM
        # filter (pick_place_to_discard in decompose_utils.py) rejected and
        # were re-labelled by hand. Same schema as the auto-generated files.
        chunks_json_path = next((p for p in [
            os.path.join(process_path, task_suite, f"episode_{episode_id}", "chunks_summary.json"),
            os.path.join(process_path, task_suite, f"episode_{episode_id}_llm", "chunks_summary.json"),
            os.path.join(process_path, task_suite, f"episode_{episode_id}_manual", "chunks_summary.json"),
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
            if end >= len(steps):
                continue
            sub_steps = steps[start:end + 1]
            
            if len(sub_steps) == 0:
                continue

            new_steps = []
            L = len(sub_steps)
            language_instruction_subtask = chunk['subtask'].lower()
            
            # Determine how many frames to exclude from progress 0.1-0.9
            if "gripper" in language_instruction_subtask:
                frames_to_oversample = 1  # Last 1 frame
            else:
                frames_to_oversample = min(3, L)  # Last 3 frames (or fewer if episode is short)
            
            # Add steps that get progress 0.1-0.9 (excluding the ones that will be oversampled)
            frames_for_progress = L - frames_to_oversample
            
            for k in range(frames_for_progress):
                step = sub_steps[k]
                new_step = {k_: step[k_] for k_ in step}
                new_step["language_instruction"] = language_instruction_subtask
                new_step["is_first"] = (k == 0)
                new_step["is_last"] = False  # These are not terminal
                new_step["reward"] = 0.0
                
                # Progress from 0.1 to 0.9
                raw_progress = (k + 1) / frames_for_progress if frames_for_progress > 0 else 0.5
                progress = min(0.9, max(0.1, round(raw_progress * 10) / 10))
                new_step["is_terminal"] = progress
                
                new_steps.append(new_step)
            
            # Add oversampled frames (these get progress 1.0)
            if "gripper" in language_instruction_subtask:
                # For gripper: repeat last frame 8 times
                last_frame = sub_steps[-1]
                for _ in range(8):
                    new_step = {k_: last_frame[k_] for k_ in last_frame}
                    new_step["language_instruction"] = language_instruction_subtask
                    new_step["is_first"] = False
                    new_step["is_last"] = True
                    new_step["is_terminal"] = 1.0
                    new_step["reward"] = 1.0
                    new_steps.append(new_step)
            else:
                # For non-gripper: repeat last 3 frames, each 4 times
                frames_to_repeat = sub_steps[-frames_to_oversample:]
                for frame in frames_to_repeat:
                    for _ in range(4):
                        new_step = {k_: frame[k_] for k_ in frame}
                        new_step["language_instruction"] = language_instruction_subtask
                        new_step["is_first"] = False
                        new_step["is_last"] = True
                        new_step["is_terminal"] = 1.0
                        new_step["reward"] = 1.0
                        new_steps.append(new_step)
            
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
                

class LIBERODecomposedProgress(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      "1.1.0": "is_terminal changed from bool to float progress (0..1].",
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
                        dtype=np.float32,
                        doc='Progress in (0,1], 1.0 at completion.'
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
    # cd /hdd2/kai/openvla-oft/rlds_dataset_builder/LIBERO_Decomposed_Progress
    # CUDA_VISIBLE_DEVICES="-1" tfds build --data_dir=/hdd2/kai/openvla-oft/decomposed_dataset/libero_sub_progress --overwrite

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
            valid_episode_ids = {int(eid) for eid in summary["episodes"].keys()}

            # Pick up hand-labelled episodes that aren't in all_episodes_summary.json.
            # These are FSM-filtered episodes the user resurrected by dropping a
            # chunks_summary.json into `episode_<N>_manual/`. We scan the disk so
            # the user doesn't have to edit the summary file.
            suite_dir = os.path.join(process_path, task_suite)
            for entry in os.listdir(suite_dir):
                if entry.startswith("episode_") and entry.endswith("_manual"):
                    eid_str = entry[len("episode_"):-len("_manual")]
                    if eid_str.isdigit() and os.path.exists(
                        os.path.join(suite_dir, entry, "chunks_summary.json")
                    ):
                        valid_episode_ids.add(int(eid_str))

            for i in sorted(valid_episode_ids):
                # if i > 5: continue
                episode_configs.append((i, original_rlds_dir, task_suite, process_path))

        return {"train": episode_configs}