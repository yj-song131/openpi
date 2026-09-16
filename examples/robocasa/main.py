import collections
import dataclasses
import logging
import math
import pathlib
import pickle
import imageio
from datetime import datetime
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import json
import os
from copy import deepcopy
from robocasa.utils.dataset_registry_utils import get_task_horizon

import robocasa.utils.robomimic.robomimic_dataset_utils as FileUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils
import robocasa
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_ds_meta
import gymnasium as gym
from robocasa.utils.env_utils import convert_action


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    split: str = "pretrain"
    num_trials: int = 50  # Number of rollouts per task
    # Task selection: either a TASK_SET_REGISTRY key or an explicit comma-separated list of task names.
    task_set: str | None = None
    tasks: str | None = None  # e.g. "TaskA,TaskB,TaskC"

    # Original-paper-style evaluation: cycle through 5 fixed (layout, style) combos.
    # When True, ignores split for layout selection and uses ORIG_EVAL_LAYOUTS below.
    # num_trials should equal len(ORIG_EVAL_LAYOUTS) * episodes_per_layout.
    eval_fixed_layouts: bool = False
    episodes_per_layout: int = 5  # episodes per layout when eval_fixed_layouts=True

    #################################################################################################################
    # Utils
    #################################################################################################################
    log_dir: str | None = None
    # When set, saves env_records/ flat under this root (SAFE-compatible).
    # The serve_policy server's --record-dir should point to <rollout_root>/policy_records/.
    rollout_root: str | None = None
    # Offset added to task_id so different groups don't collide (e.g. val_unseen starts at 100).
    task_id_offset: int = 0

    seed: int = 7  # Random Seed (for reproducibility)
    episode_start: int = 0  # Start from this episode index (for parallel collection)


def eval_main(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    split = args.split
    log_dir = args.log_dir
    num_trials = args.num_trials
    resize_size = args.resize_size
    replan_steps = args.replan_steps
    host = args.host
    port = args.port

    if args.tasks is not None:
        all_env_names = [t.strip() for t in args.tasks.split(",")]
    elif args.task_set is not None:
        all_env_names = TASK_SET_REGISTRY[args.task_set]
    else:
        raise ValueError("Provide --tasks or --task-set")

    for local_idx, env_name in enumerate(all_env_names):
        task_id = local_idx + args.task_id_offset
        eval_env(
            env_name,
            task_id,
            split,
            log_dir,
            num_trials,
            resize_size,
            replan_steps,
            host,
            port,
            args.seed,
            rollout_root=args.rollout_root,
            eval_fixed_layouts=args.eval_fixed_layouts,
            episodes_per_layout=args.episodes_per_layout,
            episode_start=args.episode_start,
        )


# 5 (layout, style) pairs from the original robocasa evaluation protocol.
ORIG_EVAL_LAYOUTS = [(1, 1), (2, 2), (4, 4), (6, 9), (7, 10)]


def eval_env(env_name, task_id, split, log_dir, num_trials, resize_size, replan_steps, host, port, seed, rollout_root=None, eval_fixed_layouts=False, episodes_per_layout=5, episode_start=0):
    # set args based on task
    assert split in ["pretrain", "target"]
    horizon = get_task_horizon(env_name)

    if rollout_root is not None:
        # Flat SAFE-compatible layout: all tasks share one env_records/ dir.
        env_records_dir = pathlib.Path(rollout_root) / "env_records"
        env_records_dir.mkdir(parents=True, exist_ok=True)
        # Check if this task already has episodes recorded (resume-safe).
        existing = list(env_records_dir.glob(f"task{task_id}--ep*--succ*.pkl"))
        if len(existing) >= num_trials - episode_start:
            print(f"[skip] task_id={task_id} ({env_name}) already has {len(existing)} episodes.")
            return
        log_path = str(pathlib.Path(rollout_root) / "logs" / f"{env_name}")
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    else:
        assert log_dir is not None, "Provide --log-dir or --rollout-root"
        now = datetime.now()
        now_formatted = now.strftime("%Y-%m-%d-%H-%M")
        log_path = f"{log_dir}/evals_1.5/{split}/{env_name}/{now_formatted}"

        for root, dirs, files in os.walk(os.path.dirname(log_path)):
            if "stats.json" in files:
                print(f"{env_name}/{split}, stats path exists, skipping.")
                return

        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
        # SAFE-compatible subdirectories
        env_records_dir = pathlib.Path(log_path) / "env_records"
        env_records_dir.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    # Get task
    if eval_fixed_layouts:
        # Original-paper eval: 5 fixed (layout, style) combos, cycled per episode.
        env = gym.make(
            f"robocasa/{env_name}",
            split=None,
            layout_and_style_ids=ORIG_EVAL_LAYOUTS,
            obj_instance_split="target",
            seed=seed,
        )
    else:
        env = gym.make(f"robocasa/{env_name}", split=split, seed=seed)
    task_description = env_name  # use env name as task description

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(episode_start, num_trials)):

        # Reset environment — for fixed-layout eval, pin the layout before reset.
        if eval_fixed_layouts:
            layout = ORIG_EVAL_LAYOUTS[episode_idx // episodes_per_layout]
            env.env.layout_and_style_ids = [layout]
        obs, info = env.reset()
        task_lang = obs["annotation.human.task_description"]
        task_description = task_lang  # use actual language annotation
        action_plan = collections.deque()

        # Setup
        t = 0
        replay_images = []
        model_infer_times = 0  # count how many times the policy was queried
        infer_timestep = 0     # timestep index for policy record naming

        logging.info(f"Starting episode {task_episodes+1}...")
        while t < horizon:
            # Get preprocessed image
            img = np.ascontiguousarray(obs["video.robot0_agentview_left"])
            wrist_img = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])
            img_right = np.ascontiguousarray(obs["video.robot0_agentview_right"])

            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, resize_size, resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
            )
            img_right = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_right, resize_size, resize_size)
            )

            if not action_plan:
                state = np.concatenate(
                    (
                        obs["state.end_effector_position_relative"],
                        obs["state.end_effector_rotation_relative"],
                        obs["state.base_position"],
                        obs["state.base_rotation"],
                        obs["state.gripper_qpos"],
                    ),
                    axis=0,
                )

                # Prepare observations dict — include run metadata for SAFE policy record naming
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/right_image": img_right,
                    "observation/state": state,
                    "prompt": task_lang,
                    "run/task_id": task_id,
                    "run/episode_idx": episode_idx,
                    "run/timestep": infer_timestep,
                }

                # Query model to get action
                action_chunk = client.infer(element)["actions"]
                assert (
                    len(action_chunk) >= replan_steps
                ), f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[: replan_steps])
                model_infer_times += 1
                infer_timestep += 1

            action = action_plan.popleft()
            action = convert_action(action)

            # Execute action in environment
            obs, reward, done, truncated, info = env.step(action)
            done = info["success"]  # for robocasa, use success entry in info

            replay_img = env.render()
            replay_img = np.ascontiguousarray(replay_img)
            replay_img = image_tools.convert_to_uint8(replay_img)

            if t % 2 == 0 or t == horizon - 1 or done:
                replay_images.append(replay_img)

            if done:
                task_successes += 1
                total_successes += 1
                break
            t += 1

        task_episodes += 1
        total_episodes += 1

        # Save mp4 in SAFE-compatible naming under env_records/
        succ_int = 1 if done else 0
        mp4_name = f"task{task_id}--ep{episode_idx}--succ{succ_int}.mp4"
        mp4_path = env_records_dir / mp4_name
        imageio.mimwrite(mp4_path, [np.asarray(x) for x in replay_images], fps=20)

        # Save env_record pkl in SAFE-compatible format
        env_record = {
            "task_suite_name": "robocasa365",
            "task_id": task_id,
            "task_description": task_description,
            "episode_idx": episode_idx,
            "episode_success": bool(done),
            "mp4_path": str(mp4_path),
            "model_infer_times": model_infer_times,
            "replan_steps": replan_steps,
            "end_step": t,
        }
        pkl_name = f"task{task_id}--ep{episode_idx}--succ{succ_int}.pkl"
        with open(env_records_dir / pkl_name, "wb") as f:
            pickle.dump(env_record, f)

        # Log current results
        logging.info(f"Success: {done}")
        logging.info(f"# episodes completed so far: {total_episodes}")
        logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"[{env_name}] Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"[{env_name}] Total episodes: {total_episodes}")
    print()
    stats = {
        "env_name": env_name,
        "task_id": task_id,
        "num_episodes": total_episodes,
        "success_rate": float(total_successes) / float(total_episodes),
    }
    with open(os.path.join(log_path, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # close and delete the env
    env.env.close()
    del env.env
    del env


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_main)
