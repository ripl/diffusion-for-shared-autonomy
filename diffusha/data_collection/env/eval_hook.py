#!/usr/bin/env python3
from __future__ import annotations
from typing import Optional
import cv2
import numpy as np
import wandb
from pfrl.experiments.evaluation_hooks import EvaluationHook


def get_frame(env, episode, step, obs=None, reward=None, reward_sum=None, action: Optional[np.ndarray] = None,
              scale: float | None = None, frame: None | np.ndarray = None, info=None):
    # frame: (h, w, c)
    fontScale = .3

    env_name = env.unwrapped.spec.id

    if frame is None:
        frame = np.ascontiguousarray(env.render(mode='rgb_array'), dtype=np.uint8)
    frame = cv2.putText(frame, f'EP: {episode} STEP: {step}', org=(0, 20), fontFace=3, fontScale=fontScale,
                        color=(0, 255, 0), thickness=1)
    if reward is not None:
        frame = cv2.putText(frame, f'R: {reward:.4f}', org=(0, 60), fontFace=3, fontScale=fontScale, color=(0, 255, 0),
                            thickness=1)
        frame = cv2.putText(frame, f'R-sum: {reward_sum:.4f}', org=(0, 100), fontFace=3, fontScale=fontScale,
                            color=(0, 255, 0), thickness=1)

    if 'Push' not in env_name:
        if action is not None:
            frame = cv2.putText(frame, f'act 0: {action[0]:.3f}', org=(0, 140), fontFace=3, fontScale=fontScale,
                                color=(0, 255, 0), thickness=2)
            frame = cv2.putText(frame, f'act 1: {action[1]:.3f}', org=(0, 180), fontFace=3, fontScale=fontScale,
                                color=(0, 255, 0), thickness=2)

        if obs is not None:
            frame = cv2.putText(frame, f'obs[0]: {obs[0]:.3f}', org=(0, 140), fontFace=3, fontScale=fontScale,
                                color=(0, 255, 0), thickness=2)
            frame = cv2.putText(frame, f'obs[1]: {obs[1]:.3f}', org=(0, 180), fontFace=3, fontScale=fontScale,
                                color=(0, 255, 0), thickness=2)
            frame = cv2.putText(frame, f'obs[-2], obs[-1]: {obs[-2]:.3f}, {obs[-1]:.3f}', org=(0, 220), fontFace=3,
                                fontScale=fontScale, color=(0, 255, 0), thickness=2)

        if info is not None:
            frame = cv2.putText(frame, f'game-over: {info.get("game_over_reason", "")}', org=(0, 260), fontFace=3,
                                fontScale=fontScale, color=(0, 255, 0), thickness=2)
            frame = cv2.putText(frame, f'timelimit: {info.get("TimeLimit.truncated", "")}', org=(0, 300), fontFace=3,
                                fontScale=fontScale, color=(0, 255, 0), thickness=2)

    else:
        # import pdb; pdb.set_trace()
        if info is not None:
            frame = cv2.putText(frame, f"state: {info['state']}", org=(0, 140), fontFace=3, fontScale=fontScale,
                                color=(0, 255, 0), thickness=1)

        with np.printoptions(precision=2, suppress=True):
            frame = cv2.putText(frame, f"block: {obs[:3]}", org=(0, 160), fontFace=3, fontScale=fontScale,
                                color=(0, 255, 0), thickness=1)
            frame = cv2.putText(frame, f"ee: {obs[3:5]}", org=(0, 180), fontFace=3, fontScale=fontScale,
                                color=(0, 255, 0), thickness=1)
            frame = cv2.putText(frame, f"action: {action}", org=(0, 200), fontFace=3, fontScale=fontScale,
                                color=(0, 255, 0), thickness=1)

    # if obs is not None:
    #     x, y, vx, vy, rx, ry, rvx, rvy, *_ = obs
    #     frame = cv2.putText(frame, f'(X, Y): ({x:.1f}, {y:.1f})  (VX, VY): ({vx:.1f}, {vy:.1f})', org=(0, 140), fontFace=3, fontScale=fontScale, color=(0, 255, 0), thickness=2)
    #     frame = cv2.putText(frame, f'(RX, RY): ({rx:.1f}, {ry:.1f})  (RVX, RVY): ({rvx:.1f}, {rvy:.1f})', org=(0, 180), fontFace=3, fontScale=fontScale, color=(0, 255, 0), thickness=2)

    if scale is not None:
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, dsize=(width, height))

    return frame


def draw_rectangle(frame, color):
    h, w = frame.shape[0:2]
    base = np.zeros_like(frame)
    cv2.rectangle(base, (0, 0), (w, h), color, 40)
    base[15:h-15,15:w-15,...] = frame[15:h-15,15:w-15,...]
    return base


def get_eval_frame(env, episode, step, scale: float | None = None, frame: None | np.ndarray = None, info=None):
    # frame: (h, w, c)
    from diffusha.data_collection.env import is_lunarlander, is_maze2d, is_blockpush
    # fontScale = 1.3
    is_ll = is_lunarlander(env)
    is_bp = is_blockpush(env)

    if frame is None:
        frame = np.ascontiguousarray(env.render(mode='rgb_array'), dtype=np.uint8)

    if is_ll:
        frame = cv2.putText(frame, f'EP: {episode} STEP: {step}', org=(20, 65), fontFace=3, fontScale=1.3,
                            color=(0, 255, 0), thickness=1)
    if is_bp:
        frame = cv2.putText(frame, f'EP: {episode} STEP: {step}', org=(20, 45), fontFace=3, fontScale=0.8,
                            color=(0, 0, 0), thickness=1)

    if is_ll:
        if info is not None:
            success = ('goal' in info.keys() and (info['goal'] == 'landed' or info['goal'] == 'target-reached'))
            if success:
                frame = draw_rectangle(frame, (0, 255, 0))
            else:
                frame = draw_rectangle(frame, (255, 0, 0))

    elif is_bp:
        if info is not None:
            correct = (info['finished'])
            if correct:
                frame = draw_rectangle(frame, (0, 255, 0))
            else:
                frame = draw_rectangle(frame, (255, 0, 0))

    if scale is not None:
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, dsize=(width, height))

    return frame


class WandBLogger(EvaluationHook):
    support_train_agent = True

    # support_train_agent_batch = False
    # support_train_agent_async = False
    def __call__(self, env, agent, evaluator, step, eval_stats, agent_stats, env_stats) -> None:
        """Visualize trajectories for 5 episodes """
        num_eval_episodes = 5
        frames = []
        with agent.eval_mode():
            for ep in range(num_eval_episodes):
                step = 0
                done = False
                obs = env.reset()
                rew_sum = 0
                # TODO: add episode step in the image
                frames.append(get_frame(env, ep, step, obs))
                while not done:
                    action = agent.act(obs)
                    obs, rew, done, info = env.step(action)
                    step += 1
                    rew_sum += rew
                    frames.append(get_frame(env, ep, step, obs, rew, rew_sum, action=action, info=info))

                # import pdb; pdb.set_trace()

        wandb.log({
            'eval/mean': eval_stats['mean'],
            'eval/median': eval_stats['median'],
            'eval/ep_length': eval_stats['length_mean'],
            # frames: (t, h, w, c) -> (t, c, h, w)
            'eval/video': wandb.Video(np.asarray(frames).transpose(0, 3, 1, 2), fps=30, format='mp4'),
            'step': step
        })

        # print('env_stats', env_stats)  # []
        # print('agent_stats', agent_stats)  # [('average_q1', 1.5966111), ('average_q2', 1.6748627), ('average_q_func1_loss', 0.1904826650226658), ('average_q_func2_loss', 0.30078676901757717), ('n_updates', 44), ('average_entropy', 1.3570675), ('temperature', 0.986836314201355)]
