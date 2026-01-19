from typing import Dict, List, Optional

import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.franka_dual import real_env as _real_env


class FrankaDualEnvironment(_environment.Environment):
    """Environment for dual-arm Franka robots with 3 cameras."""

    def __init__(
        self,
        *,
        arm_ids: List[str],
        camera_topics: Dict[str, str],
        render_height: int = 224,
        render_width: int = 224,
        joint_state_topics: Optional[Dict[str, str]] = None,
        velocity_cmd_topics: Optional[Dict[str, str]] = None,
        gripper_state_topics: Optional[Dict[str, str]] = None,
        gripper_move_actions: Optional[Dict[str, str]] = None,
        gripper_grasp_actions: Optional[Dict[str, str]] = None,
    ) -> None:
        self._env = _real_env.make_real_env(
            arm_ids=arm_ids,
            camera_topics=camera_topics,
            joint_state_topics=joint_state_topics,
            velocity_cmd_topics=velocity_cmd_topics,
            gripper_state_topics=gripper_state_topics,
            gripper_move_actions=gripper_move_actions,
            gripper_grasp_actions=gripper_grasp_actions,
            init_node=True,
        )
        self._render_height = render_height
        self._render_width = render_width
        self._ts = None

    @override
    def reset(self) -> None:
        self._ts = self._env.get_observation()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts
        images = {}
        for cam_name, img in obs["images"].items():
            rgb = img[..., ::-1]
            rgb = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(rgb, self._render_height, self._render_width)
            )
            images[cam_name] = einops.rearrange(rgb, "h w c -> c h w")

        return {
            "state": obs["qpos"],
            "images": images,
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._env.apply_action(action["actions"])
        self._ts = self._env.get_observation()
