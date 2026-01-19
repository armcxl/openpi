import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_franka_dual_example() -> dict:
    """Creates a random input example for the dual Franka policy."""
    return {
        "state": np.ones((16,)),
        "images": {
            "cam_external": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaDualInputs(transforms.DataTransformFn):
    """Inputs for a dual-arm Franka policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]
    - state: [16] -> [left(7), left_gripper(1), right(7), right_gripper(1)]
    - actions: [action_horizon, 16]
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_external",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        if set(in_images) != set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        external = _parse_image(in_images["cam_external"])
        left_wrist = _parse_image(in_images["cam_left_wrist"])
        right_wrist = _parse_image(in_images["cam_right_wrist"])

        inputs = {
            "state": np.asarray(data["state"]),
            "image": {
                "base_0_rgb": external,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaDualOutputs(transforms.DataTransformFn):
    """Outputs for a dual-arm Franka policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :16])}
