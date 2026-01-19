# ruff: noqa
import dataclasses
import time
from typing import Dict, List, Optional

import actionlib
import numpy as np
import rospy
from cv_bridge import CvBridge
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal, GripperState
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray


@dataclasses.dataclass(frozen=True)
class GripperDefaults:
    max_width: float = 0.08
    grasp_width: float = 0.03
    force_min: float = 10.0
    force_max: float = 60.0
    grasp_speed: float = 0.05
    open_speed: float = 0.10
    epsilon_inner: float = 0.005
    epsilon_outer: float = 0.005


class _ImageCache:
    def __init__(self, topic: str, bridge: CvBridge) -> None:
        self._topic = topic
        self._bridge = bridge
        self._image = None
        self._stamp = 0.0
        rospy.Subscriber(self._topic, Image, self._cb, queue_size=1)

    def _cb(self, msg: Image) -> None:
        self._image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._stamp = msg.header.stamp.to_sec()

    def get(self, timeout_s: float = 1.0):
        start = time.time()
        while self._image is None:
            if time.time() - start > timeout_s:
                raise RuntimeError(f"Timed out waiting for image on {self._topic}")
            time.sleep(0.001)
        return self._image


class _JointStateCache:
    def __init__(self, topic: str) -> None:
        self._topic = topic
        self._msg = None
        rospy.Subscriber(self._topic, JointState, self._cb, queue_size=1)

    def _cb(self, msg: JointState) -> None:
        self._msg = msg

    def get_positions(self, joint_names: List[str], timeout_s: float = 1.0) -> np.ndarray:
        start = time.time()
        while self._msg is None:
            if time.time() - start > timeout_s:
                raise RuntimeError(f"Timed out waiting for joint_states on {self._topic}")
            time.sleep(0.001)

        name_to_pos = dict(zip(self._msg.name, self._msg.position, strict=False))
        return np.array([name_to_pos[n] for n in joint_names], dtype=np.float32)


class _GripperStateCache:
    def __init__(self, topic: str) -> None:
        self._topic = topic
        self._msg = None
        rospy.Subscriber(self._topic, GripperState, self._cb, queue_size=1)

    def _cb(self, msg: GripperState) -> None:
        self._msg = msg

    def get_width(self, timeout_s: float = 1.0) -> Optional[float]:
        start = time.time()
        while self._msg is None:
            if time.time() - start > timeout_s:
                return None
            time.sleep(0.001)
        return float(self._msg.width)


class FrankaDualRealEnv:
    def __init__(
        self,
        arm_ids: List[str],
        camera_topics: Dict[str, str],
        *,
        joint_state_topics: Optional[Dict[str, str]] = None,
        velocity_cmd_topics: Optional[Dict[str, str]] = None,
        gripper_state_topics: Optional[Dict[str, str]] = None,
        gripper_move_actions: Optional[Dict[str, str]] = None,
        gripper_grasp_actions: Optional[Dict[str, str]] = None,
        gripper_defaults: GripperDefaults = GripperDefaults(),
        max_joint_speed: float = 0.7,
        init_node: bool = True,
    ) -> None:
        if init_node:
            rospy.init_node("franka_dual_env", anonymous=True)

        self._arm_ids = arm_ids
        self._defaults = gripper_defaults
        self._max_joint_speed = max_joint_speed
        self._bridge = CvBridge()

        self._images = {name: _ImageCache(topic, self._bridge) for name, topic in camera_topics.items()}

        self._joint_states = {}
        for arm_id in arm_ids:
            topic = joint_state_topics.get(arm_id) if joint_state_topics else f"/{arm_id}/joint_states"
            self._joint_states[arm_id] = _JointStateCache(topic)

        self._gripper_states = {}
        for arm_id in arm_ids:
            topic = gripper_state_topics.get(arm_id) if gripper_state_topics else f"/{arm_id}/franka_gripper/gripper_state"
            self._gripper_states[arm_id] = _GripperStateCache(topic)

        self._vel_pubs = {}
        for arm_id in arm_ids:
            topic = velocity_cmd_topics.get(arm_id) if velocity_cmd_topics else f"/{arm_id}/joint_velocity_controller/command"
            self._vel_pubs[arm_id] = rospy.Publisher(topic, Float64MultiArray, queue_size=1)

        self._grasp_clients = {}
        self._move_clients = {}
        for arm_id in arm_ids:
            grasp = gripper_grasp_actions.get(arm_id) if gripper_grasp_actions else f"/{arm_id}/franka_gripper/grasp"
            move = gripper_move_actions.get(arm_id) if gripper_move_actions else f"/{arm_id}/franka_gripper/move"
            self._grasp_clients[arm_id] = actionlib.SimpleActionClient(grasp, GraspAction)
            self._move_clients[arm_id] = actionlib.SimpleActionClient(move, MoveAction)

        self._last_gripper_state = {arm_id: {"mode": None, "g": 0.0, "t": 0.0} for arm_id in arm_ids}

    def _joint_names(self, arm_id: str) -> List[str]:
        return [f"{arm_id}_joint{i}" for i in range(1, 8)]

    def _normalize_gripper(self, width: Optional[float]) -> float:
        if width is None:
            return 0.0
        g = 1.0 - (width / self._defaults.max_width)
        return float(np.clip(g, 0.0, 1.0))

    def get_observation(self) -> Dict[str, np.ndarray]:
        images = {name: cache.get() for name, cache in self._images.items()}

        left_id, right_id = self._arm_ids
        left_q = self._joint_states[left_id].get_positions(self._joint_names(left_id))
        right_q = self._joint_states[right_id].get_positions(self._joint_names(right_id))

        left_g = self._normalize_gripper(self._gripper_states[left_id].get_width())
        right_g = self._normalize_gripper(self._gripper_states[right_id].get_width())

        qpos = np.concatenate([left_q, [left_g], right_q, [right_g]]).astype(np.float32)
        return {"qpos": qpos, "images": images}

    def apply_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        if action.shape[-1] != 16:
            raise ValueError(f"Expected 16-dim action, got {action.shape}")

        left = action[:8]
        right = action[8:]
        self._send_joint_velocity(self._arm_ids[0], left[:7])
        self._send_joint_velocity(self._arm_ids[1], right[:7])
        self._send_gripper(self._arm_ids[0], float(left[-1]))
        self._send_gripper(self._arm_ids[1], float(right[-1]))

    def _send_joint_velocity(self, arm_id: str, qdot: np.ndarray) -> None:
        qdot = np.clip(qdot, -self._max_joint_speed, self._max_joint_speed)
        msg = Float64MultiArray(data=qdot.tolist())
        self._vel_pubs[arm_id].publish(msg)

    def _send_gripper(self, arm_id: str, g: float) -> None:
        now = time.time()
        state = self._last_gripper_state[arm_id]
        g = float(np.clip(g, 0.0, 1.0))

        if g > 0.5:
            if state["mode"] != "grasp" or abs(g - state["g"]) > 0.1 or (now - state["t"]) > 0.5:
                goal = GraspGoal()
                goal.width = self._defaults.grasp_width
                goal.speed = self._defaults.grasp_speed
                goal.force = self._defaults.force_min + g * (self._defaults.force_max - self._defaults.force_min)
                goal.epsilon.inner = self._defaults.epsilon_inner
                goal.epsilon.outer = self._defaults.epsilon_outer
                self._grasp_clients[arm_id].send_goal(goal)
                state.update({"mode": "grasp", "g": g, "t": now})
        else:
            if state["mode"] != "open":
                goal = MoveGoal()
                goal.width = self._defaults.max_width
                goal.speed = self._defaults.open_speed
                self._move_clients[arm_id].send_goal(goal)
                state.update({"mode": "open", "g": g, "t": now})


def make_real_env(
    arm_ids: List[str],
    camera_topics: Dict[str, str],
    *,
    joint_state_topics: Optional[Dict[str, str]] = None,
    velocity_cmd_topics: Optional[Dict[str, str]] = None,
    gripper_state_topics: Optional[Dict[str, str]] = None,
    gripper_move_actions: Optional[Dict[str, str]] = None,
    gripper_grasp_actions: Optional[Dict[str, str]] = None,
    gripper_defaults: GripperDefaults = GripperDefaults(),
    max_joint_speed: float = 0.7,
    init_node: bool = True,
) -> FrankaDualRealEnv:
    return FrankaDualRealEnv(
        arm_ids=arm_ids,
        camera_topics=camera_topics,
        joint_state_topics=joint_state_topics,
        velocity_cmd_topics=velocity_cmd_topics,
        gripper_state_topics=gripper_state_topics,
        gripper_move_actions=gripper_move_actions,
        gripper_grasp_actions=gripper_grasp_actions,
        gripper_defaults=gripper_defaults,
        max_joint_speed=max_joint_speed,
        init_node=init_node,
    )
