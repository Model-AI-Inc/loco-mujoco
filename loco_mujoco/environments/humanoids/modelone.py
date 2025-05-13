from typing import Tuple, List, Union
import mujoco
from mujoco import MjSpec
import numpy as np

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core. utils import info_property


class ModelOne(BaseRobotHumanoid):

    """
    Description
    ------------

    Mujoco environment of the ModelOne robot based on the provided XML.


    Default Observation Space
    -----------------
    ============ ========================== ================ ==================================== ============================== ===
    Index in Obs Name                       ObservationType  Min                                  Max                            Dim
    ============ ========================== ================ ==================================== ============================== ===
    0 - 4        q_root                     FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    5            q_left_hip_pitch           JointPos         [-1.39626]                           [1.39626]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    6            q_left_hip_roll            JointPos         [-0.87266]                           [0.08727]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    7            q_left_hip_yaw             JointPos         [-0.34907]                           [0.34907]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    8            q_left_knee_pitch          JointPos         [-1.57080]                           [0.87266]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    9            q_left_ankle_pitch         JointPos         [-0.34907]                           [0.34907]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    10           q_left_toe                 JointPos         [0.00000]                            [0.69813]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    11           q_right_hip_pitch          JointPos         [-1.39626]                           [1.39626]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    12           q_right_hip_roll           JointPos         [-0.87266]                           [0.08727]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    13           q_right_hip_yaw            JointPos         [-0.34907]                           [0.34907]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    14           q_right_knee_pitch         JointPos         [-1.57080]                           [0.87266]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    15           q_right_ankle_pitch        JointPos         [-0.34907]                           [0.34907]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    16           q_right_toe                JointPos         [0.00000]                            [0.69813]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    17           q_torso                    JointPos         [-0.69811]                           [0.69816]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    18           q_left_shoulder_roll       JointPos         [-0.87266]                           [0.87266]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    19           q_left_shoulder_yaw        JointPos         [0.00000]                            [2.44346]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    20           q_left_elbow_pitch         JointPos         [-1.57080]                           [0.00000]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    21           q_right_shoulder_roll      JointPos         [-0.87266]                           [0.87266]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    22           q_right_shoulder_yaw       JointPos         [-1.57080]                           [0.00000]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    23           q_right_elbow_pitch        JointPos         [-1.57080]                           [0.00000]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    24           q_head                     JointPos         [-1.39626]                           [1.39626]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    25 - 30      dq_root                    FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    31           dq_left_hip_pitch          JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    32           dq_left_hip_roll           JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    33           dq_left_hip_yaw            JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    34           dq_left_knee_pitch         JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    35           dq_left_ankle_pitch        JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    36           dq_left_toe                JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    37           dq_right_hip_pitch         JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    38           dq_right_hip_roll          JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    39           dq_right_hip_yaw           JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    40           dq_right_knee_pitch        JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    41           dq_right_ankle_pitch       JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    42           dq_right_toe               JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    43           dq_torso                   JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    44           dq_left_shoulder_roll      JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    45           dq_left_shoulder_yaw       JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    46           dq_left_elbow_pitch        JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    47           dq_right_shoulder_roll     JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    48           dq_right_shoulder_yaw      JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    49           dq_right_elbow_pitch       JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    50           dq_head                    JointVel         [-inf]                               [inf]                          1
    ============ ========================== ================ ==================================== ============================== ===

    Default Action Space
    ----------------

    Control function type: **DefaultControl**

    See control function interface for more details.

    =============== ==== ===
    Index in Action Min  Max
    =============== ==== ===
    0               -1.0 1.0
    --------------- ---- ---
    1               -1.0 1.0
    --------------- ---- ---
    2               -1.0 1.0
    --------------- ---- ---
    3               -1.0 1.0
    --------------- ---- ---
    4               -1.0 1.0
    --------------- ---- ---
    5               -1.0 1.0
    --------------- ---- ---
    6               -1.0 1.0
    --------------- ---- ---
    7               -1.0 1.0
    --------------- ---- ---
    8               -1.0 1.0
    --------------- ---- ---
    9               -1.0 1.0
    --------------- ---- ---
    10              -1.0 1.0
    --------------- ---- ---
    11              -1.0 1.0
    --------------- ---- ---
    12              -1.0 1.0
    --------------- ---- ---
    13              -1.0 1.0
    --------------- ---- ---
    14              -1.0 1.0
    --------------- ---- ---
    15              -1.0 1.0
    --------------- ---- ---
    16              -1.0 1.0
    --------------- ---- ---
    17              -1.0 1.0
    --------------- ---- ---
    18              -1.0 1.0
    --------------- ---- ---
    19              -1.0 1.0
    =============== ==== ===

    Methods
    ------------

    """

    mjx_enabled = False

    def __init__(self, disable_arms: bool = False,
                 disable_back_joint: bool = False,
                 spec: Union[str, MjSpec] = None,
                 observation_spec: List[Observation] = None,
                 actuation_spec: List[str] = None,
                 **kwargs) -> None:

        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint

        if spec is None:
            spec = self.get_default_xml_file_path()

        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        if observation_spec is None:
            observation_spec = self._get_observation_specification(spec)
        else:
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)
        if disable_arms or disable_back_joint:
            joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
            actuation_spec = [ac for ac in actuation_spec if ac not in motors_to_remove]
            spec = self._delete_from_spec(spec, joints_to_remove,
                                          motors_to_remove, equ_constr_to_remove)
            if disable_arms:
                spec = self._reorient_arms(spec)

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)

    def _get_xml_modifications(self) -> Tuple[List[str], List[str], List[str]]:
        joints_to_remove = ["left_ankle_pitch","right_ankle_pitch"]
        motors_to_remove = []
        equ_constr_to_remove = []

        if self._disable_arms:
            arm_joints = ["left_shoulder_roll", "left_shoulder_yaw", "left_elbow_pitch",
                          "right_shoulder_roll", "right_shoulder_yaw", "right_elbow_pitch"]
            joints_to_remove += arm_joints
            motors_to_remove += arm_joints

        if self._disable_back_joint:
            joints_to_remove += ["torso"]
            motors_to_remove += ["torso"]

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        observation_spec = [
                            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
                            ObservationType.JointPos("q_left_hip_pitch", xml_name="left_hip_pitch"),
                            ObservationType.JointPos("q_left_hip_roll", xml_name="left_hip_roll"),
                            ObservationType.JointPos("q_left_hip_yaw", xml_name="left_hip_yaw"),
                            ObservationType.JointPos("q_left_knee_pitch", xml_name="left_knee_pitch"),
                            ObservationType.JointPos("q_left_ankle_pitch", xml_name="left_ankle_pitch"),
                            ObservationType.JointPos("q_left_toe", xml_name="left_toe"),
                            ObservationType.JointPos("q_right_hip_pitch", xml_name="right_hip_pitch"),
                            ObservationType.JointPos("q_right_hip_roll", xml_name="right_hip_roll"),
                            ObservationType.JointPos("q_right_hip_yaw", xml_name="right_hip_yaw"),
                            ObservationType.JointPos("q_right_knee_pitch", xml_name="right_knee_pitch"),
                            ObservationType.JointPos("q_right_ankle_pitch", xml_name="right_ankle_pitch"),
                            ObservationType.JointPos("q_right_toe", xml_name="right_toe"),
                            ObservationType.JointPos("q_torso", xml_name="torso"),
                            ObservationType.JointPos("q_left_shoulder_roll", xml_name="left_shoulder_roll"),
                            ObservationType.JointPos("q_left_shoulder_yaw", xml_name="left_shoulder_yaw"),
                            ObservationType.JointPos("q_left_elbow_pitch", xml_name="left_elbow_pitch"),
                            ObservationType.JointPos("q_right_shoulder_roll", xml_name="right_shoulder_roll"),
                            ObservationType.JointPos("q_right_shoulder_yaw", xml_name="right_shoulder_yaw"),
                            ObservationType.JointPos("q_right_elbow_pitch", xml_name="right_elbow_pitch"),
                            ObservationType.JointPos("q_head", xml_name="head"),

                            ObservationType.FreeJointVel("dq_root", xml_name="root"),
                            ObservationType.JointVel("dq_left_hip_pitch", xml_name="left_hip_pitch"),
                            ObservationType.JointVel("dq_left_hip_roll", xml_name="left_hip_roll"),
                            ObservationType.JointVel("dq_left_hip_yaw", xml_name="left_hip_yaw"),
                            ObservationType.JointVel("dq_left_knee_pitch", xml_name="left_knee_pitch"),
                            ObservationType.JointVel("dq_left_ankle_pitch", xml_name="left_ankle_pitch"),
                            ObservationType.JointVel("dq_left_toe", xml_name="left_toe"),
                            ObservationType.JointVel("dq_right_hip_pitch", xml_name="right_hip_pitch"),
                            ObservationType.JointVel("dq_right_hip_roll", xml_name="right_hip_roll"),
                            ObservationType.JointVel("dq_right_hip_yaw", xml_name="right_hip_yaw"),
                            ObservationType.JointVel("dq_right_knee_pitch", xml_name="right_knee_pitch"),
                            ObservationType.JointVel("dq_right_ankle_pitch", xml_name="right_ankle_pitch"),
                            ObservationType.JointVel("dq_right_toe", xml_name="right_toe"),
                            ObservationType.JointVel("dq_torso", xml_name="torso"),
                            ObservationType.JointVel("dq_left_shoulder_roll", xml_name="left_shoulder_roll"),
                            ObservationType.JointVel("dq_left_shoulder_yaw", xml_name="left_shoulder_yaw"),
                            ObservationType.JointVel("dq_left_elbow_pitch", xml_name="left_elbow_pitch"),
                            ObservationType.JointVel("dq_right_shoulder_roll", xml_name="right_shoulder_roll"),
                            ObservationType.JointVel("dq_right_shoulder_yaw", xml_name="right_shoulder_yaw"),
                            ObservationType.JointVel("dq_right_elbow_pitch", xml_name="right_elbow_pitch"),
                            ObservationType.JointVel("dq_head", xml_name="head"),]
        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        return [
                #"left_toe", 
                #"left_ankle_pitch", 
                #"right_ankle_pitch", 
                "left_knee_pitch", "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
                "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee_pitch", 
                #"right_toe",
                #"torso",
                "left_shoulder_roll", "left_shoulder_yaw", "left_elbow_pitch",
                "right_shoulder_roll", "right_shoulder_yaw", "right_elbow_pitch"
                #"head"
                ]

    @staticmethod
    def _reorient_arms(spec: MjSpec) -> MjSpec:
        # left_shoulder_link = spec.find_body("shoulder") # Corresponds to old left_shoulder_pitch_link
        # if left_shoulder_link:
        #     left_shoulder_link.quat = [1.0, 0.25, 0.1, 0.0]

        # right_elbow_link = spec.find_body("forearm_2") # Corresponds to old right_elbow_pitch_link
        # if right_elbow_link:
        #     right_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]

        # right_shoulder_link = spec.find_body("shoulder_2") # Corresponds to old right_shoulder_pitch_link
        # if right_shoulder_link:
        #     right_shoulder_link.quat = [1.0, -0.25, 0.1, 0.0]

        # left_elbow_link = spec.find_body("forearm") # Corresponds to old left_elbow_pitch_link
        # if left_elbow_link:
        #     left_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]
        return spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        return (loco_mujoco.PATH_TO_MODELS / "modelone" / "modelone.xml").as_posix()

    @info_property
    def upper_body_xml_name(self) -> str:
        return "bodyupper"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        return (0.0, 0.01)