#!/usr/bin/env python3
"""
Noeud d'evaluation quantitative : execute UN essai autonome (aucune correction
humaine), mesure les metriques demandees par le sujet de stage, puis ecrit une
ligne de resultat dans un CSV et se termine tout seul.

Version corrigee du filtre de securite : ajout d'hysteresis pour eviter les
deux bugs precedents (blocage immobile / rotation en boucle sans fin).
Une direction d'evitement est choisie UNE FOIS au declenchement, puis
maintenue pendant ESCAPE_LOCK_CYCLES cycles avant de re-evaluer, et le
cone avant est plus etroit (obstacles vraiment de face uniquement).
"""

import csv
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from irobot_create_msgs.msg import HazardDetectionVector

NUM_SCAN_SAMPLES = 36
LIDAR_DISTANCE_CAP = 10.0
MAX_LINEAR_SPEED = 0.5
MAX_ANGULAR_SPEED = 2.0
MAX_GOAL_DISTANCE = 12.0
MODEL_PATH = os.path.expanduser(os.environ.get('BC_MODEL_PATH', '~/imitanav_evaluation/models/bc_model.pt'))
RESULTS_CSV = os.environ.get('EVAL_RESULTS_CSV', os.path.expanduser('~/imitanav_evaluation/results/evaluation.csv'))

GOAL_X = float(os.environ.get('EVAL_GOAL_X', 6.5))
GOAL_Y = float(os.environ.get('EVAL_GOAL_Y', -2.0))
GOAL_REACHED_THRESHOLD = 0.5
MAX_TRIAL_DURATION = 160.0

SAFETY_DISTANCE = 0.30
FRONT_CONE_HALF_WIDTH = 3
STUCK_CYCLES_BEFORE_REVERSE = 15
REVERSE_LINEAR_SPEED = -0.15
ESCAPE_ANGULAR_SPEED = 1.0
ESCAPE_LOCK_CYCLES = 12

COLLISION_RECENCY_S = 3.0
COLLISION_REPEAT_WINDOW_S = 5.0
COLLISION_REPEAT_COUNT = 3

FINAL_APPROACH_DISTANCE = 1.5
FINAL_APPROACH_LINEAR = 0.25
FINAL_APPROACH_ANGULAR_GAIN = 1.5


def euler_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class BCPolicy(nn.Module):
    def __init__(self, input_dim=40, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class EvalTrialNode(Node):
    def __init__(self, trial_id):
        super().__init__('eval_trial_node')
        self.trial_id = trial_id

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_heading = 0.0
        self.goal_distance = 0.0
        self.goal_angle = 0.0
        self.scan_ranges = [1.0] * NUM_SCAN_SAMPLES
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.odom_received = False

        self.gt_x = 0.0
        self.gt_y = 0.0
        self.gt_distance = 0.0
        self.gt_received = False
        self.last_gt_x = None
        self.last_gt_y = None
        self.trajectory_length = 0.0

        self.hazard_active = False
        self.hazard_timestamps = []
        self.angular_history = []

        self.stuck_cycles = 0
        self.safety_triggers = 0
        self.escape_direction = 0.0
        self.escape_lock_remaining = 0

        self.device = torch.device('cpu')
        self.model = BCPolicy().to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

        self.start_time = None
        self.finished = False
        self.result = None

        qos = QoSProfile(depth=10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        self.gt_sub = self.create_subscription(
            Odometry, '/sim_ground_truth_pose', self.ground_truth_callback, qos)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile=qos_profile_sensor_data)
        self.hazard_sub = self.create_subscription(
            HazardDetectionVector, '/hazard_detection', self.hazard_callback, qos)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_pose_callback, qos)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)

        self.timer = self.create_timer(0.1, self.control_step)
        self.get_logger().info(f"Essai #{trial_id} demarre.")

    def goal_pose_callback(self, msg):
        global GOAL_X, GOAL_Y
        GOAL_X = msg.pose.position.x
        GOAL_Y = msg.pose.position.y
        self.get_logger().info(f"Nouveau goal recu: x={GOAL_X:.2f} y={GOAL_Y:.2f}")

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_heading = euler_from_quaternion(msg.pose.pose.orientation)

        diff_x = GOAL_X - self.robot_x
        diff_y = GOAL_Y - self.robot_y
        self.goal_distance = math.sqrt(diff_x ** 2 + diff_y ** 2)

        heading_to_goal = math.atan2(diff_y, diff_x)
        angle = heading_to_goal - self.robot_heading
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        self.goal_angle = angle
        self.odom_received = True

    def ground_truth_callback(self, msg):
        self.gt_x = msg.pose.pose.position.x
        self.gt_y = msg.pose.pose.position.y
        diff_x = GOAL_X - self.gt_x
        diff_y = GOAL_Y - self.gt_y
        self.gt_distance = math.sqrt(diff_x ** 2 + diff_y ** 2)

        if self.last_gt_x is not None:
            step = math.sqrt((self.gt_x - self.last_gt_x) ** 2 + (self.gt_y - self.last_gt_y) ** 2)
            self.trajectory_length += step
        self.last_gt_x = self.gt_x
        self.last_gt_y = self.gt_y
        self.gt_received = True

    def scan_callback(self, msg):
        n = min(len(msg.ranges), NUM_SCAN_SAMPLES)
        for i in range(n):
            r = msg.ranges[i]
            if math.isinf(r) or math.isnan(r):
                r = LIDAR_DISTANCE_CAP
            self.scan_ranges[i] = float(np.clip(r / LIDAR_DISTANCE_CAP, 0.0, 1.0))

    def hazard_callback(self, msg):
        if len(msg.detections) > 0:
            self.hazard_active = True
            self.hazard_timestamps.append(time.time())
        else:
            self.hazard_active = False

    def build_observation(self):
        obs = list(self.scan_ranges)
        obs.append(float(np.clip(self.goal_distance / MAX_GOAL_DISTANCE, 0, 1)))
        obs.append(float(self.goal_angle) / math.pi)
        obs.append(float(np.clip(self.prev_linear / MAX_LINEAR_SPEED, -1, 1)))
        obs.append(float(np.clip(self.prev_angular / MAX_ANGULAR_SPEED, -1, 1)))
        return np.array(obs, dtype=np.float32)

    def predict_action(self, obs):
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = self.model(x).squeeze(0).numpy()
        linear = float(np.clip(action[0], -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED))
        angular = float(np.clip(action[1], -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))
        return linear, angular

    def final_approach_action(self):
        angular = float(np.clip(FINAL_APPROACH_ANGULAR_GAIN * self.goal_angle,
                                 -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))
        if abs(self.goal_angle) > 0.5:
            linear = 0.0
        else:
            linear = FINAL_APPROACH_LINEAR
        return linear, angular

    def safety_filter(self, linear, angular):
        n = len(self.scan_ranges)
        center = n // 2
        front_indices = list(range(
            max(0, center - FRONT_CONE_HALF_WIDTH),
            min(n, center + FRONT_CONE_HALF_WIDTH + 1)
        ))
        front_min_normalized = min(self.scan_ranges[i] for i in front_indices)
        front_min_meters = front_min_normalized * LIDAR_DISTANCE_CAP

        obstacle_ahead = front_min_meters < SAFETY_DISTANCE

        if obstacle_ahead and linear > 0:
            self.safety_triggers += 1
            self.stuck_cycles += 1

            if self.escape_lock_remaining <= 0:
                right_indices = front_indices[:len(front_indices) // 2 + 1]
                left_indices = front_indices[len(front_indices) // 2:]
                min_right = min(self.scan_ranges[i] for i in right_indices)
                min_left = min(self.scan_ranges[i] for i in left_indices)
                self.escape_direction = 1.0 if min_left > min_right else -1.0
                self.escape_lock_remaining = ESCAPE_LOCK_CYCLES

            angular = self.escape_direction * ESCAPE_ANGULAR_SPEED
            self.escape_lock_remaining -= 1

            if self.stuck_cycles > STUCK_CYCLES_BEFORE_REVERSE:
                linear = REVERSE_LINEAR_SPEED
            else:
                linear = 0.0
        else:
            self.stuck_cycles = 0
            self.escape_lock_remaining = 0

        return linear, angular

    def publish_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        try:
            self.cmd_vel_pub.publish(msg)
        except Exception:
            pass

    def is_blocking_collision(self, now):
        if not self.hazard_timestamps:
            return False
        if self.hazard_active and (now - self.hazard_timestamps[-1]) < COLLISION_RECENCY_S:
            return True
        recent = [t for t in self.hazard_timestamps if (now - t) < COLLISION_REPEAT_WINDOW_S]
        if len(recent) >= COLLISION_REPEAT_COUNT:
            return True
        return False

    def control_step(self):
        if not self.odom_received or not self.gt_received:
            return

        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time

        if self.gt_distance < GOAL_REACHED_THRESHOLD:
            self.publish_cmd(0.0, 0.0)
            self.finish(reached_goal=True, timeout=False, elapsed=elapsed)
            return

        if elapsed > MAX_TRIAL_DURATION:
            self.publish_cmd(0.0, 0.0)
            self.finish(reached_goal=False, timeout=True, elapsed=elapsed)
            return

        obs = self.build_observation()
        if self.gt_distance < FINAL_APPROACH_DISTANCE:
            linear, angular = self.final_approach_action()
        else:
            linear, angular = self.predict_action(obs)
        linear, angular = self.safety_filter(linear, angular)
        self.publish_cmd(linear, angular)
        self.prev_linear = linear
        self.prev_angular = angular
        self.angular_history.append(angular)

    def finish(self, reached_goal, timeout, elapsed):
        if self.finished:
            return
        self.finished = True

        now = time.time()
        blocking_collision = self.is_blocking_collision(now)
        had_any_contact = len(self.hazard_timestamps) > 0

        smoothness = float(np.std(self.angular_history)) if len(self.angular_history) > 1 else 0.0
        self.result = {
            'trial_id': self.trial_id,
            'success': int(reached_goal and not blocking_collision),
            'blocking_collision': int(blocking_collision),
            'minor_contact': int(had_any_contact and not blocking_collision),
            'timeout': int(timeout),
            'time_s': round(elapsed, 2),
            'trajectory_length_m': round(self.trajectory_length, 3),
            'angular_std': round(smoothness, 4),
            'safety_triggers': self.safety_triggers,
        }
        if self.result['success']:
            status = "REUSSI"
        elif blocking_collision:
            status = "COLLISION_BLOQUANTE"
        elif timeout:
            status = "TIMEOUT"
        else:
            status = "ECHEC_AUTRE"
        self.get_logger().info(
            f"Essai #{self.trial_id} termine : {status} "
            f"(temps={elapsed:.1f}s, traj={self.trajectory_length:.2f}m, "
            f"contacts_mineurs={had_any_contact and not blocking_collision}, "
            f"safety_triggers={self.safety_triggers})")


def write_result(result):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def main(args=None):
    if len(sys.argv) < 2:
        print("Usage: eval_trial_node.py <trial_id>")
        sys.exit(1)
    trial_id = int(sys.argv[1])

    rclpy.init(args=[])
    node = EvalTrialNode(trial_id)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node.result is not None:
            write_result(node.result)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()