#!/usr/bin/env python3
"""
Noeud d'evaluation de la CAMPAGNE : execute UN essai autonome et ajoute une
ligne au CSV maitre. Copie corrigee de eval_trial_node.py (l'original reste
intact).

Differences avec l'original :
  - collision detectee EN DIRECT : l'essai s'arrete au N-ieme contact
    distinct (EVAL_MAX_BUMPS, defaut 3 ; 1 = regle stricte) ;
  - temps de navigation et timeout en TEMPS DE SIMULATION (horodatage de la
    position reelle), le temps reel est enregistre a part ;
  - une seule issue par essai : success / collision / timeout ;
  - CSV complet (goal, methode, modele, seeds, pose de depart, ...) ;
  - modele obligatoire (BC_MODEL_PATH), goal fixe (pas de /goal_pose) ;
  - interrupteurs : EVAL_USE_SAFETY, EVAL_USE_FINAL_APPROACH (1 par defaut).

Usage : python3 eval_campaign_node.py <trial_uid>
Toute la configuration passe par des variables d'environnement (voir ENV).
"""

import csv
import datetime
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

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from irobot_create_msgs.msg import HazardDetectionVector


def env(name, default=None, required=False):
    v = os.environ.get(name, default)
    if required and (v is None or v == ''):
        print(f"ERREUR : variable d'environnement {name} manquante")
        sys.exit(2)
    return v


NUM_SCAN_SAMPLES = 36
LIDAR_DISTANCE_CAP = 10.0
MAX_LINEAR_SPEED = 0.5
MAX_ANGULAR_SPEED = 2.0
MAX_GOAL_DISTANCE = 12.0

MODEL_PATH = os.path.expanduser(env('BC_MODEL_PATH', required=True))
RESULTS_CSV = os.path.expanduser(env('EVAL_RESULTS_CSV', required=True))
GOAL_X = float(env('EVAL_GOAL_X', required=True))
GOAL_Y = float(env('EVAL_GOAL_Y', required=True))

# Pose de depart (monde). /odom est RELATIF au point d'apparition du robot :
# le but doit donc etre exprime dans le repere odometrique pour l'observation.
START_X = float(env('EVAL_START_X', 0.0) or 0.0)
START_Y = float(env('EVAL_START_Y', 0.0) or 0.0)
START_YAW = float(env('EVAL_START_YAW', 0.0) or 0.0)
_dx, _dy = GOAL_X - START_X, GOAL_Y - START_Y
if env('EVAL_ODOM_AT_START', '1') == '1':
    GOAL_ODOM_X = math.cos(START_YAW) * _dx + math.sin(START_YAW) * _dy
    GOAL_ODOM_Y = -math.sin(START_YAW) * _dx + math.cos(START_YAW) * _dy
else:
    GOAL_ODOM_X, GOAL_ODOM_Y = GOAL_X, GOAL_Y

GOAL_REACHED_THRESHOLD = 0.5
MAX_TRIAL_DURATION = float(env('EVAL_TIMEOUT_S', 160.0))

# Detection de PANNES du systeme (essai invalide, pas un echec de la methode)
NO_DATA_ABORT_S = 60.0        # pas d'odom / ground truth -> code 4
NO_MOTION_CHECK_S = 20.0      # commande envoyee mais robot immobile -> code 3
NO_MOTION_MIN_PATH_M = 0.05
NO_MOTION_MIN_CMD_STEPS = 50
SIM_STALL_WALL_S = 300.0      # temps reel ecoule sans temps simule -> code 5

USE_SAFETY = env('EVAL_USE_SAFETY', '1') == '1'
USE_FINAL_APPROACH = env('EVAL_USE_FINAL_APPROACH', '1') == '1'
# Collision = N-ieme contact DISTINCT (pare-chocs) hors zone de depart.
# N=1 : regle stricte (1er contact) ; N=3 (defaut) : contacts isoles toleres.
# Les messages repetes d'un meme choc (< BUMP_DEBOUNCE_S) comptent pour un seul.
MAX_BUMPS = int(env('EVAL_MAX_BUMPS', 3))
BUMP_DEBOUNCE_S = 1.0
# Le robot demarre a ~16 cm de sa station de recharge : les contacts survenant
# a moins de START_GRACE_M du point de depart sont ignores (comptes a part).
START_GRACE_M = float(env('EVAL_START_GRACE_M', 0.5))

SAFETY_DISTANCE = 0.30
FRONT_CONE_HALF_WIDTH = 3
STUCK_CYCLES_BEFORE_REVERSE = 15
REVERSE_LINEAR_SPEED = -0.15
ESCAPE_ANGULAR_SPEED = 1.0
ESCAPE_LOCK_CYCLES = 12


FINAL_APPROACH_DISTANCE = 1.5
FINAL_APPROACH_LINEAR = 0.25
FINAL_APPROACH_ANGULAR_GAIN = 1.5

FIELDS = [
    'trial_uid', 'experiment', 'goal', 'method', 'model_id', 'pct_correction',
    'train_seed', 'trial_seed', 'start_x', 'start_y', 'start_yaw',
    'start_gt_x', 'start_gt_y', 'start_gt_yaw', 'start_odom_x', 'start_odom_y',
    'outcome', 'success', 'collision', 'timeout',
    'nav_time_s', 'wall_time_s', 'path_length_m', 'final_dist_m',
    'n_hazard_events', 'n_hazard_ignored', 'n_bump_events', 'first_bump_s',
    'leave_start_s', 'scan_msgs',
    'hazard_types', 'safety_triggers', 'angular_std',
    'use_safety_filter', 'use_final_approach', 'max_bumps',
    'git_commit', 'timestamp',
]


def euler_yaw(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class BCPolicy(nn.Module):
    def __init__(self, input_dim=40, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class EvalCampaignNode(Node):
    def __init__(self, trial_uid):
        super().__init__('eval_campaign_node')
        self.trial_uid = trial_uid

        self.robot_x = self.robot_y = self.robot_heading = 0.0
        self.goal_distance = 0.0
        self.goal_angle = 0.0
        self.scan_ranges = [1.0] * NUM_SCAN_SAMPLES
        self.prev_linear = self.prev_angular = 0.0
        self.odom_received = False

        self.gt_x = self.gt_y = self.gt_yaw = 0.0
        self.start_snapshot = None
        self.gt_distance = 0.0
        self.gt_received = False
        self.last_gt_x = self.last_gt_y = None
        self.trajectory_length = 0.0
        self.sim_t = None
        self.sim_t0 = None

        self.hazard_active = False
        self.hazard_timestamps = []
        self.hazard_types = set()
        self.angular_history = []

        self.stuck_cycles = 0
        self.safety_triggers = 0
        self.escape_direction = 0.0
        self.escape_lock_remaining = 0

        self.model = BCPolicy()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        self.model.eval()

        self.start_time = None
        self.finished = False
        self.result = None
        self.hazard_ignored = 0
        self.bump_events = 0
        self.first_bump_s = None
        self.last_hazard_t = None
        self.abort_code = None
        self.t_init = time.time()
        self.moving_cmd_steps = 0
        self.scan_count = 0
        self.leave_start_s = None

        qos = QoSProfile(depth=10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        self.create_subscription(
            Odometry, '/sim_ground_truth_pose', self.ground_truth_callback, qos)
        self.create_subscription(
            LaserScan, '/scan', self.scan_callback,
            qos_profile=qos_profile_sensor_data)
        self.create_subscription(
            HazardDetectionVector, '/hazard_detection', self.hazard_callback, qos)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)

        self.timer = self.create_timer(0.1, self.control_step)
        self.get_logger().info(
            f"Essai {trial_uid} | modele={MODEL_PATH} | goal=({GOAL_X},{GOAL_Y}) "
            f"| safety={USE_SAFETY} final_approach={USE_FINAL_APPROACH} "
            f"max_bumps={MAX_BUMPS}")

    # ---------- callbacks ----------
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_heading = euler_yaw(msg.pose.pose.orientation)
        dx, dy = GOAL_ODOM_X - self.robot_x, GOAL_ODOM_Y - self.robot_y
        self.goal_distance = math.hypot(dx, dy)
        angle = math.atan2(dy, dx) - self.robot_heading
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        self.goal_angle = angle
        self.odom_received = True

    def ground_truth_callback(self, msg):
        self.gt_x = msg.pose.pose.position.x
        self.gt_y = msg.pose.pose.position.y
        self.gt_yaw = euler_yaw(msg.pose.pose.orientation)
        self.gt_distance = math.hypot(GOAL_X - self.gt_x, GOAL_Y - self.gt_y)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.sim_t = t if t > 0 else None
        if self.start_time is not None and self.last_gt_x is not None:
            self.trajectory_length += math.hypot(
                self.gt_x - self.last_gt_x, self.gt_y - self.last_gt_y)
        self.last_gt_x, self.last_gt_y = self.gt_x, self.gt_y
        self.gt_received = True

    def scan_callback(self, msg):
        self.scan_count += 1
        n = min(len(msg.ranges), NUM_SCAN_SAMPLES)
        for i in range(n):
            r = msg.ranges[i]
            if math.isinf(r) or math.isnan(r):
                r = LIDAR_DISTANCE_CAP
            self.scan_ranges[i] = float(np.clip(r / LIDAR_DISTANCE_CAP, 0.0, 1.0))

    def hazard_callback(self, msg):
        if self.start_time is None:      # ignore les contacts avant le depart
            return
        if len(msg.detections) > 0:
            if self.start_snapshot is not None and math.hypot(
                    self.gt_x - self.start_snapshot[0],
                    self.gt_y - self.start_snapshot[1]) < START_GRACE_M:
                self.hazard_ignored += 1      # zone de depart (station)
                self.hazard_active = False
                return
            now = time.time()
            if self.last_hazard_t is None or now - self.last_hazard_t > BUMP_DEBOUNCE_S:
                self.bump_events += 1
                if self.first_bump_s is None:
                    self.first_bump_s = self.elapsed_sim()
            self.last_hazard_t = now
            self.hazard_active = True
            self.hazard_timestamps.append(now)
            for d in msg.detections:
                self.hazard_types.add(str(getattr(d, 'type', '?')))
        else:
            self.hazard_active = False

    # ---------- politique ----------
    def build_observation(self):
        obs = list(self.scan_ranges)
        obs.append(float(np.clip(self.goal_distance / MAX_GOAL_DISTANCE, 0, 1)))
        obs.append(float(self.goal_angle) / math.pi)
        obs.append(float(np.clip(self.prev_linear / MAX_LINEAR_SPEED, -1, 1)))
        obs.append(float(np.clip(self.prev_angular / MAX_ANGULAR_SPEED, -1, 1)))
        return np.array(obs, dtype=np.float32)

    def predict_action(self, obs):
        with torch.no_grad():
            a = self.model(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()
        return (float(np.clip(a[0], -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)),
                float(np.clip(a[1], -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)))

    def final_approach_action(self):
        angular = float(np.clip(FINAL_APPROACH_ANGULAR_GAIN * self.goal_angle,
                                -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))
        linear = 0.0 if abs(self.goal_angle) > 0.5 else FINAL_APPROACH_LINEAR
        return linear, angular

    def safety_filter(self, linear, angular):
        n = len(self.scan_ranges)
        center = n // 2
        front = list(range(max(0, center - FRONT_CONE_HALF_WIDTH),
                           min(n, center + FRONT_CONE_HALF_WIDTH + 1)))
        front_min_m = min(self.scan_ranges[i] for i in front) * LIDAR_DISTANCE_CAP
        if front_min_m < SAFETY_DISTANCE and linear > 0:
            self.safety_triggers += 1
            self.stuck_cycles += 1
            if self.escape_lock_remaining <= 0:
                right = front[:len(front) // 2 + 1]
                left = front[len(front) // 2:]
                min_r = min(self.scan_ranges[i] for i in right)
                min_l = min(self.scan_ranges[i] for i in left)
                self.escape_direction = 1.0 if min_l > min_r else -1.0
                self.escape_lock_remaining = ESCAPE_LOCK_CYCLES
            angular = self.escape_direction * ESCAPE_ANGULAR_SPEED
            self.escape_lock_remaining -= 1
            linear = REVERSE_LINEAR_SPEED if self.stuck_cycles > STUCK_CYCLES_BEFORE_REVERSE else 0.0
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

    # ---------- collision / temps ----------
    def collision_now(self):
        return self.bump_events >= MAX_BUMPS

    def elapsed_sim(self):
        """Temps de simulation ecoule ; repli sur le temps reel si pas d'horodatage."""
        if self.sim_t is not None and self.sim_t0 is not None:
            return self.sim_t - self.sim_t0
        return time.time() - self.start_time

    # ---------- boucle ----------
    def abort(self, code, msg):
        self.get_logger().error(f"ESSAI INVALIDE (panne systeme, code {code}) : {msg}")
        self.publish_cmd(0.0, 0.0)
        self.abort_code = code
        self.finished = True

    def control_step(self):
        if self.finished:
            return
        if not self.odom_received or not self.gt_received or self.scan_count == 0:
            if time.time() - self.t_init > NO_DATA_ABORT_S:
                self.abort(4, f"aucune donnee odom/ground truth apres {NO_DATA_ABORT_S:.0f} s "
                              f"(odom={self.odom_received}, gt={self.gt_received})")
            return
        if self.start_time is None:
            self.start_time = time.time()
            self.sim_t0 = self.sim_t
            self.trajectory_length = 0.0
            self.start_snapshot = (self.gt_x, self.gt_y, self.gt_yaw,
                                   self.robot_x, self.robot_y)
            self.get_logger().info(
                f"Depart : gt=({self.gt_x:.2f},{self.gt_y:.2f},{self.gt_yaw:.2f}) "
                f"odom=({self.robot_x:.2f},{self.robot_y:.2f}) "
                f"demande=({START_X:.2f},{START_Y:.2f},{START_YAW:.2f})")

        elapsed = self.elapsed_sim()
        if (self.leave_start_s is None and self.start_snapshot is not None and
                math.hypot(self.gt_x - self.start_snapshot[0],
                           self.gt_y - self.start_snapshot[1]) >= START_GRACE_M):
            self.leave_start_s = elapsed

        if self.gt_distance < GOAL_REACHED_THRESHOLD:
            self.publish_cmd(0.0, 0.0)
            self.finish('success', elapsed)
            return
        if self.collision_now():
            self.publish_cmd(0.0, 0.0)
            self.finish('collision', elapsed)
            return
        if elapsed > MAX_TRIAL_DURATION:
            self.publish_cmd(0.0, 0.0)
            self.finish('timeout', elapsed)
            return
        if (elapsed > NO_MOTION_CHECK_S and self.trajectory_length < NO_MOTION_MIN_PATH_M
                and self.moving_cmd_steps >= NO_MOTION_MIN_CMD_STEPS):
            self.abort(3, f"commande envoyee ({self.moving_cmd_steps} pas) mais robot "
                          f"immobile apres {elapsed:.0f} s (trajet {self.trajectory_length:.3f} m)")
            return
        if time.time() - self.start_time > SIM_STALL_WALL_S and elapsed < 10.0:
            self.abort(5, "simulation figee (temps simule quasi nul)")
            return

        obs = self.build_observation()
        if USE_FINAL_APPROACH and self.gt_distance < FINAL_APPROACH_DISTANCE:
            linear, angular = self.final_approach_action()
        else:
            linear, angular = self.predict_action(obs)
        if USE_SAFETY:
            linear, angular = self.safety_filter(linear, angular)
        self.publish_cmd(linear, angular)
        if abs(linear) > 0.05:
            self.moving_cmd_steps += 1
        self.prev_linear, self.prev_angular = linear, angular
        self.angular_history.append(angular)

    def finish(self, outcome, elapsed):
        if self.finished:
            return
        self.finished = True
        std = float(np.std(self.angular_history)) if len(self.angular_history) > 1 else 0.0
        self.result = {
            'trial_uid': self.trial_uid,
            'experiment': env('EVAL_EXPERIMENT', ''),
            'goal': env('EVAL_GOAL_NAME', ''),
            'method': env('EVAL_METHOD', ''),
            'model_id': env('EVAL_MODEL_ID', os.path.basename(MODEL_PATH)),
            'pct_correction': env('EVAL_PCT_CORR', ''),
            'train_seed': env('EVAL_TRAIN_SEED', ''),
            'trial_seed': env('EVAL_TRIAL_SEED', ''),
            'start_x': env('EVAL_START_X', ''),
            'start_y': env('EVAL_START_Y', ''),
            'start_yaw': env('EVAL_START_YAW', ''),
            'start_gt_x': round(self.start_snapshot[0], 3),
            'start_gt_y': round(self.start_snapshot[1], 3),
            'start_gt_yaw': round(self.start_snapshot[2], 3),
            'start_odom_x': round(self.start_snapshot[3], 3),
            'start_odom_y': round(self.start_snapshot[4], 3),
            'outcome': outcome,
            'success': int(outcome == 'success'),
            'collision': int(outcome == 'collision'),
            'timeout': int(outcome == 'timeout'),
            'nav_time_s': round(elapsed, 2),
            'wall_time_s': round(time.time() - self.start_time, 2),
            'path_length_m': round(self.trajectory_length, 3),
            'final_dist_m': round(self.gt_distance, 3),
            'n_hazard_events': len(self.hazard_timestamps),
            'n_hazard_ignored': self.hazard_ignored,
            'n_bump_events': self.bump_events,
            'leave_start_s': '' if self.leave_start_s is None else round(self.leave_start_s, 2),
            'scan_msgs': self.scan_count,
            'first_bump_s': '' if self.first_bump_s is None else round(self.first_bump_s, 2),
            'hazard_types': '|'.join(sorted(self.hazard_types)),
            'safety_triggers': self.safety_triggers,
            'angular_std': round(std, 4),
            'use_safety_filter': int(USE_SAFETY),
            'use_final_approach': int(USE_FINAL_APPROACH),
            'max_bumps': MAX_BUMPS,
            'git_commit': env('EVAL_GIT_COMMIT', ''),
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        }
        self.get_logger().info(
            f"Essai {self.trial_uid} : {outcome.upper()} "
            f"(sim={elapsed:.1f}s, reel={self.result['wall_time_s']}s, "
            f"traj={self.trajectory_length:.2f}m, dist_finale={self.gt_distance:.2f}m)")


def write_result(result):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if not exists:
            w.writeheader()
        w.writerow(result)
        f.flush()
        os.fsync(f.fileno())


def main():
    if len(sys.argv) < 2:
        print("Usage: eval_campaign_node.py <trial_uid>")
        sys.exit(1)
    rclpy.init(args=[])
    node = EvalCampaignNode(sys.argv[1])
    code = 0
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node.result is not None:
            write_result(node.result)
        code = node.abort_code or 0
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()