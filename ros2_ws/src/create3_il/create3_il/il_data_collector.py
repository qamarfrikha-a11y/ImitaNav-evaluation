#!/usr/bin/env python3
"""
Noeud de collecte de demonstrations pour Behavioral Cloning / DAgger.
Adapte de la logique d'observation de turtlebot3_drlnav (drl_environment.py),
simplifie pour un usage BC/DAgger sans rosbag: enregistrement direct en .npy.
"""

import math
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

NUM_SCAN_SAMPLES = 36
LIDAR_DISTANCE_CAP = 10.0
MAX_LINEAR_SPEED = 0.5    # ajuste selon la vitesse max que tu utilises au teleop
MAX_ANGULAR_SPEED = 2.0   # idem pour la rotation
SAVE_DIR = os.path.expanduser('~/imitanav_evaluation/data/raw')


def euler_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class IlDataCollector(Node):
    def __init__(self):
        super().__init__('il_data_collector')
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
        self.goal_x = 6.5
        self.goal_y = -2.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_heading = 0.0
        self.goal_distance = 0.0
        self.goal_angle = 0.0
        self.scan_ranges = [1.0] * NUM_SCAN_SAMPLES

        self.last_linear = 0.0
        self.last_angular = 0.0
        self.prev_linear = 0.0
        self.prev_angular = 0.0

        self.has_started_moving = False  # pour ignorer seulement l'attente initiale, pas les arrets volontaires

        self.observations = []
        self.actions = []
        qos = QoSProfile(depth=10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile=qos_profile_sensor_data)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, qos)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, qos)
        self.timer = self.create_timer(0.1, self.record_step)
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.get_logger().info(f"Collecte demarree. Les demonstrations seront sauvees dans {SAVE_DIR}")
        self.get_logger().info("Deplace le robot avec teleop_twist_keyboard. Ctrl+C pour arreter et sauvegarder.")

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_heading = euler_from_quaternion(msg.pose.pose.orientation)

        diff_x = self.goal_x - self.robot_x
        diff_y = self.goal_y - self.robot_y
        self.goal_distance = math.sqrt(diff_x ** 2 + diff_y ** 2)

        heading_to_goal = math.atan2(diff_y, diff_x)
        angle = heading_to_goal - self.robot_heading
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        self.goal_angle = angle

    def scan_callback(self, msg):
        if len(msg.ranges) != NUM_SCAN_SAMPLES:
            self.get_logger().warn(
                f"Nombre de rayons LiDAR inattendu: recu {len(msg.ranges)}, attendu {NUM_SCAN_SAMPLES}")
        n = min(len(msg.ranges), NUM_SCAN_SAMPLES)
        for i in range(n):
            r = msg.ranges[i]
            if math.isinf(r) or math.isnan(r):
                r = LIDAR_DISTANCE_CAP
            self.scan_ranges[i] = float(np.clip(r / LIDAR_DISTANCE_CAP, 0.0, 1.0))

    def cmd_vel_callback(self, msg):
        self.last_linear = msg.linear.x
        self.last_angular = msg.angular.z
        if abs(self.last_linear) > 1e-3 or abs(self.last_angular) > 1e-3:
            self.has_started_moving = True

    def goal_callback(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.get_logger().info(f"Nouvel objectif recu: x={self.goal_x:.2f} y={self.goal_y:.2f}")

    def build_observation(self):
        MAX_GOAL_DISTANCE = 12.0
        obs = list(self.scan_ranges)
        obs.append(float(np.clip(self.goal_distance / MAX_GOAL_DISTANCE, 0, 1)))
        obs.append(float(self.goal_angle) / math.pi)
        obs.append(float(np.clip(self.prev_linear / MAX_LINEAR_SPEED, -1, 1)))
        obs.append(float(np.clip(self.prev_angular / MAX_ANGULAR_SPEED, -1, 1)))
        return np.array(obs, dtype=np.float32)

    def record_step(self):
        # On ignore seulement la phase avant le tout premier mouvement (evite d'enregistrer
        # une longue attente initiale a vitesse nulle), mais on enregistre bien les arrets
        # volontaires une fois la demonstration commencee (important pour apprendre a s'arreter).
        if not self.has_started_moving:
            return
        obs = self.build_observation()
        action = np.array([self.last_linear, self.last_angular], dtype=np.float32)
        self.observations.append(obs)
        self.actions.append(action)
        self.prev_linear = self.last_linear
        self.prev_angular = self.last_angular
        if len(self.observations) % 50 == 0:
            self.get_logger().info(f"{len(self.observations)} pas enregistres...")

    def save_and_exit(self):
        if len(self.observations) == 0:
            self.get_logger().warn("Aucune donnee enregistree, rien a sauvegarder.")
            return
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        obs_path = os.path.join(SAVE_DIR, f'obs_{timestamp}.npy')
        act_path = os.path.join(SAVE_DIR, f'act_{timestamp}.npy')
        np.save(obs_path, np.array(self.observations))
        np.save(act_path, np.array(self.actions))
        self.get_logger().info(
            f"Sauvegarde: {len(self.observations)} pas -> {obs_path} / {act_path}")


def main(args=None):
    rclpy.init(args=args)
    node = IlDataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_and_exit()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
