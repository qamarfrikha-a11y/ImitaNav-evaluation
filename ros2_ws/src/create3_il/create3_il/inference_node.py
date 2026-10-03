#!/usr/bin/env python3
"""
Nœud d'inférence pour le modèle BC.
Charge le modèle entraîné et pilote le robot en temps réel.
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

NUM_SCAN_SAMPLES = 36
LIDAR_DISTANCE_CAP = 10.0
MAX_LINEAR_SPEED = 0.5
MAX_ANGULAR_SPEED = 2.0
MAX_GOAL_DISTANCE = 12.0

MODEL_PATH = os.path.expanduser('~/imitanav_evaluation/models/bc_model.pt')
INPUT_DIM = 40
OUTPUT_DIM = 2


class BCPolicy(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM):
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


def euler_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
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
        self.prev_linear = 0.0
        self.prev_angular = 0.0    
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
        # Charger le modèle
        self.device = torch.device('cpu')
        self.model = BCPolicy().to(self.device)
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.get_logger().info(f"Modèle chargé depuis {MODEL_PATH}")
        else:
            self.get_logger().error(f"Modèle non trouvé: {MODEL_PATH}")
            return
        self.model.eval()

        # QoS et subscriptions
        qos = QoSProfile(depth=10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile=qos_profile_sensor_data)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, qos)

        # Publisher pour les commandes
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', qos)

        # Timer pour la boucle de contrôle
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("Nœud d'inférence démarré. Publie un objectif sur /goal_pose pour lancer le robot.")

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
        n = min(len(msg.ranges), NUM_SCAN_SAMPLES)
        for i in range(n):
            r = msg.ranges[i]
            if math.isinf(r) or math.isnan(r):
                r = LIDAR_DISTANCE_CAP
            self.scan_ranges[i] = float(np.clip(r / LIDAR_DISTANCE_CAP, 0.0, 1.0))

    def goal_callback(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.get_logger().info(f"Nouvel objectif: x={self.goal_x:.2f} y={self.goal_y:.2f}")

    def build_observation(self):
        obs = list(self.scan_ranges)
        obs.append(float(np.clip(self.goal_distance / MAX_GOAL_DISTANCE, 0, 1)))
        obs.append(float(self.goal_angle) / math.pi)
        obs.append(float(np.clip(self.prev_linear / MAX_LINEAR_SPEED, -1, 1)))
        obs.append(float(np.clip(self.prev_angular / MAX_ANGULAR_SPEED, -1, 1)))
        return np.array(obs, dtype=np.float32)

    def control_loop(self):
        # Si le robot est très proche de l'objectif, on s'arrête
        if self.goal_distance < 0.3:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        # Construire l'observation et faire la prédiction
        obs = self.build_observation()
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.model(obs_tensor).squeeze().cpu().numpy()

        linear = float(np.clip(action[0], -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED))
        angular = float(np.clip(action[1], -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))

        # Mettre à jour l'action précédente
        self.prev_linear = linear
        self.prev_angular = angular

        # Publier la commande
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
