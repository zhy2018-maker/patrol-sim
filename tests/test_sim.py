"""仿真内核单元测试（对应方案 B6）。

运行：
    cd handoff/code
    python -m unittest discover -s tests -v
或：
    python tests/test_sim.py
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.sim import GridMap, Robot, RADIUS  # noqa: E402


def open_map(cols=20, rows=12, walls=()):
    grid = [[0] * cols for _ in range(rows)]
    for r, c in walls:
        grid[r][c] = 1
    return GridMap(grid)


class TestKinematics(unittest.TestCase):
    def test_forward_one_meter(self):
        robot = Robot(open_map(), 2.0, 2.0, 0.0)
        self.assertTrue(robot.apply({"type": "move", "direction": "forward", "distance": 1.0})["ok"])
        for _ in range(200):
            robot.advance(0.02)
        self.assertAlmostEqual(robot.x, 3.0, delta=0.02)
        self.assertAlmostEqual(robot.y, 2.0, delta=0.02)
        self.assertAlmostEqual(robot.theta, 0.0, delta=1e-6)
        self.assertEqual(robot.status, "idle")

    def test_backward_keeps_heading(self):
        robot = Robot(open_map(), 3.0, 2.0, 0.0)
        robot.apply({"type": "move", "direction": "backward", "distance": 0.5})
        for _ in range(100):
            robot.advance(0.02)
        self.assertAlmostEqual(robot.x, 2.5, delta=0.02)
        self.assertAlmostEqual(robot.theta, 0.0, delta=1e-6)

    def test_turn_left_ninety(self):
        robot = Robot(open_map(), 2.0, 2.0, 0.0)
        robot.apply({"type": "turn", "direction": "left", "angle": math.pi / 2})
        for _ in range(100):
            robot.advance(0.02)
        self.assertAlmostEqual(robot.theta, math.pi / 2, delta=math.radians(2))

    def test_step_size_does_not_matter(self):
        """一次补 0.2 s 与分五次补 0.04 s，结果应完全一致。"""
        a = Robot(open_map(), 2.0, 2.0, 0.4)
        b = Robot(open_map(), 2.0, 2.0, 0.4)
        for r in (a, b):
            r.apply({"type": "move", "direction": "forward", "distance": 1.0})
        a.advance(0.2)
        for _ in range(5):
            b.advance(0.04)
        self.assertAlmostEqual(a.x, b.x, delta=1e-9)
        self.assertAlmostEqual(a.y, b.y, delta=1e-9)


class TestCollision(unittest.TestCase):
    def test_out_of_bounds_target_rejected(self):
        robot = Robot(open_map(), 0.5, 2.0, math.pi)
        res = robot.apply({"type": "move", "direction": "forward", "distance": 1.0})
        self.assertFalse(res["ok"])
        self.assertEqual(res["code"], "OUT_OF_BOUNDS")
        self.assertAlmostEqual(robot.x, 0.5, delta=1e-6)
        self.assertEqual(robot.status, "error")

    def test_stops_before_wall(self):
        walls = [(r, 4) for r in range(12)]  # 一整列障碍：x ∈ [2.0, 2.5)
        robot = Robot(open_map(walls=walls), 3.2, 2.0, math.pi)
        robot.apply({"type": "move", "direction": "forward", "distance": 2.0})
        for _ in range(200):
            robot.advance(0.02)
        self.assertEqual(robot.status, "error")
        self.assertEqual(robot.error, "OBSTACLE_BLOCKED")
        # 墙面 x=2.5，半径 0.18 → 停在约 2.68
        self.assertGreater(robot.x, 2.5 + RADIUS - 0.03)
        self.assertLess(robot.x, 2.5 + RADIUS + 0.03)

    def test_robot_never_leaves_map(self):
        robot = Robot(open_map(), 1.0, 1.0, 0.0)
        for _ in range(60):
            for cmd in (
                {"type": "move", "direction": "forward", "distance": 3.0},
                {"type": "turn", "direction": "left", "angle": math.pi / 2},
            ):
                robot.apply(cmd)
                for _ in range(100):
                    robot.advance(0.02)
            self.assertGreaterEqual(robot.x, 0.0)
            self.assertGreaterEqual(robot.y, 0.0)
            self.assertLessEqual(robot.x, robot.map.width)
            self.assertLessEqual(robot.y, robot.map.height)


class TestCommandQueue(unittest.TestCase):
    def test_queue_runs_in_order(self):
        robot = Robot(open_map(), 2.0, 2.0, 0.0)
        robot.apply({"type": "move", "direction": "forward", "distance": 0.3})
        res = robot.apply({"type": "move", "direction": "forward", "distance": 0.3})
        self.assertTrue(res["ok"])
        self.assertEqual(robot.state()["queue"], 1)
        for _ in range(300):
            robot.advance(0.02)
        self.assertAlmostEqual(robot.x, 2.6, delta=0.02)
        self.assertEqual(robot.state()["queue"], 0)
        self.assertEqual(robot.status, "idle")

    def test_stop_clears_queue(self):
        robot = Robot(open_map(), 2.0, 2.0, 0.0)
        robot.apply({"type": "move", "direction": "forward", "distance": 1.0})
        robot.apply({"type": "move", "direction": "forward", "distance": 1.0})
        robot.apply({"type": "stop"})
        self.assertEqual(robot.state()["queue"], 0)
        frozen = robot.x
        robot.advance(1.0)
        self.assertAlmostEqual(robot.x, frozen, delta=1e-9)

    def test_queue_full_returns_busy(self):
        robot = Robot(open_map(), 2.0, 2.0, 0.0)
        robot.apply({"type": "move", "direction": "forward", "distance": 0.3})
        for _ in range(8):
            self.assertTrue(robot.apply({"type": "move", "direction": "forward", "distance": 0.3})["ok"])
        res = robot.apply({"type": "move", "direction": "forward", "distance": 0.3})
        self.assertFalse(res["ok"])
        self.assertEqual(res["code"], "BUSY")
        self.assertEqual(robot.state()["queue"], 8)


class TestStateAndSensors(unittest.TestCase):
    def test_status_lifecycle(self):
        robot = Robot(open_map(), 2.0, 2.0, 0.0)
        self.assertEqual(robot.status, "idle")
        robot.apply({"type": "move", "direction": "forward", "distance": 0.5})
        self.assertEqual(robot.status, "moving")
        for _ in range(100):
            robot.advance(0.02)
        self.assertEqual(robot.status, "idle")

    def test_stop_interrupts_motion(self):
        robot = Robot(open_map(), 2.0, 2.0, 0.0)
        robot.apply({"type": "move", "direction": "forward", "distance": 5.0})
        robot.advance(0.2)
        robot.apply({"type": "stop"})
        x_after_stop = robot.x
        robot.advance(1.0)
        self.assertAlmostEqual(robot.x, x_after_stop, delta=1e-9)

    def test_sensor_reads_distance(self):
        walls = [(r, 10) for r in range(12)]  # 墙在 x = 5.0
        robot = Robot(open_map(walls=walls), 4.0, 2.0, 0.0)
        front = robot.sensors()["front"]
        self.assertAlmostEqual(front, 1.0, delta=0.06)

        robot.x = 2.0
        self.assertGreater(robot.sensors()["front"], front + 1.0)

    def test_rays_have_angle_and_distance(self):
        walls = [(r, 10) for r in range(12)]
        robot = Robot(open_map(walls=walls), 4.0, 2.0, 0.0)
        data = robot.sensors()
        self.assertEqual(len(data["rays"]), 16)
        self.assertEqual(set(data["rays"][0]), {"a", "d"})
        # 第 8 条射线正对车头（角度 0），到墙约 1.0 m
        self.assertAlmostEqual(data["rays"][8]["a"], 0.0, delta=1e-6)
        self.assertAlmostEqual(data["rays"][8]["d"], 1.0, delta=0.06)

    def test_reset_restores_home(self):
        robot = Robot(open_map(), 2.0, 2.0, 0.0)
        robot.apply({"type": "move", "direction": "forward", "distance": 1.0})
        robot.advance(2.0)
        robot.apply({"type": "reset"})
        self.assertAlmostEqual(robot.x, 2.0, delta=1e-9)
        self.assertAlmostEqual(robot.y, 2.0, delta=1e-9)
        self.assertEqual(robot.status, "idle")


if __name__ == "__main__":
    unittest.main(verbosity=2)

