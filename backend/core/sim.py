"""二维巡检机器人仿真内核（对应方案中的 B · 嵌入式）。

设计约定（与 docs/02_小组分工.md 第七章一致）：
- 世界坐标，单位米；x 向右、y 向上；theta 为弧度，0 指向 +x，逆时针为正。
- 栅格地图 grid[row][col]，row 0 在最下方，每格 cell 米。
- 机器人视为半径 radius 的圆，撞墙/越界时停在原地并置 status="error"。
- 纯标准库实现，无第三方依赖，可直接被 FastAPI 调用，也可单独跑测试。

命令行自测：
    python core/sim.py
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

CELL = 0.5          # 每格边长（米）
RADIUS = 0.18       # 机器人碰撞半径（米）
V_MAX = 1.0         # 最大线速度（m/s）
W_MAX = 2.0         # 最大角速度（rad/s）
ARRIVE = 0.02       # 到点判定（米）
ANGLE_ARRIVE = 0.02  # 转向到位判定（弧度）
SUB_STEP = 0.02     # 积分步长（秒）
LIDAR_RANGE = 5.0   # 雷达量程（米）
QUEUE_MAX = 8       # 指令队列上限


def wrap_angle(a: float) -> float:
    """把角度归一化到 (-pi, pi]。"""
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


class GridMap:
    """栅格地图：0 可通行，1 障碍。row 0 对应 y 最小的一行。"""

    def __init__(self, grid: Sequence[Sequence[int]], cell: float = CELL):
        if not grid or not grid[0]:
            raise ValueError("grid 不能为空")
        self.grid: List[List[int]] = [list(row) for row in grid]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.cell = cell

    @property
    def width(self) -> float:
        return self.cols * self.cell

    @property
    def height(self) -> float:
        return self.rows * self.cell

    def is_obstacle(self, row: int, col: int) -> bool:
        if row < 0 or col < 0 or row >= self.rows or col >= self.cols:
            return True
        return self.grid[row][col] == 1

    def obstacle_at(self, x: float, y: float) -> bool:
        """点 (x, y) 是否落在障碍格内；越界视为障碍。"""
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return True
        return self.is_obstacle(int(y // self.cell), int(x // self.cell))

    def blocked(self, x: float, y: float, radius: float = RADIUS) -> bool:
        """圆 (x, y, radius) 是否与边界或障碍相交。"""
        if x - radius < 0 or y - radius < 0:
            return True
        if x + radius > self.width or y + radius > self.height:
            return True
        c0, c1 = int((x - radius) // self.cell), int((x + radius) // self.cell)
        r0, r1 = int((y - radius) // self.cell), int((y + radius) // self.cell)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if not self.is_obstacle(rr, cc):
                    continue
                x0, y0 = cc * self.cell, rr * self.cell
                x1, y1 = x0 + self.cell, y0 + self.cell
                dx = max(x0 - x, 0.0, x - x1)
                dy = max(y0 - y, 0.0, y - y1)
                if dx * dx + dy * dy < radius * radius:
                    return True
        return False

    def raycast(self, x: float, y: float, angle: float,
                max_range: float = LIDAR_RANGE, step: float = 0.02) -> float:
        """从 (x, y) 沿 angle 射出，返回命中距离；未命中返回 max_range。"""
        ca, sa = math.cos(angle), math.sin(angle)
        d = 0.0
        while d < max_range:
            d = min(d + step, max_range)
            if self.obstacle_at(x + ca * d, y + sa * d):
                return d
        return max_range


class Robot:
    """差速机器人：接受离散指令，按固定步长推进。"""

    def __init__(self, grid_map: GridMap, x: float, y: float,
                 theta: float = 0.0, radius: float = RADIUS):
        self.map = grid_map
        self.radius = radius
        self.home = (x, y, theta)
        self.reset()

    # ---------------------------------------------------------------- 状态
    def reset(self) -> None:
        self.x, self.y, self.theta = self.home
        self.status = "idle"
        self.error: str | None = None
        self.mode: str | None = None
        self._tx = self.x
        self._ty = self.y
        self._target_theta = self.theta
        self._queue: List[Dict] = []

    def state(self) -> Dict[str, float | str | None]:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "theta": round(wrap_angle(self.theta), 4),
            "status": self.status,
            "error": self.error,
            "queue": len(self._queue),
        }

    def sensors(self, rays: int = 16) -> Dict[str, object]:
        """返回三向距离、一圈扫描点，以及带角度的射线（前端画扇形用）。"""
        angles: List[float] = []
        for i in range(rays):
            # 除以 rays（而非 rays-1），保证偶数条时正前方（0 弧度）恰好有一条射线
            angles.append(wrap_angle(-math.pi + 2 * math.pi * i / rays))

        scan: List[float] = [
            round(self.map.raycast(self.x, self.y, self.theta + a), 3) for a in angles
        ]

        def cone(center: float, half: float) -> float:
            vals = [scan[i] for i in range(rays) if abs(wrap_angle(angles[i] - center)) <= half]
            return round(min(vals), 3) if vals else round(LIDAR_RANGE, 3)

        return {
            "front": cone(0.0, math.pi / 8),
            "left": cone(math.pi / 2, math.pi / 8),
            "right": cone(-math.pi / 2, math.pi / 8),
            "scan": scan,
            "rays": [{"a": round(a, 4), "d": d} for a, d in zip(angles, scan)],
        }

    # ---------------------------------------------------------------- 指令
    def apply(self, cmd: Dict) -> Dict[str, object]:
        """执行一条指令；正在移动时自动排队，队列满返回 BUSY。"""
        kind = cmd.get("type")

        if kind == "reset":
            self.reset()
            return {"ok": True}

        if kind == "stop":
            self._queue.clear()
            self.mode = None
            self.status = "idle"
            self.error = None
            return {"ok": True}

        if kind == "move":
            if float(cmd.get("distance", 0.0)) <= 0:
                return self._fail("INVALID_COMMAND", "距离必须大于 0")
        elif kind == "turn":
            if float(cmd.get("angle", math.pi / 2)) == 0:
                return self._fail("INVALID_COMMAND", "角度不能为 0")
        else:
            return self._fail("INVALID_COMMAND", f"未知指令: {kind!r}")

        if self.mode is not None:
            if len(self._queue) >= QUEUE_MAX:
                return self._fail("BUSY", "指令队列已满，稍后再试")
            self._queue.append(dict(cmd))
            return {"ok": True, "queued": True}

        return self._start(cmd)

    def _start(self, cmd: Dict) -> Dict[str, object]:
        """立即开始执行一条已校验的指令。"""
        self.error = None
        if cmd["type"] == "turn":
            angle = float(cmd.get("angle", math.pi / 2))
            direction = cmd.get("direction", "left")
            delta = angle if direction == "left" else -angle
            self._target_theta = wrap_angle(self.theta + delta)
            self.mode = "turn"
            self.status = "moving"
            return {"ok": True}

        distance = float(cmd.get("distance", 0.0))
        direction = cmd.get("direction", "forward")
        hx, hy = math.cos(self.theta), math.sin(self.theta)
        sign = -1.0 if direction == "backward" else 1.0
        self._tx = self.x + hx * distance * sign
        self._ty = self.y + hy * distance * sign
        if self.map.blocked(self._tx, self._ty, self.radius):
            # 目标点本身不可达：停在原地并给出原因
            self.status = "error"
            self.error = self._block_code(self._tx, self._ty)
            self.mode = None
            return self._fail(self.error, "目标点被边界或障碍占据")
        self.mode = "back" if sign < 0 else "drive"
        self.status = "moving"
        return {"ok": True}

    def _dequeue(self) -> bool:
        """取下一条指令并开始；队列空返回 False。"""
        if not self._queue:
            return False
        self._start(self._queue.pop(0))
        return True

    # ---------------------------------------------------------------- 推进
    def advance(self, dt: float) -> Dict[str, float | str | None]:
        """按秒推进仿真；内部拆成 20 ms 子步，保证结果与调用频率无关。

        调用方应传入真实经过的时间，建议单次不超过 0.2 s；
        长时间挂起后不要一次补完，否则机器人会沿路径“瞬移”。
        """
        if dt <= 0:
            return self.state()
        steps = max(1, int(math.ceil(dt / SUB_STEP)))
        h = dt / steps
        for _ in range(steps):
            if self.status == "error":
                break
            if self.mode is None and not self._dequeue():
                break
            self._substep(h)
        return self.state()

    def _substep(self, h: float) -> None:
        if self.mode == "turn":
            err = wrap_angle(self._target_theta - self.theta)
            if abs(err) <= ANGLE_ARRIVE:
                self.theta = self._target_theta
                self._finish()
            else:
                self.theta = wrap_angle(self.theta + max(-W_MAX * h, min(W_MAX * h, err)))
            return

        dx, dy = self._tx - self.x, self._ty - self.y
        dist = math.hypot(dx, dy)
        if dist <= ARRIVE:
            self.x, self.y = self._tx, self._ty
            self._finish()
            return

        step = min(V_MAX * h, dist)
        if self.mode == "back":
            step = -step
        nx = self.x + math.cos(self.theta) * step
        ny = self.y + math.sin(self.theta) * step

        if self.map.blocked(nx, ny, self.radius):
            self.status = "error"
            self.error = self._block_code(nx, ny)
            self.mode = None
            return
        self.x, self.y = nx, ny

    # ---------------------------------------------------------------- 内部
    def _finish(self) -> None:
        self.mode = None
        self.status = "idle"

    def _block_code(self, x: float, y: float) -> str:
        outside = (x - self.radius < 0 or y - self.radius < 0
                   or x + self.radius > self.map.width or y + self.radius > self.map.height)
        return "OUT_OF_BOUNDS" if outside else "OBSTACLE_BLOCKED"

    def _fail(self, code: str, message: str) -> Dict[str, object]:
        return {"ok": False, "code": code, "message": message}


# -------------------------------------------------------------------- 地图
def demo_map() -> GridMap:
    """一张 40×30 的巡检车间地图：外墙 + 两段隔断 + 四个巡检点。"""
    rows, cols = 30, 40
    grid = [[0] * cols for _ in range(rows)]
    for c in range(cols):
        grid[0][c] = grid[rows - 1][c] = 1
    for r in range(rows):
        grid[r][0] = grid[r][cols - 1] = 1
    # 两段隔断，留出通道
    for r in range(6, 24):
        if r < 14 or r > 18:
            grid[r][10] = 1
    for c in range(18, 34):
        if c < 24 or c > 28:
            grid[20][c] = 1
    return GridMap(grid)


if __name__ == "__main__":
    m = demo_map()
    robot = Robot(m, 1.5, 1.5, 0.0)
    print("初始:", robot.state())

    robot.apply({"type": "move", "direction": "forward", "distance": 1.0})
    robot.advance(3.0)
    print("前进 1 m:", robot.state())

    robot.apply({"type": "turn", "direction": "left", "angle": math.pi / 2})
    robot.advance(2.0)
    print("左转 90°:", robot.state())
    print("传感器:", robot.sensors()["front"], robot.sensors()["left"])
