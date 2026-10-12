from typing import Tuple, List, Dict, Any
import numpy as np


def project_point(
    point: np.ndarray,
    img_size: int = 224,
    camera_pos: np.ndarray = np.array([0.0, -0.6, 0.7]),
    camera_target: np.ndarray = np.array([0.0, 0.0, 0.1]),
    fov: float = 60.0,
) -> Tuple[int, int, float]:
    """
    Project 3D world coordinate to 2D image coordinates (u, v) and depth.
    """
    # Camera coordinate system
    forward = camera_target - camera_pos
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    rel_pos = point - camera_pos
    x_cam = np.dot(rel_pos, right)
    y_cam = np.dot(rel_pos, up)
    z_cam = np.dot(rel_pos, forward)

    if z_cam <= 0.01:
        z_cam = 0.01

    focal = (img_size / 2.0) / np.tan(np.radians(fov / 2.0))
    u = int(img_size / 2.0 + (x_cam / z_cam) * focal)
    v = int(img_size / 2.0 - (y_cam / z_cam) * focal)

    return u, v, z_cam


def draw_circle(
    canvas: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw solid filled circle on canvas."""
    H, W, _ = canvas.shape
    u0, v0 = center
    r = max(1, radius)
    u_min, u_max = max(0, u0 - r), min(W, u0 + r + 1)
    v_min, v_max = max(0, v0 - r), min(H, v0 + r + 1)

    for v in range(v_min, v_max):
        for u in range(u_min, u_max):
            if (u - u0) ** 2 + (v - v0) ** 2 <= r**2:
                canvas[v, u] = color


def draw_box(
    canvas: np.ndarray,
    center: Tuple[int, int],
    width: int,
    height: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw solid filled rectangle on canvas."""
    H, W, _ = canvas.shape
    u0, v0 = center
    w2, h2 = max(1, width // 2), max(1, height // 2)
    u_min, u_max = max(0, u0 - w2), min(W, u0 + w2 + 1)
    v_min, v_max = max(0, v0 - h2), min(H, v0 + h2 + 1)
    canvas[v_min:v_max, u_min:u_max] = color


def render_scene(
    img_size: int,
    table_bounds: Tuple[float, float, float, float],
    ee_pos: np.ndarray,
    gripper_open: float,
    object_pos: np.ndarray = None,
    goal_pos: np.ndarray = None,
    drawer_pos: np.ndarray = None,
    drawer_open_dist: float = 0.0,
    task_name: str = "reach",
) -> np.ndarray:
    """
    Render third-person RGB camera view of robot workspace.
    """
    # Background: clean lab environment wall gradient
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for y in range(img_size):
        # subtle top-to-bottom wall tone
        tone = int(195 + 25 * (y / img_size))
        canvas[y, :] = [tone, tone - 5, tone - 10]

    # Draw tabletop surface
    x_min, x_max, y_min, y_max = table_bounds
    table_corners = [
        np.array([x_min, y_min, 0.0]),
        np.array([x_max, y_min, 0.0]),
        np.array([x_max, y_max, 0.0]),
        np.array([x_min, y_max, 0.0]),
    ]
    poly_pts = [project_point(p, img_size)[:2] for p in table_corners]
    
    # Fill table quad
    min_v = max(0, min(p[1] for p in poly_pts))
    max_v = min(img_size - 1, max(p[1] for p in poly_pts))
    min_u = max(0, min(p[0] for p in poly_pts))
    max_u = min(img_size - 1, max(p[0] for p in poly_pts))
    canvas[min_v : max_v + 1, min_u : max_u + 1] = [170, 150, 130] # wood/tabletop tone

    # Table rim shadow
    canvas[max_v : min(img_size, max_v + 4), min_u : max_u + 1] = [110, 95, 80]

    # Draw Goal Region if applicable
    if goal_pos is not None:
        gu, gv, gz = project_point(goal_pos, img_size)
        g_rad = int(14 * (0.6 / max(0.2, gz)))
        if task_name == "reach":
            # Glowing translucent green target ring
            draw_circle(canvas, (gu, gv), g_rad + 2, (50, 180, 50))
            draw_circle(canvas, (gu, gv), max(1, g_rad - 2), (210, 240, 210))
        elif task_name == "push":
            # Target zone circle
            draw_circle(canvas, (gu, gv), g_rad + 2, (40, 120, 220))
            draw_circle(canvas, (gu, gv), max(1, g_rad - 3), (180, 210, 245))
        elif task_name == "pick_place":
            # Receptacle basket / box
            draw_box(canvas, (gu, gv), g_rad * 2 + 4, g_rad * 2 + 4, (80, 80, 80))
            draw_box(canvas, (gu, gv), g_rad * 2, g_rad * 2, (200, 200, 200))

    # Draw Drawer fixture if drawer_open task
    if task_name == "drawer_open" and drawer_pos is not None:
        du, dv, dz = project_point(drawer_pos, img_size)
        dw = int(38 * (0.6 / max(0.2, dz)))
        dh = int(24 * (0.6 / max(0.2, dz)))
        # Main cabinet housing
        draw_box(canvas, (du, dv), dw, dh, (90, 70, 50))
        # Sliding drawer front
        slide_offset_y = int(drawer_open_dist * 70)
        draw_box(canvas, (du, dv + slide_offset_y), dw - 4, dh - 6, (140, 105, 75))
        # Handle
        draw_box(canvas, (du, dv + slide_offset_y), 10, 4, (220, 220, 220))

    # Draw Manipulated Object if applicable
    if object_pos is not None and task_name in ["push", "pick_place"]:
        ou, ov, oz = project_point(object_pos, img_size)
        ow = int(16 * (0.6 / max(0.2, oz)))
        # Object shadow on table
        shadow_u, shadow_v, _ = project_point(np.array([object_pos[0], object_pos[1], 0.005]), img_size)
        draw_circle(canvas, (shadow_u, shadow_v), ow // 2, (120, 105, 90))
        # Solid colored manipulation block (Crimson red / Coral)
        draw_box(canvas, (ou, ov), ow, ow, (215, 45, 45))
        # Top highlight
        draw_box(canvas, (ou, ov - ow // 4), ow - 2, ow // 2, (240, 90, 90))

    # Draw Robot End-Effector (Franka Panda parallel-jaw gripper)
    eu, ev, ez = project_point(ee_pos, img_size)
    ew = int(22 * (0.6 / max(0.2, ez)))
    
    # EE Wrist flange
    draw_box(canvas, (eu, ev - ew // 2), ew, ew // 2, (60, 60, 65))
    
    # Gripper fingers
    finger_spread = int((gripper_open) * 6)
    finger_w = 4
    finger_h = int(12 * (0.6 / max(0.2, ez)))
    # Left finger
    draw_box(canvas, (eu - finger_spread - 3, ev + finger_h // 2), finger_w, finger_h, (180, 180, 185))
    # Right finger
    draw_box(canvas, (eu + finger_spread + 3, ev + finger_h // 2), finger_w, finger_h, (180, 180, 185))

    return canvas
