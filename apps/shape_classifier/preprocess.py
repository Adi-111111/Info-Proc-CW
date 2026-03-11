import math


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def remove_consecutive_duplicates(points):
    if not points:
        return []
    out = [points[0]]
    for p in points[1:]:
        if p != out[-1]:
            out.append(p)
    return out


def remove_small_movements(points, min_distance=2.0):
    if not points:
        return []
    out = [points[0]]
    for p in points[1:]:
        if dist(out[-1], p) >= min_distance:
            out.append(p)
    return out


def signed_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def turning_strength(points, i):
    n = len(points)

    p_prev = points[(i - 1) % n]
    p = points[i]
    p_next = points[(i + 1) % n]

    v1 = (p_prev[0] - p[0], p_prev[1] - p[1])
    v2 = (p_next[0] - p[0], p_next[1] - p[1])

    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])

    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0

    cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cosang = max(-1.0, min(1.0, cosang))
    ang = math.degrees(math.acos(cosang))

    # larger value = sharper corner
    return 180.0 - ang


def canonicalize_closed_stroke(points):
    if not points:
        return points

    pts = points[:]

    # Force consistent orientation
    if signed_area(pts) < 0:
        pts = pts[::-1]

    strengths = [turning_strength(pts, i) for i in range(len(pts))]
    best_idx = max(range(len(pts)), key=lambda i: strengths[i])

    # If the shape is smooth/circle-ish, fall back to old stable rule
    if strengths[best_idx] < 20.0:
        best_idx = min(range(len(pts)), key=lambda i: (pts[i][1], pts[i][0]))

    pts = pts[best_idx:] + pts[:best_idx]
    return pts


def is_closed(points, threshold=0.15):
    if len(points) < 3:
        return False

    d = dist(points[0], points[-1])

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    return d / scale < threshold


def cumulative_lengths(points):
    if not points:
        return []
    lengths = [0.0]
    for i in range(1, len(points)):
        lengths.append(lengths[-1] + dist(points[i - 1], points[i]))
    return lengths


def interpolate(p1, p2, t):
    return (
        p1[0] + t * (p2[0] - p1[0]),
        p1[1] + t * (p2[1] - p1[1])
    )


def resample_stroke(points, num_points=32):
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [points[0]] * num_points

    lengths = cumulative_lengths(points)
    total_len = lengths[-1]

    if total_len == 0:
        return [points[0]] * num_points

    targets = [i * total_len / (num_points - 1) for i in range(num_points)]
    out = []
    seg_idx = 0

    for target in targets:
        while seg_idx < len(lengths) - 2 and lengths[seg_idx + 1] < target:
            seg_idx += 1

        d1 = lengths[seg_idx]
        d2 = lengths[seg_idx + 1]
        p1 = points[seg_idx]
        p2 = points[seg_idx + 1]

        if d2 == d1:
            out.append(p1)
        else:
            t = (target - d1) / (d2 - d1)
            out.append(interpolate(p1, p2, t))

    return out


def normalise_stroke(points):
    if not points:
        return []

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    width = max_x - min_x
    height = max_y - min_y
    scale = max(width, height)

    if scale < 1e-6:
        scale = 1.0

    return [((x - cx) / scale, (y - cy) / scale) for x, y in points]


def flatten_points(points):
    flat = []
    for x, y in points:
        flat.append(x)
        flat.append(y)
    return flat


def compute_geometry_features(raw_points, norm_points):
    if not raw_points or not norm_points:
        return [0.0] * 6

    raw_xs = [p[0] for p in raw_points]
    raw_ys = [p[1] for p in raw_points]
    raw_w = max(raw_xs) - min(raw_xs)
    raw_h = max(raw_ys) - min(raw_ys)
    raw_scale = max(raw_w, raw_h, 1e-6)

    closed_flag = 1.0 if is_closed(raw_points) else 0.0
    start_end_dist_norm = dist(raw_points[0], raw_points[-1]) / raw_scale
    aspect_ratio = raw_w / max(raw_h, 1e-6)

    path_len = cumulative_lengths(raw_points)[-1] if len(raw_points) > 1 else 0.0
    path_length_norm = path_len / raw_scale

    xs = [p[0] for p in norm_points]
    ys = [p[1] for p in norm_points]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    radii = [math.hypot(x - cx, y - cy) for x, y in norm_points]
    r_mean = sum(radii) / len(radii)
    radial_std_norm = math.sqrt(sum((r - r_mean) ** 2 for r in radii) / len(radii))

    area = signed_area(norm_points)

    return [
        closed_flag,
        start_end_dist_norm,
        aspect_ratio,
        path_length_norm,
        radial_std_norm,
        area,
    ]


def preprocess(points, num_points=32, min_distance=2.0):
    points = remove_consecutive_duplicates(points)
    points = remove_small_movements(points, min_distance=min_distance)

    if len(points) < 2:
        return None

    points = resample_stroke(points, num_points=num_points)

    if is_closed(points):
        points = canonicalize_closed_stroke(points)

    points = normalise_stroke(points)
    return points


def preprocess_to_vector(points, num_points=32, min_distance=2.0):
    cleaned = remove_consecutive_duplicates(points)
    cleaned = remove_small_movements(cleaned, min_distance=min_distance)

    if len(cleaned) < 2:
        return None

    resampled = resample_stroke(cleaned, num_points=num_points)

    if is_closed(resampled):
        resampled = canonicalize_closed_stroke(resampled)

    norm_points = normalise_stroke(resampled)

    coord_features = flatten_points(norm_points)
    geom_features = compute_geometry_features(cleaned, norm_points)

    return coord_features + geom_features