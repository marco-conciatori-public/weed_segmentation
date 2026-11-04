"""Where each plant of an image stands against the crop rows: how far off the nearest row, how
close to the next plant along it and, where a session's plants are evenly spaced, how far from
where that spacing expects a plant.

The step knows nothing of species or labels. It takes positions in mm and which plants count as
evidence for the rows (e.g. those a classifier calls a crop), and judges every plant from the
other evidence plants only: a plant can't vouch for itself.

Images cover less than a metre, so a session (images of one field and day) supplies what one
image can't:

- the direction of the rows, which an image can refine when it has plants enough;
- the distance between rows, so that the wandering box centres of large plants stay one row;
- the usual gap between plants along a row;
- the pitch, the spacing the sowing machine aimed for: only when the gaps really are periodic,
  which densely sown crops such as beans aren't.

Images are judged alone, without the rows of the images before them. See docs/twostage.md (Crop
rows and spacing).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ANGLES = np.deg2rad(np.arange(180.0))
NORMALS = np.stack([-np.sin(ANGLES), np.cos(ANGLES)], 1)
TANGENTS = np.stack([np.cos(ANGLES), np.sin(ANGLES)], 1)
# Per plant, NaN when it can't be told:
# - row_distance: mm to the nearest row of the others;
# - phase: along that row, how far from an expected plant, as a share of the pitch (0 to 0.5);
# - spacing: mm to the nearest of the others on that row, as a share of the session's usual gap;
# - on_row: how many of the others are on that row;
# - strongest_row: how many of the others are on the image's fullest row;
# - size_ratio: the plant's size against the median of that fullest row.
POSITIONS = np.dtype([(name, np.float64) for name in (
    "row_distance", "phase", "spacing", "on_row", "strongest_row", "size_ratio")])


@dataclass(frozen=True)
class RowSettings:
    """See configs/evaluate_rows.yaml, where each is explained."""

    line_tolerance_mm: float = 30
    same_row_mm: float = 100
    row_spacing_range_mm: tuple[float, float] = (100, 1200)
    pitch_range_mm: tuple[float, float] = (100, 600)
    blur_mm: float = 150
    min_gaps: int = 8
    min_regularity: float = 0.12
    regularity_errors: float = 4


DEFAULTS = RowSettings()


@dataclass
class Geometry:
    """What a session's evidence plants say about its rows."""

    direction_scores: np.ndarray  # per angle of ANGLES, for an average image of the session
    direction: float  # radians
    row_spacing: float | None  # mm, when plants on two rows show it
    same_row: float  # mm across the rows: plants closer than this share a row
    usual_gap: float | None  # mm, the median gap between neighbours on a row
    pitch: float | None  # mm, only when the gaps are periodic enough
    regularity: float  # the periodicity found, whether enough or not
    gaps: int


def _pair_scores(points: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[pairs, angles]: how well each pair of plants lines up along each direction, with the
    plants of each pair."""
    first, second = np.triu_indices(len(points), 1)
    across = (points[first] - points[second]) @ NORMALS.T
    return np.exp(-0.5 * (across / tolerance) ** 2), first, second


def split_rows(across: np.ndarray, same_row: float) -> np.ndarray:
    """The row of each plant, numbered in order, from its position across the rows: neighbours
    further apart than `same_row` are on different rows."""
    order = np.argsort(across)
    rows = np.zeros(len(across), dtype=np.int64)
    rows[order[1:]] = np.cumsum(np.diff(across[order]) > same_row)
    return rows


def find_row_spacing(frames: Sequence[np.ndarray], direction: int,
                     settings: RowSettings) -> float | None:
    """The first clear peak of the distances across the rows between plants of one image."""
    low, high = settings.row_spacing_range_mm
    edges = np.arange(low - 2.5, high + 2.5, 5.0)
    counts = np.zeros(len(edges) - 1)
    for points in frames:
        across = points @ NORMALS[direction]
        first, second = np.triu_indices(len(across), 1)
        counts += np.histogram(np.abs(across[first] - across[second]), edges)[0]
    if counts.sum() < 20:
        return None
    kernel = np.exp(-0.5 * (np.arange(-20, 21) / 4.0) ** 2)  # 5 mm bins: a 20 mm kernel
    density = np.convolve(counts, kernel, mode="same")
    peaks = [k for k in range(1, len(density) - 1)
             if density[k] >= density[k - 1] and density[k] > density[k + 1]
             and density[k] >= 0.5 * density.max()]
    return float(edges[peaks[0]] + 2.5) if peaks else None


def find_pitch(distances: np.ndarray, settings: RowSettings) -> tuple[float | None, float]:
    """The pitch that best fits the distances along a row between all pairs of plants, and how
    well: the mean of cos(2 pi distance / pitch), which is 1 when every distance is a multiple
    of the pitch, so a missing plant does no harm. Long pitches fit anything, since few pairs
    are a period apart: that is taken out by subtracting the same mean after blurring the
    distances, which leaves only the part blurring doesn't wash out, and by asking that enough
    pairs reach most of a period. Fractions of the pitch fit as well as the pitch itself, so the
    longest pitch close to the best fit is the one chosen."""
    low, high = settings.pitch_range_mm
    pitches = np.exp(np.arange(np.log(low), np.log(high), 0.01))
    if not len(distances):
        return None, 0.0
    washed_out = 1 - np.exp(-2 * (np.pi * settings.blur_mm / pitches) ** 2)
    fits = np.array([np.cos(2 * np.pi * distances / pitch).mean() for pitch in pitches])
    enough = np.array([(distances > 0.7 * pitch).mean() > 0.3 for pitch in pitches])
    fits = np.where(enough, fits * washed_out, -1.0)
    if fits.max() <= 0:
        return None, 0.0
    k = int(np.flatnonzero(fits >= 0.85 * fits.max())[-1])
    while k > 0 and fits[k - 1] > fits[k]:
        k -= 1
    return float(pitches[k]), float(fits[k])


def session_geometry(frames: Sequence[np.ndarray],
                     settings: RowSettings = DEFAULTS) -> Geometry | None:
    """The rows of a session from its evidence plants: per image, their positions in mm
    [plants, 2]. None when no image has two of them. Images with three or more decide the
    direction, since any two plants line up; the others only count when there are none such."""
    usable = [p for p in frames if len(p) >= 3] or [p for p in frames if len(p) >= 2]
    if not usable:
        return None
    tolerance = settings.line_tolerance_mm
    scores = sum(_pair_scores(p, tolerance)[0].sum(0) for p in usable) / len(usable)
    direction = int(scores.argmax())
    pairs = [p for p in frames if len(p) >= 2]
    row_spacing = find_row_spacing(pairs, direction, settings)
    same_row = max(settings.same_row_mm, 0.5 * row_spacing) if row_spacing else settings.same_row_mm
    gaps, distances = [], []
    for points in pairs:
        along = points @ TANGENTS[direction]
        rows = split_rows(points @ NORMALS[direction], same_row)
        for row in range(rows.max() + 1):
            on_row = np.sort(along[rows == row])
            first, second = np.triu_indices(len(on_row), 1)
            gaps.append(np.diff(on_row))
            distances.append(on_row[second] - on_row[first])
    gaps, distances = np.concatenate(gaps), np.concatenate(distances)
    enough = len(gaps) >= settings.min_gaps
    pitch, regularity = find_pitch(distances, settings) if enough else (None, 0.0)
    periodic = pitch is not None and regularity >= settings.min_regularity and (
        regularity >= settings.regularity_errors * 0.7 / np.sqrt(len(gaps)))
    return Geometry(scores, float(ANGLES[direction]), row_spacing, same_row,
                    float(np.median(gaps)) if enough else None, pitch if periodic else None,
                    regularity, len(gaps))


def plant_positions(points: np.ndarray, evidence: np.ndarray, geometry: Geometry,
                    sizes: np.ndarray | None = None,
                    settings: RowSettings = DEFAULTS) -> np.ndarray:
    """Each plant of an image against the rows of the other evidence plants: an array of
    POSITIONS. `points` are all its plants in mm, `evidence` says which count for the rows,
    `sizes` are in any one unit. Each plant's direction comes from the session's plus the image's
    without it, so an image with plants enough can correct the session's."""
    found = np.full(len(points), np.nan, dtype=POSITIONS)
    witnesses = np.flatnonzero(evidence)
    pair_scores, first, second = _pair_scores(points[witnesses], settings.line_tolerance_mm)
    image_scores = pair_scores.sum(0)
    for plant in range(len(points)):
        own = np.flatnonzero(witnesses == plant)
        others = np.delete(witnesses, own)
        if not len(others):
            continue
        scores = image_scores
        if len(own):
            scores = image_scores - pair_scores[(first == own[0]) | (second == own[0])].sum(0)
        direction = int((scores + geometry.direction_scores).argmax())
        across = points @ NORMALS[direction]
        along = points @ TANGENTS[direction]
        rows = split_rows(across[others], geometry.same_row)
        centres = np.array([np.median(across[others][rows == row])
                            for row in range(rows.max() + 1)])
        distances = np.abs(centres - across[plant])
        nearest = int(distances.argmin())
        on_row = along[others][rows == nearest]
        members = np.bincount(rows)
        found[plant]["row_distance"] = distances[nearest]
        found[plant]["on_row"] = len(on_row)
        found[plant]["strongest_row"] = members.max()
        if geometry.usual_gap:
            found[plant]["spacing"] = np.abs(on_row - along[plant]).min() / geometry.usual_gap
        if geometry.pitch:
            expected = np.angle(np.exp(2j * np.pi * on_row / geometry.pitch).sum()) / (2 * np.pi)
            found[plant]["phase"] = abs((along[plant] / geometry.pitch - expected + 0.5) % 1 - 0.5)
        if sizes is not None:
            found[plant]["size_ratio"] = sizes[plant] / np.median(
                sizes[others][rows == int(members.argmax())])
    return found
