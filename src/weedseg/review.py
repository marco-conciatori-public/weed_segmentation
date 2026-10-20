"""The model's worst errors, sorted by what they would cost the robot, for review by an expert.

Each labelled plant and each detection (paired by evaluation.match_detections) is given an
error type through the roles of the decision rules (see weedseg.decisions). Within a type,
errors are ranked by the model's confidence; plants the model missed have no confidence and
are ranked by size instead, largest first.
"""

from collections import defaultdict
from dataclasses import dataclass

from weedseg.data.coco import OTHER_CATEGORY
from weedseg.decisions import decide
from weedseg.evaluation import Match

# (folder name, description), in order of cost to the robot.
ERROR_TYPES = [
    ("crop_removed",
     "Crop the robot would remove: labelled as a crop, predicted as a harmful species."),
    ("harmful_kept",
     "Harmful weed the robot would keep: predicted as a crop or a neutral species."),
    ("missed_harmful",
     "Harmful weed the model did not detect at all (ranked by size)."),
    ("neutral_removed",
     "Neutral plant the robot would remove: predicted as a harmful species."),
    ("other_plant_removed",
     ("Plant of a species the model doesn't know (other_plant) that the robot would "
      "remove instead of handing it to the generic model.")),
    ("no_plant_detection",
     ("Confident detection where no plant is labelled: either a plant the annotators "
      "missed, or a false detection.")),
    ("same_role_wrong_species",
     ("Wrong species with the same role (no effect with the current rules, but it may "
      "matter with others).")),
    ("other_wrong_class",
     ("Other wrong classes, e.g. a crop called neutral, or a wrong class that is deferred "
      "to the generic model.")),
    ("missed_other",
     "Crop, neutral or other_plant not detected at all (ranked by size)."),
]


@dataclass
class ReviewItem:
    error_type: str
    match: Match
    true_name: str | None  # class names (EPPO codes or other_plant)
    predicted_name: str | None
    action: str | None  # what the robot would do with the detection

    @property
    def score(self) -> float | None:
        return self.match.detection["score"] if self.match.detection else None

    @property
    def box(self) -> list[float]:
        """The box of the plant (or of the detection when no plant is labelled), x y w h."""
        return (self.match.plant or self.match.detection)["bbox"]


def classify(match: Match, names: dict[int, str], roles: dict[str, str],
             defer_below: float | None) -> ReviewItem | None:
    """The error type of a match, or None when the class is right."""
    plant, detection = match.plant, match.detection
    true_name = names[plant["category_id"]] if plant else None
    predicted_name = names[detection["category_id"]] if detection else None
    action = None
    if detection is not None:
        action = decide(detection, names, roles, defer_below)

    if plant is None:
        error_type = "no_plant_detection"
    elif detection is None:
        error_type = "missed_harmful" if roles[true_name] == "harmful" else "missed_other"
    elif true_name == predicted_name:
        return None
    else:
        true_role = roles[true_name]
        if true_role == "crop" and action == "remove":
            error_type = "crop_removed"
        elif true_role == "harmful" and action == "keep":
            error_type = "harmful_kept"
        elif true_role == "neutral" and action == "remove":
            error_type = "neutral_removed"
        elif true_role == OTHER_CATEGORY and action == "remove":
            error_type = "other_plant_removed"
        elif true_role == roles[predicted_name]:
            error_type = "same_role_wrong_species"
        else:
            error_type = "other_wrong_class"
    return ReviewItem(error_type, match, true_name, predicted_name, action)


def rank_errors(items: list[ReviewItem]) -> dict[str, list[ReviewItem]]:
    """Errors per type, most confident first (missed plants: largest first)."""
    by_type = defaultdict(list)
    for item in items:
        by_type[item.error_type].append(item)
    for error_type, group in by_type.items():
        if error_type.startswith("missed_"):
            group.sort(key=lambda item: -max(item.box[2:]))
        else:
            group.sort(key=lambda item: -item.score)
    return {error_type: by_type.get(error_type, []) for error_type, _ in ERROR_TYPES}


def crop_window(box: list[float], width: int, height: int, min_side: int = 400,
                context: float = 1.0) -> tuple[int, int, int, int]:
    """A square-ish window (x0, y0, x1, y1) around a box (x y w h), with `context` times the
    box's longer side around it and at least `min_side` px, moved inside the image."""
    x, y, w, h = box
    side = max(min_side, max(w, h) * (1 + 2 * context))
    cx, cy = x + w / 2, y + h / 2
    x0 = min(max(0, round(cx - side / 2)), max(0, width - round(side)))
    y0 = min(max(0, round(cy - side / 2)), max(0, height - round(side)))
    return x0, y0, min(width, x0 + round(side)), min(height, y0 + round(side))
