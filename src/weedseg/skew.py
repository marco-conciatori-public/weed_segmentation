"""Skew a classifier's probabilities with the species expected in a field: one crop list and one
weed list.

The step doesn't know where the lists come from (an expert, a first pass of the model, a
regional list...) and doesn't judge them: a bad list is applied like a good one. A list only
makes the species off it less likely, so the model can still name them. An empty list says
nothing, and that side stays as it was:

1. With a weed list, off-list weeds are divided by the multiplier, then all weeds are scaled
   back to their total. The list says which weeds to expect, not whether the plant is a weed,
   so it can't push a plant towards the crop.
2. With a crop list, off-list crops are divided by the multiplier. That probability goes to all
   the other classes: a crop that isn't in the field doesn't explain the plant.

A class on either list is never divided, and the lists override the `crops` tags: volunteer
potatoes on the weed list count as a weed, oilseed rape on the crop list as a crop. `other_plant`
is never divided either, since it hands the plant to a model that knows more species. Names on
a list that aren't among `classes` have no effect.

The result sums to 1 per row. Decisions made on it are not comparable with those on the raw
probabilities at the same threshold, because a list raises the confidence: compare them at the
same share of crops removed (weedseg.decisions.harmful_removed_at). See docs/twostage.md
(Species lists).
"""

from collections.abc import Collection, Sequence

import numpy as np

from weedseg.data.coco import OTHER_CATEGORY

DEFAULT_MULTIPLIER = 5.0


def skew_probabilities(probs: np.ndarray, classes: Sequence[str], crops: Collection[str],
                       crop_list: Collection[str] = (), weed_list: Collection[str] = (),
                       use_lists: bool = False,
                       multiplier: float = DEFAULT_MULTIPLIER) -> np.ndarray:
    """Probabilities [plants, classes] skewed by the lists (see module doc). `classes` names the
    columns, `crops` tags those that are crops. Without `use_lists`, or with both lists empty,
    a copy of `probs` as it is. `multiplier` must be at least 1."""
    if multiplier < 1:
        raise ValueError(f"multiplier must be at least 1, not {multiplier}")
    probs = np.array(probs, dtype=np.float64)
    crop_list, weed_list = set(crop_list), set(weed_list)
    if not use_lists or not (crop_list or weed_list):
        return probs
    is_crop = np.array([(name in crops or name in crop_list) and name not in weed_list
                        for name in classes])
    is_weed = np.array([name != OTHER_CATEGORY for name in classes]) & ~is_crop
    off_list = np.array([name not in crop_list | weed_list for name in classes])
    if weed_list:
        total = probs[:, is_weed].sum(1, keepdims=True)
        probs[:, is_weed & off_list] /= multiplier
        left = probs[:, is_weed].sum(1, keepdims=True)
        probs[:, is_weed] *= np.divide(total, left, out=np.ones_like(total), where=left > 0)
    if crop_list:
        probs[:, is_crop & off_list] /= multiplier
    total = probs.sum(1, keepdims=True)
    return np.divide(probs, total, out=probs, where=total > 0)
