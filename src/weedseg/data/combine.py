"""Combine converted datasets into one training set, and choose its classes.

Each dataset is first converted on its own (taxonomy `species`: one category per EPPO code,
small plants as ignore regions, its own balanced split). Combining merges the splits and
decides the classes on the combined training images, so the rules see every dataset:

1. A species code with at least `min_images` training images is a class.
2. A rarer species goes to OTHER_CATEGORY (handed to a generic model at inference).
3. A coarser code (e.g. a genus-only label) goes to OTHER_CATEGORY, unless a class belongs to
   it: then it becomes an ignore region, since such a plant may well be of that class. A code
   above order, e.g. Plantae (1PLAK) for a plant of unknown species, counts as holding a
   class, since the taxa table doesn't record its species.
4. Ignore regions stay ignore regions (e.g. unidentified plants).

Every plant keeps its own EPPO code in an `eppo` field, whatever class it ends up in. Plants
that are ignore regions only because of rule 3 are marked `hidden_plant`: the two-stage
detector learns them as plants (weedseg.data.plant), unlike plants the converter left out
(unidentified, too small, untrusted or without a usable outline).

A hidden plant's label still rules out some classes, e.g. a *Poa* can only be *Poa annua*
or other_plant, and a PhenoBench weed (`none_of` BEAVA in its label map) anything but sugar
beet. Such plants get `allowed_classes`, and the species classifier learns from them that
they belong to one of those classes (a partial label, see `allowed_classes`).
"""

from collections import Counter
from typing import Any

from weedseg.data.coco import IGNORE_CATEGORY, OTHER_CATEGORY
from weedseg.eppo import ABOVE_ORDER, PARENT_RANKS

SPLITS = ("train", "val", "test")
ROLES = ("crop", "weed")


def common_species(codes: set[str], images: Counter, taxa: dict[str, dict[str, str]],
                   min_images: int) -> dict[str, str]:
    """Class of each EPPO code under rules 1-3: the code itself, OTHER_CATEGORY, or "" for
    an ignore region. `images` counts the training images of each code."""
    missing = sorted(code for code in codes if not taxa.get(code, {}).get("rank"))
    if missing:
        raise ValueError(f"EPPO codes missing from the taxa table: {', '.join(missing)}")
    classes = {code for code in codes
               if taxa[code]["rank"] == "species" and images[code] >= min_images}
    ancestors = {taxa[code][rank] for code in classes for rank in PARENT_RANKS}
    if classes:  # which species a code above order holds isn't recorded: assume a class
        ancestors |= {code for code in codes if taxa[code]["rank"] in ABOVE_ORDER}
    decisions = {}
    for code in codes:
        if code in classes:
            decisions[code] = code
        elif taxa[code]["rank"] == "species" or code not in ancestors:
            decisions[code] = OTHER_CATEGORY
        else:
            decisions[code] = ""
    return decisions


def within(code: str, taxon: str, taxa: dict[str, dict[str, str]]) -> bool:
    """Whether the species class `code` belongs to `taxon` (itself, its genus, family or
    order)."""
    return taxon == code or taxon in (taxa[code][rank] for rank in PARENT_RANKS)


def allowed_classes(code: str, role: str, one_of: list[str], none_of: list[str],
                    classes: list[str], class_roles: dict[str, str],
                    taxa: dict[str, dict[str, str]]) -> list[str] | None:
    """The classes a plant hidden by rule 3 may belong to, from what its label says; None
    when that is every class (nothing to learn).

    - A genus, family or order code allows the classes in that taxon. If the label's role
      is crop or weed, only those of the same role: a crop labelled *Papaver* is not the
      weed *Papaver rhoeas*.
    - A code above order (e.g. 1PLAK) allows every class.
    - `one_of` keeps the classes in one of those taxa, `none_of` drops those in any of them.
    - other_plant is allowed too (a species without a class of its own), unless `one_of`
      names only species that are all classes.
    """
    species = [c for c in classes if c != OTHER_CATEGORY]
    if taxa[code]["rank"] in PARENT_RANKS:
        kept = [c for c in species if within(c, code, taxa)
                and (role not in ROLES or class_roles[c] not in ROLES or class_roles[c] == role)]
    else:
        kept = species
    if one_of:
        kept = [c for c in kept if any(within(c, taxon, taxa) for taxon in one_of)]
    kept = [c for c in kept if not any(within(c, taxon, taxa) for taxon in none_of)]
    other = OTHER_CATEGORY in classes and not (one_of and all(t in species for t in one_of))
    allowed = kept + [OTHER_CATEGORY] * other
    return allowed if allowed and len(allowed) < len(classes) else None


def describe_classes(allowed: list[str], classes: list[str]) -> str:
    """A set of classes, listed, or as "every class but ..." when that is shorter."""
    if 2 * len(allowed) <= len(classes):
        return ", ".join(allowed)
    return "every class but " + ", ".join(c for c in classes if c not in allowed)


def class_report(decisions: dict[str, str], images: Counter, taxa: dict[str, dict[str, str]],
                 min_images: int) -> list[str]:
    """Report lines: every class, every species sent to OTHER_CATEGORY, every coarser code."""
    def line(code: str) -> str:
        return f"  {code:6} {taxa[code]['name'][:30]:30} {images[code]:5} images"

    def by_images(codes):
        return sorted(codes, key=lambda c: (-images[c], c))

    classes = [code for code, decision in decisions.items() if decision == code]
    species = [code for code in decisions if taxa[code]["rank"] == "species"]
    return [
        f"{len(classes)} classes (species with >= {min_images} training images):",
        *(line(code) for code in by_images(classes)),
        f"{OTHER_CATEGORY} (rarer species):",
        *(line(code) for code in by_images(set(species) - set(classes))),
        "Labels coarser than species:",
        *(f"{line(code)} -> {decisions[code] or 'ignore (one of its species is a class)'}"
          for code in sorted(set(decisions) - set(species))),
    ]


def partial_report(groups: Counter, classes: list[str],
                   taxa: dict[str, dict[str, str]]) -> list[str]:
    """Report lines: the partial labels, per dataset and label, with their training plants."""
    if not groups:
        return []
    return ["Partial labels (hidden plants the classifier learns as one of these classes):",
            *(f"  {code:6} {taxa[code]['name'][:22]:22} {dataset:16} {n:5} train plants -> "
              f"{describe_classes(list(allowed), classes)}"
              for (dataset, code, allowed), n in sorted(groups.items(),
                                                        key=lambda kv: (-kv[1], kv[0])))]


def combine(datasets: dict[str, dict[str, dict[str, Any]]], taxa: dict[str, dict[str, str]],
            min_images: int) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Merge converted datasets ({name: {split: COCO dict}}) and choose the classes.

    Returns the combined COCO dict of each split and the class report. Image file names
    become paths relative to the project root (`info.image_dir` is "."), since each dataset
    keeps its images in its own folder. Hidden plants whose label rules out some classes get
    `allowed_classes` (see `allowed_classes`).
    """
    sizes = {name: {coco["info"].get("min_plant_size_mm") for coco in splits.values()}
             for name, splits in datasets.items()}
    if len({size for values in sizes.values() for size in values}) > 1:
        raise ValueError(f"The datasets were converted with different min_plant_size_mm: {sizes}")
    code_of = {}  # (dataset, category id) -> EPPO code; None for the ignore category
    role_of = {}  # (dataset, category id) -> its role (supercategory)
    roles: dict[str, set[str]] = {}
    for name, splits in datasets.items():
        if set(splits) != set(SPLITS):
            raise ValueError(f"{name} must have the splits {', '.join(SPLITS)}")
        for category in splits["train"]["categories"]:
            if category["name"] == IGNORE_CATEGORY:
                code_of[name, category["id"]] = None
                continue
            if category["name"] not in taxa:
                raise ValueError(f"Category {category['name']!r} of {name} is not an EPPO code; "
                                 "convert the datasets with taxonomy species")
            code_of[name, category["id"]] = category["name"]
            role_of[name, category["id"]] = category["supercategory"]
            roles.setdefault(category["name"], set()).add(category["supercategory"])

    images = Counter()  # training images per EPPO code, over all datasets
    for name, splits in datasets.items():
        per_image = {}
        for annotation in splits["train"]["annotations"]:
            code = code_of[name, annotation["category_id"]]
            if code and not annotation.get("iscrowd"):
                per_image.setdefault(annotation["image_id"], set()).add(code)
        for codes in per_image.values():
            images.update(codes)
    decisions = common_species(set(roles), images, taxa, min_images)

    def role(codes: list[str]) -> str:
        found = {r for code in codes for r in roles[code]}
        return found.pop() if len(found) == 1 else "mixed"

    members: dict[str, list[str]] = {}
    for code, decision in sorted(decisions.items()):
        if decision:
            members.setdefault(decision, []).append(code)
    names = sorted(members, key=lambda n: (role(members[n]) != "crop", n))
    categories = [{
        "id": i + 1,
        "name": name,
        "supercategory": "other" if name == OTHER_CATEGORY else role(members[name]),
        "eppo_codes": members[name],
        "scientific_names": [taxa[code]["name"] for code in members[name]],
    } for i, name in enumerate(names)]
    categories.append({"id": len(categories) + 1, "name": IGNORE_CATEGORY,
                       "supercategory": IGNORE_CATEGORY, "eppo_codes": [],
                       "scientific_names": []})
    category_id = {c["name"]: c["id"] for c in categories}
    class_roles = {c["name"]: c["supercategory"] for c in categories}

    combined = {}
    partial_groups = Counter()  # (dataset, EPPO code, allowed classes) -> training plants
    for split in SPLITS:
        out_images, out_annotations = [], []
        hidden = partial = 0
        for name, splits in datasets.items():
            coco = splits[split]
            image_dir = coco["info"]["image_dir"].rstrip("/")
            new_image_id = {}
            for image in coco["images"]:
                new_image_id[image["id"]] = len(out_images) + 1
                out_images.append({**image, "id": len(out_images) + 1, "dataset": name,
                                   "file_name": f"{image_dir}/{image['file_name']}"})
            for annotation in coco["annotations"]:
                own_id = annotation.get("ignored_category_id", annotation["category_id"])
                code = code_of[name, own_id]
                decision = decisions[code] if code else ""
                new = {key: value for key, value in annotation.items()
                       if key != "ignored_category_id"}
                new.update(id=len(out_annotations) + 1,
                           image_id=new_image_id[annotation["image_id"]])
                if code:
                    new["eppo"] = code
                if decision and own_id == annotation["category_id"]:
                    new["category_id"] = category_id[decision]
                else:  # ignore region: unidentified, too small (size filter) or rule 3
                    new.update(category_id=category_id[IGNORE_CATEGORY], iscrowd=1)
                    if code and not annotation.get("iscrowd"):  # rule 3 alone
                        new["hidden_plant"] = True
                        hidden += 1
                        allowed = allowed_classes(code, role_of[name, own_id],
                                                  annotation.get("one_of", []),
                                                  annotation.get("none_of", []), names,
                                                  class_roles, taxa)
                        if allowed:
                            new["allowed_classes"] = allowed
                            partial += 1
                            if split == "train":
                                partial_groups[name, code, tuple(allowed)] += 1
                out_annotations.append(new)
        combined[split] = {
            "info": {
                "description": f"{' + '.join(datasets)}, {split} split, species with at least "
                               f"{min_images} training images",
                "datasets": {name: splits[split]["info"] for name, splits in datasets.items()},
                "licences": sorted({splits[split]["info"].get("licence", "unknown")
                                    for splits in datasets.values()}),
                "image_dir": ".",
                "split": split,
                "min_train_images": min_images,
                "min_plant_size_mm": next(iter(next(iter(sizes.values())))),
                "hidden_plants": hidden,
                "partial_labels": partial,
            },
            "images": out_images,
            "annotations": out_annotations,
            "categories": categories,
        }
    report = class_report(decisions, images, taxa, min_images)
    return combined, report + partial_report(partial_groups, names, taxa)
