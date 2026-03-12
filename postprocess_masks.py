#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def parse_id_list(text):
    if text is None or str(text).strip() == "":
        return []
    ids = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids


def remove_small_components(mask_arr, class_id, min_area, bg_id):
    class_mask = (mask_arr == class_id)
    if not class_mask.any():
        return 0

    labeled, num = ndi.label(class_mask)
    if num == 0:
        return 0

    sizes = np.bincount(labeled.ravel())
    # label 0 is background in connected-component map
    small_labels = np.where((sizes < min_area) & (np.arange(len(sizes)) != 0))[0]
    if small_labels.size == 0:
        return 0

    remove_mask = np.isin(labeled, small_labels)
    changed = int(remove_mask.sum())
    mask_arr[remove_mask] = bg_id
    return changed


def keep_largest_component(mask_arr, class_id, bg_id):
    class_mask = (mask_arr == class_id)
    if not class_mask.any():
        return 0

    labeled, num = ndi.label(class_mask)
    if num <= 1:
        return 0

    sizes = np.bincount(labeled.ravel())
    # ignore index 0 (background in connected-component map)
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    remove_mask = (labeled != 0) & (labeled != largest_label)
    changed = int(remove_mask.sum())
    mask_arr[remove_mask] = bg_id
    return changed


def _neighbor_majority_label(mask_arr, comp_mask, current_class, fallback_label):
    # 1-pixel outer ring around the component
    ring = ndi.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool)) & (~comp_mask)
    if not ring.any():
        return fallback_label

    labels = mask_arr[ring]
    if labels.size == 0:
        return fallback_label

    vals, cnts = np.unique(labels, return_counts=True)
    # Prefer non-self labels as merge target.
    valid = vals != current_class
    vals = vals[valid]
    cnts = cnts[valid]
    if vals.size == 0:
        return fallback_label
    return int(vals[int(np.argmax(cnts))])


def merge_non_main_components_to_neighbor(mask_arr, class_id, fallback_label):
    """
    Keep largest component of class_id; merge all other components to the
    neighboring class with largest contact length.
    """
    class_mask = (mask_arr == class_id)
    if not class_mask.any():
        return 0

    labeled, num = ndi.label(class_mask)
    if num <= 1:
        return 0

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    changed = 0

    for comp_label in range(1, num + 1):
        if comp_label == largest_label:
            continue
        comp_mask = labeled == comp_label
        target = _neighbor_majority_label(mask_arr, comp_mask, class_id, fallback_label)
        changed += int(comp_mask.sum())
        mask_arr[comp_mask] = target
    return changed


def recursive_neighbor_merge(mask_arr, class_ids, bg_id, max_iter):
    total_changed = 0
    for _ in range(max_iter):
        iter_changed = 0
        for class_id in class_ids:
            iter_changed += merge_non_main_components_to_neighbor(mask_arr, class_id, bg_id)
        total_changed += iter_changed
        if iter_changed == 0:
            break
    return total_changed


def remove_small_foreground_blobs(mask_arr, bg_id, min_area):
    """
    Remove tiny disconnected foreground blobs in one shot.
    This works even when a noisy blob contains multiple classes.
    """
    fg_mask = (mask_arr != bg_id)
    if not fg_mask.any():
        return 0

    labeled, num = ndi.label(fg_mask)
    if num == 0:
        return 0

    sizes = np.bincount(labeled.ravel())
    small_labels = np.where((sizes < min_area) & (np.arange(len(sizes)) != 0))[0]
    if small_labels.size == 0:
        return 0

    remove_mask = np.isin(labeled, small_labels)
    changed = int(remove_mask.sum())
    mask_arr[remove_mask] = bg_id
    return changed


def fill_small_holes(mask_arr, class_id, max_hole_area):
    class_mask = (mask_arr == class_id)
    if not class_mask.any():
        return 0

    filled = ndi.binary_fill_holes(class_mask)
    holes = filled & (~class_mask)
    if not holes.any():
        return 0

    labeled, num = ndi.label(holes)
    if num == 0:
        return 0

    sizes = np.bincount(labeled.ravel())
    if max_hole_area <= 0:
        # Fill all enclosed holes for this class (no area threshold).
        small_holes = np.where(np.arange(len(sizes)) != 0)[0]
    else:
        small_holes = np.where((sizes <= max_hole_area) & (np.arange(len(sizes)) != 0))[0]
    if small_holes.size == 0:
        return 0

    fill_mask = np.isin(labeled, small_holes)
    changed = int(fill_mask.sum())
    mask_arr[fill_mask] = class_id
    return changed


def process_one(mask_path, out_path, args):
    pil = Image.open(mask_path)
    palette = pil.getpalette() if pil.mode == "P" else None
    arr = np.array(pil, dtype=np.uint8)
    before = arr.copy()

    changed = 0
    if args.main_only_mode:
        # Strict mode:
        # 1) For each class, keep largest component unchanged.
        # 2) Merge all non-main components by neighbor-majority, recursively.
        merge_ids = args.main_only_ids
        if not merge_ids:
            merge_ids = [int(x) for x in np.unique(arr).tolist() if int(x) != args.bg_id]
        changed += recursive_neighbor_merge(arr, merge_ids, args.bg_id, args.recursive_merge_max_iter)
    else:
        # First pass: remove tiny disconnected foreground blobs (mixed-class safe).
        changed += remove_small_foreground_blobs(arr, args.bg_id, args.min_fg_blob_area)

        # Second pass: remove tiny noisy islands from major classes.
        changed += remove_small_components(arr, args.hair_id, args.min_hair_area, args.bg_id)
        changed += remove_small_components(arr, args.skin_id, args.min_skin_area, args.bg_id)
        changed += remove_small_components(arr, args.cloth_id, args.min_cloth_area, args.bg_id)

        # Third pass (optional): keep only largest component for selected classes.
        if args.enable_keep_largest:
            for class_id in args.keep_largest_ids:
                changed += keep_largest_component(arr, class_id, args.bg_id)

        # Fourth pass (optional): recursively merge non-main components by neighbor contact.
        if args.enable_recursive_neighbor_merge:
            changed += recursive_neighbor_merge(
                arr,
                args.recursive_merge_ids,
                args.bg_id,
                args.recursive_merge_max_iter,
            )

        # Fill tiny holes inside major classes to smooth boundaries.
        changed += fill_small_holes(arr, args.hair_id, args.max_hole_area)
        changed += fill_small_holes(arr, args.skin_id, args.max_hole_area)
        changed += fill_small_holes(arr, args.cloth_id, args.max_hole_area)

    if args.inplace:
        out_path = mask_path

    if palette is not None:
        out_img = Image.fromarray(arr, mode="P")
        out_img.putpalette(palette)
    else:
        out_img = Image.fromarray(arr)
    out_img.save(out_path)

    pixel_changed = int((arr != before).sum())
    return changed, pixel_changed


def main():
    parser = argparse.ArgumentParser(description="Post-process segmentation masks to reduce speckle noise.")
    parser.add_argument("--input_dir", required=True, help="Directory containing predicted PNG masks.")
    parser.add_argument("--output_dir", default="", help="Output directory (default: <input_dir>_post).")
    parser.add_argument("--inplace", action="store_true", help="Overwrite masks in input_dir.")

    parser.add_argument("--bg_id", type=int, default=0)
    parser.add_argument("--skin_id", type=int, default=1)
    parser.add_argument("--hair_id", type=int, default=13)
    parser.add_argument("--cloth_id", type=int, default=18)

    parser.add_argument("--min_hair_area", type=int, default=400)
    parser.add_argument("--min_skin_area", type=int, default=250)
    parser.add_argument("--min_cloth_area", type=int, default=400)
    parser.add_argument("--min_fg_blob_area", type=int, default=180,
                        help="Remove disconnected non-background blobs smaller than this area.")
    parser.add_argument("--max_hole_area", type=int, default=140)
    parser.add_argument("--enable_keep_largest", action="store_true", default=False,
                        help="For selected classes, keep only the largest connected component.")
    parser.add_argument(
        "--keep_largest_ids",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,17",
        help="Comma-separated class ids used with --enable_keep_largest.",
    )
    parser.add_argument("--enable_recursive_neighbor_merge", action="store_true", default=False,
                        help="Recursively merge non-main components to neighbor-majority classes.")
    parser.add_argument(
        "--recursive_merge_ids",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,17",
        help="Comma-separated class ids for neighbor-majority recursive merge.",
    )
    parser.add_argument("--recursive_merge_max_iter", type=int, default=6,
                        help="Maximum recursive iterations for neighbor-majority merge.")
    parser.add_argument("--main_only_mode", action="store_true", default=False,
                        help="Strict mode: for each class keep largest component; merge others by neighbor-majority.")
    parser.add_argument(
        "--main_only_ids",
        type=str,
        default="",
        help="Optional class ids for --main_only_mode (default: all non-background ids in each mask).",
    )
    args = parser.parse_args()
    args.keep_largest_ids = parse_id_list(args.keep_largest_ids)
    args.recursive_merge_ids = parse_id_list(args.recursive_merge_ids)
    args.main_only_ids = parse_id_list(args.main_only_ids)

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {in_dir}")

    if args.inplace:
        out_dir = in_dir
    else:
        out_dir = Path(args.output_dir) if args.output_dir else Path(f"{args.input_dir}_post")
        out_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() == ".png"])
    if not mask_files:
        print(f"No PNG masks found in: {in_dir}")
        return

    total_region_changed = 0
    total_pixel_changed = 0
    changed_files = 0
    for p in mask_files:
        out_path = out_dir / p.name
        region_changed, pixel_changed = process_one(p, out_path, args)
        total_region_changed += region_changed
        total_pixel_changed += pixel_changed
        if pixel_changed > 0:
            changed_files += 1

    print(f"Processed: {len(mask_files)} files")
    print(f"Changed files: {changed_files}")
    print(f"Changed pixels (total): {total_pixel_changed}")
    print(f"Changed regions (total): {total_region_changed}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
