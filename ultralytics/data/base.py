# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob
import math
import os
import random
from collections import Counter
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM

from .utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(
        self,
        img_path,
        imgir_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        self.img_path = img_path
        self.imgir_path = imgir_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction

        # ------------------------------------------------------------------
        # Build RGB-IR pairs explicitly.
        #
        # Do NOT infer the IR path from the RGB string (e.g. images -> image)
        # and do NOT assume both modalities use the same file extension.
        #
        # Example supported pair:
        #   RGB: .../images/train/FLIR_00001.jpg
        #   IR : .../image/train/FLIR_00001.jpeg
        #
        # Pairing is based on the filename stem and the resulting IR list is
        # reordered to exactly follow the RGB list.
        # ------------------------------------------------------------------
        rgb_files = self.get_img_files(self.img_path, apply_fraction=False)
        ir_files = self.get_img_files(self.imgir_path, apply_fraction=False)
        self.im_files, self.imir_files = self._pair_modal_files(rgb_files, ir_files)

        # Apply fraction AFTER pairing so RGB and IR can never be sliced
        # independently into different sample sets.
        if self.fraction < 1:
            n = max(1, round(len(self.im_files) * self.fraction))
            self.im_files = self.im_files[:n]
            self.imir_files = self.imir_files[:n]

        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class

        self.ni = len(self.labels)  # number of images
        if self.ni != len(self.im_files):
            raise RuntimeError(
                f"{self.prefix}RGB/label count mismatch after pairing: "
                f"{len(self.im_files)} RGB pairs vs {self.ni} labels."
            )

        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache the COMBINED RGB-IR tensor.
        # Use a dedicated suffix to avoid accidentally reading legacy RGB-only
        # .npy files produced by older versions of this loader.
        self.ims = [None] * self.ni
        self.imsir = [None] * self.ni  # kept for backward compatibility
        self.im_hw0 = [None] * self.ni
        self.im_hw = [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".rgbir.npy") for f in self.im_files]
        self.npyir_files = [Path(f).with_suffix(".ir.npy") for f in self.imir_files]

        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if (self.cache == "ram" and self.check_cache_ram()) or self.cache == "disk":
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path, apply_fraction=True):
        """Read image files from a directory, a txt file, or a list of either."""
        try:
            files = []
            sources = img_path if isinstance(img_path, list) else [img_path]

            for source in sources:
                p = Path(source)

                if p.is_dir():
                    files += glob.glob(str(p / "**" / "*.*"), recursive=True)

                elif p.is_file():
                    with open(p, encoding="utf-8") as t:
                        entries = t.read().strip().splitlines()

                    parent = str(p.parent) + os.sep
                    files += [
                        x.replace("./", parent) if x.startswith("./") else x
                        for x in entries
                    ]

                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")

            im_files = sorted(
                os.path.normpath(x)
                for x in files
                if Path(x).suffix[1:].lower() in IMG_FORMATS
            )

            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"

        except Exception as e:
            raise FileNotFoundError(
                f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}"
            ) from e

        if apply_fraction and self.fraction < 1:
            im_files = im_files[: max(1, round(len(im_files) * self.fraction))]

        return im_files

    @staticmethod
    def _modal_pair_key(file_path):
        """
        Return the modality-independent sample key.

        File extensions are deliberately ignored:
            xxx.jpg, xxx.jpeg, xxx.PNG -> key 'xxx'

        casefold() also makes filename matching robust to case differences.
        """
        return Path(file_path).stem.casefold()

    def _pair_modal_files(self, rgb_files, ir_files):
        """
        Pair RGB and IR images by filename stem, independent of extension.

        The returned IR list is reordered to follow RGB order exactly.

        This intentionally fails loudly on:
          - duplicated stems within one modality;
          - an RGB image with no matching IR image.

        Silent index-based pairing is much more dangerous for multispectral
        training because one missing file can shift every subsequent pair.
        """
        def build_unique_map(files, modality):
            mapping = {}
            duplicates = {}

            for file_path in files:
                key = self._modal_pair_key(file_path)
                if key in mapping:
                    duplicates.setdefault(key, [mapping[key]]).append(file_path)
                else:
                    mapping[key] = file_path

            if duplicates:
                preview = "\n".join(
                    f"  {key}: {paths}"
                    for key, paths in list(duplicates.items())[:10]
                )
                raise RuntimeError(
                    f"{self.prefix}{modality} contains duplicated filename stems. "
                    f"Pairing would be ambiguous. First duplicates:\n{preview}"
                )

            return mapping

        rgb_map = build_unique_map(rgb_files, "RGB")
        ir_map = build_unique_map(ir_files, "IR")

        paired_rgb = []
        paired_ir = []
        missing_ir = []

        # Keep the original sorted RGB order because labels are generated from
        # self.im_files and must stay aligned with it.
        for rgb_file in rgb_files:
            key = self._modal_pair_key(rgb_file)
            ir_file = ir_map.get(key)

            if ir_file is None:
                missing_ir.append(rgb_file)
                continue

            paired_rgb.append(rgb_file)
            paired_ir.append(ir_file)

        if missing_ir:
            preview = "\n".join(f"  {x}" for x in missing_ir[:20])
            raise FileNotFoundError(
                f"{self.prefix}Found {len(missing_ir)} RGB images without a matching IR image. "
                f"Matching ignores extensions and uses filename stems. "
                f"First missing RGB files:\n{preview}"
            )

        # Extra IR files do not corrupt pairing, but report them because they
        # usually indicate an incomplete RGB side or a stale dataset file.
        rgb_keys = set(rgb_map)
        extra_ir_keys = set(ir_map) - rgb_keys
        if extra_ir_keys:
            examples = [ir_map[k] for k in sorted(extra_ir_keys)[:10]]
            LOGGER.warning(
                f"{self.prefix}WARNING ⚠️ {len(extra_ir_keys)} IR images have no RGB counterpart "
                f"and will be ignored. Examples: {examples}"
            )

        if not paired_rgb:
            raise RuntimeError(f"{self.prefix}No valid RGB-IR pairs were found.")

        rgb_ext = Counter(Path(x).suffix.lower() for x in paired_rgb)
        ir_ext = Counter(Path(x).suffix.lower() for x in paired_ir)

        LOGGER.info(
            f"{self.prefix}RGB-IR pairs: {len(paired_rgb)} | "
            f"RGB extensions: {dict(rgb_ext)} | IR extensions: {dict(ir_ext)}"
        )

        return paired_rgb, paired_ir

    def _read_rgb_ir_pair(self, i):
        """
        Read one aligned RGB-IR pair and concatenate to a 6-channel array.

        Shape mismatch is treated as a dataset error instead of silently
        resizing one modality, since an implicit resize can invalidate
        multispectral alignment.
        """
        rgb_file = self.im_files[i]
        ir_file = self.imir_files[i]

        rgb = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(
                f"{self.prefix}Cannot read RGB image: {rgb_file}"
            )

        ir = cv2.imread(ir_file, cv2.IMREAD_COLOR)
        if ir is None:
            raise FileNotFoundError(
                f"{self.prefix}Cannot read IR image: {ir_file}"
            )

        if rgb.shape[:2] != ir.shape[:2]:
            raise ValueError(
                f"{self.prefix}RGB/IR shape mismatch for paired sample:\n"
                f"  RGB: {rgb_file} -> {rgb.shape}\n"
                f"  IR : {ir_file} -> {ir.shape}\n"
                "Please align/resize the dataset before training. "
                "The loader will not silently resize one modality."
            )

        return np.concatenate((rgb, ir), axis=2)

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i, rect_mode=True):
        """Load one paired RGB-IR sample as a 6-channel image."""
        im = self.ims[i]
        fn = self.npy_files[i]

        if im is None:  # not cached in RAM
            if fn.exists():
                try:
                    im = np.load(fn)
                    # Guard against stale RGB-only caches from manual edits.
                    if im.ndim != 3 or im.shape[2] != 6:
                        raise ValueError(
                            f"expected 6-channel RGB-IR cache, got shape {im.shape}"
                        )
                except Exception as e:
                    LOGGER.warning(
                        f"{self.prefix}WARNING ⚠️ Removing invalid RGB-IR cache {fn} due to: {e}"
                    )
                    Path(fn).unlink(missing_ok=True)
                    im = self._read_rgb_ir_pair(i)
            else:
                im = self._read_rgb_ir_pair(i)

            h0, w0 = im.shape[:2]

            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = (
                        min(math.ceil(w0 * r), self.imgsz),
                        min(math.ceil(h0 * r), self.imgsz),
                    )
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)

            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(
                    im,
                    (self.imgsz, self.imgsz),
                    interpolation=cv2.INTER_LINEAR,
                )

            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = (
                    im,
                    (h0, w0),
                    im.shape[:2],
                )
                self.buffer.append(i)

                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def loadir_image(self, i, rect_mode=True):
        """
        Backward-compatible standalone IR loader.

        Main multispectral training should use load_image(), which loads the
        already paired RGB and IR files together.
        """
        im = self.imsir[i]
        f = self.imir_files[i]

        if im is None:
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(f"{self.prefix}Cannot read IR image: {f}")

            h0, w0 = im.shape[:2]

            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = (
                        min(math.ceil(w0 * r), self.imgsz),
                        min(math.ceil(h0 * r), self.imgsz),
                    )
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)

            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(
                    im,
                    (self.imgsz, self.imgsz),
                    interpolation=cv2.INTER_LINEAR,
                )

            return im, (h0, w0), im.shape[:2]

        return self.imsir[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self):
        """Cache paired 6-channel RGB-IR samples to RAM or disk."""
        b, gb = 0, 1 << 30
        fcn, storage = (
            (self.cache_images_to_disk, "Disk")
            if self.cache == "disk"
            else (self.load_image, "RAM")
        )

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(
                enumerate(results),
                total=self.ni,
                disable=LOCAL_RANK > 0,
            )

            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                    b += self.ims[i].nbytes

                pbar.desc = (
                    f"{self.prefix}Caching RGB-IR pairs "
                    f"({b / gb:.1f}GB {storage})"
                )

            pbar.close()

    def cache_images_to_disk(self, i):
        """Save the complete paired 6-channel sample as *.rgbir.npy."""
        f = self.npy_files[i]
        if not f.exists():
            im = self._read_rgb_ir_pair(i)
            np.save(f.as_posix(), im, allow_pickle=False)

    def cacheir_images(self):
        """Backward-compatible IR-only RAM cache."""
        b, gb = 0, 1 << 30

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(self.loadir_image, range(self.ni))
            pbar = TQDM(
                enumerate(results),
                total=self.ni,
                disable=LOCAL_RANK > 0,
            )

            for i, x in pbar:
                self.imsir[i], _, _ = x
                b += self.imsir[i].nbytes
                pbar.desc = f"{self.prefix}Caching IR images ({b / gb:.1f}GB RAM)"

            pbar.close()

    def cacheir_images_to_disk(self, i):
        """Backward-compatible standalone IR disk cache."""
        f = self.npyir_files[i]
        if not f.exists():
            im = cv2.imread(self.imir_files[i], cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(
                    f"{self.prefix}Cannot read IR image: {self.imir_files[i]}"
                )
            np.save(f.as_posix(), im, allow_pickle=False)

    def check_cache_ram(self, safety_margin=0.5):
        """Estimate RAM required for caching paired 6-channel RGB-IR images."""
        b, gb = 0, 1 << 30
        n = min(self.ni, 30)

        if n == 0:
            return False

        for i in random.sample(range(self.ni), n):
            im = self._read_rgb_ir_pair(i)
            ratio = self.imgsz / max(im.shape[0], im.shape[1])
            b += im.nbytes * ratio**2

        mem_required = b * self.ni / n * (1 + safety_margin)
        mem = psutil.virtual_memory()
        success = mem_required < mem.available

        if not success:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache RGB-IR pairs "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, "
                "not caching images ⚠️"
            )

        return success

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.imir_files = [self.imir_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.

        Example:
            ```python
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
            ```
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        raise NotImplementedError
    def get_irlabels(self):
            """
            Users can customize their own format here.

            Note:
                Ensure output is a dictionary with the following keys:
                ```python
                dict(
                    im_file=im_file,
                    shape=shape,  # format: (height, width)
                    cls=cls,
                    bboxes=bboxes, # xywh
                    segments=segments,  # xy
                    keypoints=keypoints, # xy
                    normalized=True, # or False
                    bbox_format="xyxy",  # or xywh, ltwh
                )
                ```
            """
            raise NotImplementedError
