import os
import sys
import warnings
from itertools import chain
from pathlib import Path

import mrcfile
import numpy as np
from csbdeep.data import RawData
from csbdeep.data.generate import (
    _memory_check,
    no_background_patches,
    norm_percentiles,
    sample_patches_from_multiple_stacks,
    save_training_data,
    shuffle_inplace,
)
from csbdeep.data.transform import Transform, broadcast_target, permute_axes
from csbdeep.utils import (
    _raise,
    axes_check_and_normalize,
    axes_dict,
    compose,
    consume,
)
from scipy.ndimage import zoom
from tifffile import imread, imsave
from tqdm import tqdm


class RawDataScaled(RawData):
    @staticmethod
    def from_folder(
        basepath,
        source_dirs,
        target_dir,
        scale_gt=2.0,
        axes="CZYX",
        pattern="*.tif*",
    ):
        """Get pairs of corresponding TIFF images read from folders.

        Two images correspond to each other if they have the same file name, but are located in different folders.

        Parameters
        ----------
        basepath : str
            Base folder that contains sub-folders with images.
        source_dirs : list or tuple
            List of folder names relative to `basepath` that contain the source images (e.g., with low SNR).
        target_dir : str
            Folder name relative to `basepath` that contains the target images (e.g., with high SNR).
        axes : str
            Semantics of axes of loaded images (assumed to be the same for all images).
        pattern : str
            Glob-style pattern to match the desired TIFF images.

        Returns
        -------
        RawData
            :obj:`RawData` object, whose `generator` is used to yield all matching TIFF pairs.
            The generator will return a tuple `(x,y,axes,mask)`, where `x` is from
            `source_dirs` and `y` is the corresponding image from the `target_dir`;
            `mask` is set to `None`.

        Raises
        ------
        FileNotFoundError
            If an image found in a `source_dir` does not exist in `target_dir`.

        Example
        --------
        >>> !tree data
        data
        ├── GT
        │   ├── imageA.tif
        │   ├── imageB.tif
        │   └── imageC.tif
        ├── source1
        │   ├── imageA.tif
        │   └── imageB.tif
        └── source2
            ├── imageA.tif
            └── imageC.tif

        >>> data = RawData.from_folder(basepath='data', source_dirs=['source1','source2'], target_dir='GT', axes='YX')
        >>> n_images = data.size
        >>> for source_x, target_y, axes, mask in data.generator():
        ...     pass

        """
        p = Path(basepath)
        pairs = [
            (f, p / target_dir / f.name)
            for f in chain(
                *((p / source_dir).glob(pattern) for source_dir in source_dirs)
            )
        ]
        len(pairs) > 0 or _raise(FileNotFoundError("Didn't find any images."))
        consume(t.exists() or _raise(FileNotFoundError(t)) for s, t in pairs)
        axes = axes_check_and_normalize(axes)
        n_images = len(pairs)
        description = "{p}: target='{o}', sources={s}, axes='{a}', pattern='{pt}'".format(
            p=basepath, s=list(source_dirs), o=target_dir, a=axes, pt=pattern
        )

        def _gen():
            for fx, fy in pairs:
                x, y = imread(str(fx)), imread(str(fy))
                len(axes) >= x.ndim or _raise(ValueError())
                x = scale_image_along_axes(x, scale_gt)
                yield x, y, axes[-x.ndim :], None

        return RawData(_gen, n_images, description)


def scale_image_along_axes(image, scale_factor):
    scaled_image = zoom(image, zoom=(1, scale_factor, scale_factor))
    return scaled_image


def create_folders_DN(
    root_dir,
    gt_filename="RawSIMData_level_09",
    raw_filename="RawSIMData_level_01",
    filename_extension=".mrc",
):
    gt_dir = os.path.join(root_dir, "Train", "DN", "GT")
    raw_dir = os.path.join(root_dir, "Train", "DN", "Raw")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    for idx, subdir in enumerate(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            gt_files = [
                f
                for f in os.listdir(subdir_path)
                if f.endswith(gt_filename + filename_extension)
            ]
            for gt_file in gt_files:
                raw_gt_file = os.path.join(
                    subdir_path, raw_filename + filename_extension
                )
                if os.path.exists(raw_gt_file):
                    input_gt_path = os.path.join(subdir_path, gt_file)
                    output_gt_path = os.path.join(gt_dir, f"{idx + 1}.tif")
                    convert_mrc_to_tiff(input_gt_path, output_gt_path)

                    input_raw_gt_path = raw_gt_file
                    output_raw_gt_path = os.path.join(
                        raw_dir, f"{idx + 1}.tif"
                    )
                    convert_mrc_to_tiff(input_raw_gt_path, output_raw_gt_path)
                else:
                    print(
                        f"Warning: {raw_gt_file} file does not exist for {gt_file}. Skipping..."
                    )


def create_folders_SR(
    root_dir,
    gt_filename="SIM_gt",
    raw_filename="RawSIMData_gt",
    filename_extension=".mrc",
):
    gt_dir = os.path.join(root_dir, "Train", "SR", "GT")
    raw_dir = os.path.join(root_dir, "Train", "SR", "Raw")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    for idx, subdir in enumerate(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            gt_files = [
                f
                for f in os.listdir(subdir_path)
                if f.endswith(gt_filename + filename_extension)
            ]
            for gt_file in gt_files:
                raw_gt_file = os.path.join(
                    subdir_path, raw_filename + filename_extension
                )
                if os.path.exists(raw_gt_file):
                    input_gt_path = os.path.join(subdir_path, gt_file)
                    output_gt_path = os.path.join(gt_dir, f"{idx + 1}.tif")
                    convert_mrc_to_tiff(input_gt_path, output_gt_path)

                    input_raw_gt_path = raw_gt_file
                    output_raw_gt_path = os.path.join(
                        raw_dir, f"{idx + 1}.tif"
                    )
                    convert_mrc_to_tiff(input_raw_gt_path, output_raw_gt_path)
                else:
                    print(
                        f"Warning: {raw_gt_file} file does not exist for {gt_file}. Skipping..."
                    )


def convert_mrc_to_tiff(input_path, output_path):
    with mrcfile.open(input_path, permissive=True) as mrc:
        data = mrc.data.squeeze()
        imsave(output_path, data.astype(np.float32))


def create_patches(
    raw_data,
    patch_size,
    n_patches_per_image,
    patch_axes=None,
    save_file=None,
    transforms=None,
    patch_filter=no_background_patches(),
    normalization=norm_percentiles(),
    scale_gt=1.0,
    shuffle=True,
    verbose=True,
):
    """Create normalized training data to be used for neural network training.

    Parameters
    ----------
    raw_data : :class:`RawData`
        Object that yields matching pairs of raw images.
    patch_size : tuple
        Shape of the patches to be extraced from raw images.
        Must be compatible with the number of dimensions and axes of the raw images.
        As a general rule, use a power of two along all XYZT axes, or at least divisible by 8.
    n_patches_per_image : int
        Number of patches to be sampled/extracted from each raw image pair (after transformations, see below).
    patch_axes : str or None
        Axes of the extracted patches. If ``None``, will assume to be equal to that of transformed raw data.
    save_file : str or None
        File name to save training data to disk in ``.npz`` format (see :func:`csbdeep.io.save_training_data`).
        If ``None``, data will not be saved.
    transforms : list or tuple, optional
        List of :class:`Transform` objects that apply additional transformations to the raw images.
        This can be used to augment the set of raw images (e.g., by including rotations).
        Set to ``None`` to disable. Default: ``None``.
    patch_filter : function, optional
        Function to determine for each image pair which patches are eligible to be extracted
        (default: :func:`no_background_patches`). Set to ``None`` to disable.
    normalization : function, optional
        Function that takes arguments `(patches_x, patches_y, x, y, mask, channel)`, whose purpose is to
        normalize the patches (`patches_x`, `patches_y`) extracted from the associated raw images
        (`x`, `y`, with `mask`; see :class:`RawData`). Default: :func:`norm_percentiles`.
    shuffle : bool, optional
        Randomly shuffle all extracted patches.
    verbose : bool, optional
        Display overview of images, transforms, etc.

    Returns
    -------
    tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`, str)
        Returns a tuple (`X`, `Y`, `axes`) with the normalized extracted patches from all (transformed) raw images
        and their axes.
        `X` is the array of patches extracted from source images with `Y` being the array of corresponding target patches.
        The shape of `X` and `Y` is as follows: `(n_total_patches, n_channels, ...)`.
        For single-channel images, `n_channels` will be 1.

    Raises
    ------
    ValueError
        Various reasons.

    Example
    -------
    >>> raw_data = RawData.from_folder(basepath='data', source_dirs=['source1','source2'], target_dir='GT', axes='ZYX')
    >>> X, Y, XY_axes = create_patches(raw_data, patch_size=(32,128,128), n_patches_per_image=16)

    Todo
    ----
    - Save created patches directly to disk using :class:`numpy.memmap` or similar?
      Would allow to work with large data that doesn't fit in memory.

    """
    if transforms is None:
        transforms = []
    transforms = list(transforms)
    if patch_axes is not None:
        transforms.append(permute_axes(patch_axes))
    if len(transforms) == 0:
        transforms.append(Transform.identity())

    if normalization is None:
        normalization = lambda patches_x, patches_y, x, y, mask, channel: (
            patches_x,
            patches_y,
        )

    image_pairs, n_raw_images = raw_data.generator(), raw_data.size
    tf = Transform(*zip(*transforms))
    image_pairs = compose(*tf.generator)(image_pairs)
    n_transforms = np.prod(tf.size)
    n_images = n_raw_images * n_transforms
    n_patches = n_images * n_patches_per_image
    n_required_memory_bytes = 2 * n_patches * np.prod(patch_size) * 4

    _memory_check(n_required_memory_bytes)

    if verbose:
        print("=" * 66)
        print(
            "%5d raw images x %4d transformations   = %5d images"
            % (n_raw_images, n_transforms, n_images)
        )
        print(
            "%5d images     x %4d patches per image = %5d patches in total"
            % (n_images, n_patches_per_image, n_patches)
        )
        print("=" * 66)
        print("Input data:")
        print(raw_data.description)
        print("=" * 66)
        print("Transformations:")
        for t in transforms:
            print("{t.size} x {t.name}".format(t=t))
        print("=" * 66)

        scaled_patch_size = tuple(
            int(s / scale_gt) if i > 0 else s for i, s in enumerate(patch_size)
        )
        print(" x ".join(str(p) for p in scaled_patch_size[1:]))
        print("Patch size:")
        print(" x ".join(str(p) for p in scaled_patch_size))
        print("=" * 66)

    sys.stdout.flush()

    X = np.empty((n_patches,) + tuple(patch_size), dtype=np.float32)
    new_patch_size = tuple(int(s / scale_gt) for s in patch_size[1:])
    Z = np.empty(
        (n_patches,) + (patch_size[0],) + new_patch_size, dtype=np.float32
    )

    Y = np.empty_like(X)
    for i, (x, y, _axes, mask) in tqdm(
        enumerate(image_pairs), total=n_images, disable=(not verbose)
    ):
        if i >= n_images:
            warnings.warn(
                "more raw images (or transformations thereof) than expected, skipping excess images."
            )
            break
        if i == 0:
            axes = axes_check_and_normalize(_axes, len(patch_size))
            channel = axes_dict(axes)["C"]
        axes == axes_check_and_normalize(_axes) or _raise(
            ValueError("not all images have the same axes.")
        )

        x.shape == y.shape or _raise(ValueError())
        mask is None or mask.shape == x.shape or _raise(ValueError())
        (
            channel is None
            or (isinstance(channel, int) and 0 <= channel < x.ndim)
        ) or _raise(ValueError())
        channel is None or patch_size[channel] == x.shape[channel] or _raise(
            ValueError("extracted patches must contain all channels.")
        )

        _Y, _X = sample_patches_from_multiple_stacks(
            (y, x), patch_size, n_patches_per_image, mask, patch_filter
        )

        s = slice(i * n_patches_per_image, (i + 1) * n_patches_per_image)
        X[s], Y[s] = normalization(_X, _Y, x, y, mask, channel)
        Z[s] = sample_smaller_patches_from_raw(X[s], scale_gt)

    if shuffle:
        shuffle_inplace(Z, Y)

    axes = "SC" + axes.replace("C", "")
    if channel is None:
        Z = np.expand_dims(Z, 1)
        Y = np.expand_dims(Y, 1)
    else:
        Z = np.moveaxis(Z, 1 + channel, 1)
        Y = np.moveaxis(Y, 1 + channel, 1)

    if save_file is not None:
        print("Saving data to %s." % str(Path(save_file)))
        save_training_data(save_file, Z, Y, axes)

    return Z, Y, axes


def sample_smaller_patches_from_raw(raw_patches, scale_gt):
    """Sample smaller patches from raw image to match the scale factor."""
    smaller_raw_patches = zoom(
        raw_patches, zoom=(1, 1, 1.0 / scale_gt, 1.0 / scale_gt)
    )

    return smaller_raw_patches


def create_patches_reduced_target(
    raw_data,
    patch_size,
    n_patches_per_image,
    reduction_axes,
    scale_gt=1.0,
    target_axes=None,
    **kwargs,
):
    """Create normalized training data to be used for neural network training.

    In contrast to :func:`create_patches`, it is assumed that the target image has reduced
    dimensionality (i.e. size 1) along one or several axes (`reduction_axes`).

    Parameters
    ----------
    raw_data : :class:`RawData`
        See :func:`create_patches`.
    patch_size : tuple
        See :func:`create_patches`.
    n_patches_per_image : int
        See :func:`create_patches`.
    reduction_axes : str
        Axes where the target images have a reduced dimension (i.e. size 1) compared to the source images.
    target_axes : str
        Axes of the raw target images. If ``None``, will be assumed to be equal to that of the raw source images.
    kwargs : dict
        Additional parameters as in :func:`create_patches`.

    Returns
    -------
    tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`, str)
        See :func:`create_patches`. Note that the shape of the target data will be 1 along all reduction axes.

    """
    reduction_axes = axes_check_and_normalize(reduction_axes, disallowed="S")

    transforms = kwargs.get("transforms")
    if transforms is None:
        transforms = []
    transforms = list(transforms)
    transforms.insert(0, broadcast_target(target_axes))
    kwargs["transforms"] = transforms

    save_file = kwargs.pop("save_file", None)

    if any(s is None for s in patch_size):
        patch_axes = kwargs.get("patch_axes")
        if patch_axes is not None:
            _transforms = list(transforms)
            _transforms.append(permute_axes(patch_axes))
        else:
            _transforms = transforms
        tf = Transform(*zip(*_transforms))
        image_pairs = compose(*tf.generator)(raw_data.generator())
        x, y, axes, mask = next(image_pairs)
        patch_size = list(patch_size)
        for i, (a, s) in enumerate(zip(axes, patch_size)):
            if s is not None:
                continue
            a in reduction_axes or _raise(
                ValueError(
                    "entry of patch_size is None for non reduction axis %s."
                    % a
                )
            )
            patch_size[i] = x.shape[i]
        patch_size = tuple(patch_size)
        del x, y, axes, mask

    X, Y, axes = create_patches(
        raw_data=raw_data,
        patch_size=patch_size,
        scale_gt=scale_gt,
        n_patches_per_image=n_patches_per_image,
        **kwargs,
    )

    ax = axes_dict(axes)
    for a in reduction_axes:
        a in axes or _raise(
            ValueError(
                "reduction axis %d not present in extracted patches" % a
            )
        )
        n_dims = Y.shape[ax[a]]
        if n_dims == 1:
            warnings.warn(
                "extracted target patches already have dimensionality 1 along reduction axis %s."
                % a
            )
        else:
            t = np.take(Y, (1,), axis=ax[a])
            Y = np.take(Y, (0,), axis=ax[a])
            i = np.random.choice(Y.size, size=100)
            if not np.all(t.flat[i] == Y.flat[i]):
                warnings.warn(
                    "extracted target patches vary along reduction axis %s."
                    % a
                )

    if save_file is not None:
        print("Saving data to %s." % str(Path(save_file)))
        save_training_data(save_file, X, Y, axes)

    return X, Y, axes