"""Tests for WashingtonRGBDDataset and get_washington_dataloaders."""

import json
import os

import numpy as np
import pytest
import torch

from src.data_utils.washington_dataset import (
    WashingtonRGBDDataset,
    _discover_samples,
    _load_class_names,
    _load_norm_stats,
    get_washington_dataloaders,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _create_split_samples(root, class_names, split, objs_per_class, frames_per_obj,
                           depth_low=0, depth_high=10000, zero_sentinel=True):
    """Helper to create .pt sample files in root/<split>/<class>/.

    Uses int16 depth (matching Washington preprocessing output).
    """
    split_dir = root / split
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = split_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for obj_idx in range(objs_per_class):
            for frame_idx in range(frames_per_obj):
                rgb = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
                depth = torch.randint(
                    depth_low, depth_high, (1, 256, 256), dtype=torch.int16
                )
                if zero_sentinel:
                    depth[0, :5, :5] = 0
                stem = f'{cls_name}_{obj_idx + 1}_{frame_idx + 1}_f{frame_idx:03d}'
                torch.save(rgb, cls_dir / f'{stem}_rgb.pt')
                torch.save(depth, cls_dir / f'{stem}_depth.pt')


def _make_dataset(fake_dataset, split='train', **kwargs):
    """Module-level helper to construct dataset from fake_dataset fixture."""
    data_root, class_names, norm_stats = fake_dataset
    split_dir = os.path.join(str(data_root), split)
    samples = _discover_samples(split_dir, class_names)
    return WashingtonRGBDDataset(
        data_root=str(data_root),
        split=split,
        samples=samples,
        class_names=class_names,
        norm_stats=norm_stats,
        **kwargs,
    )


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def fake_dataset(tmp_path):
    """Create a minimal fake Washington RGB-D dataset with train/val/test splits.

    Creates 3 classes:
      - train: 4 objects x 2 frames = 8 samples per class (24 total)
      - val:   1 object  x 2 frames = 2 samples per class (6 total)
      - test:  1 object  x 2 frames = 2 samples per class (6 total)
    Includes some depth pixels set to 0 (missing data sentinel).
    """
    class_names = ['apple', 'bowl', 'camera']

    with open(tmp_path / 'class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    norm_stats = {
        'rgb_mean': [0.485, 0.456, 0.406],
        'rgb_std': [0.229, 0.224, 0.225],
        'depth_mean': [0.85],
        'depth_std': [0.25],
    }
    with open(tmp_path / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f)

    _create_split_samples(tmp_path, class_names, 'train',
                           objs_per_class=4, frames_per_obj=2)
    _create_split_samples(tmp_path, class_names, 'val',
                           objs_per_class=1, frames_per_obj=2)
    _create_split_samples(tmp_path, class_names, 'test',
                           objs_per_class=1, frames_per_obj=2)

    return tmp_path, class_names, norm_stats


@pytest.fixture
def fake_dataset_scalar_depth_stats(tmp_path):
    """Dataset where norm_stats.json has scalar depth_mean/depth_std (not lists).

    This matches what our preprocessing notebook writes (float, not list).
    """
    class_names = ['apple', 'bowl']

    with open(tmp_path / 'class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    norm_stats = {
        'rgb_mean': [0.485, 0.456, 0.406],
        'rgb_std': [0.229, 0.224, 0.225],
        'depth_mean': 0.85,
        'depth_std': 0.25,
    }
    with open(tmp_path / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f)

    _create_split_samples(tmp_path, class_names, 'train',
                           objs_per_class=2, frames_per_obj=2)
    _create_split_samples(tmp_path, class_names, 'val',
                           objs_per_class=1, frames_per_obj=2)

    return tmp_path, class_names, norm_stats


@pytest.fixture
def fake_dataset_indexed_classnames(tmp_path):
    """Like fake_dataset but with '0: apple' format in class_names.txt."""
    class_names = ['apple', 'bowl', 'camera']
    with open(tmp_path / 'class_names.txt', 'w') as f:
        for i, name in enumerate(class_names):
            f.write(f"{i}: {name}\n")

    norm_stats = {
        'rgb_mean': [0.485, 0.456, 0.406],
        'rgb_std': [0.229, 0.224, 0.225],
        'depth_mean': [0.85],
        'depth_std': [0.25],
    }
    with open(tmp_path / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f)

    _create_split_samples(tmp_path, class_names, 'train',
                           objs_per_class=2, frames_per_obj=2,
                           depth_low=100, depth_high=5000, zero_sentinel=False)
    _create_split_samples(tmp_path, class_names, 'val',
                           objs_per_class=1, frames_per_obj=2,
                           depth_low=100, depth_high=5000, zero_sentinel=False)

    return tmp_path, class_names


# ── Tests: _load_class_names ─────────────────────────────────────────────


class TestLoadClassNames:
    """Tests for _load_class_names helper."""

    def test_plain_format(self, fake_dataset):
        """Plain 'apple' format lines are parsed correctly."""
        data_root, expected_names, _ = fake_dataset
        names = _load_class_names(str(data_root))
        assert names == expected_names

    def test_indexed_format(self, fake_dataset_indexed_classnames):
        """'0: apple' format lines are parsed correctly."""
        data_root, expected_names = fake_dataset_indexed_classnames
        names = _load_class_names(str(data_root))
        assert names == expected_names

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError raised when class_names.txt missing."""
        with pytest.raises(FileNotFoundError, match="class_names.txt"):
            _load_class_names(str(tmp_path))

    def test_empty_lines_skipped(self, tmp_path):
        """Empty lines in class_names.txt are skipped."""
        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("apple\n\nbowl\n\n")
        names = _load_class_names(str(tmp_path))
        assert names == ['apple', 'bowl']


# ── Tests: _load_norm_stats ──────────────────────────────────────────────


class TestLoadNormStats:
    """Tests for _load_norm_stats helper."""

    def test_loads_correctly(self, fake_dataset):
        """Stats loaded match what was written (list depth format)."""
        data_root, _, expected_stats = fake_dataset
        stats = _load_norm_stats(str(data_root))
        assert stats == expected_stats

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError raised when norm_stats.json missing."""
        with pytest.raises(FileNotFoundError, match="norm_stats.json"):
            _load_norm_stats(str(tmp_path))

    def test_value_types(self, fake_dataset):
        """Verify norm_stats values are lists of floats with correct lengths."""
        data_root, _, _ = fake_dataset
        stats = _load_norm_stats(str(data_root))
        assert isinstance(stats['rgb_mean'], list) and len(stats['rgb_mean']) == 3
        assert isinstance(stats['rgb_std'], list) and len(stats['rgb_std']) == 3
        assert isinstance(stats['depth_mean'], list) and len(stats['depth_mean']) == 1
        assert isinstance(stats['depth_std'], list) and len(stats['depth_std']) == 1
        for key in ['rgb_mean', 'rgb_std', 'depth_mean', 'depth_std']:
            assert all(isinstance(v, float) for v in stats[key])

    def test_scalar_depth_stats_wrapped_as_list(self, fake_dataset_scalar_depth_stats):
        """Scalar depth_mean/depth_std in JSON are wrapped into single-element lists."""
        data_root, _, _ = fake_dataset_scalar_depth_stats
        stats = _load_norm_stats(str(data_root))
        assert isinstance(stats['depth_mean'], list)
        assert isinstance(stats['depth_std'], list)
        assert len(stats['depth_mean']) == 1
        assert len(stats['depth_std']) == 1
        assert stats['depth_mean'][0] == pytest.approx(0.85)
        assert stats['depth_std'][0] == pytest.approx(0.25)


# ── Tests: _discover_samples ────────────────────────────────────────────


class TestDiscoverSamples:
    """Tests for _discover_samples helper."""

    def test_discovers_all_paired_train_samples(self, fake_dataset):
        """All 24 train samples (3 classes x 4 objs x 2 frames) discovered."""
        data_root, class_names, _ = fake_dataset
        samples = _discover_samples(str(data_root / 'train'), class_names)
        assert len(samples) == 24
        labels = [s[2] for s in samples]
        assert labels.count(0) == 8
        assert labels.count(1) == 8
        assert labels.count(2) == 8

    def test_discovers_val_samples(self, fake_dataset):
        """Val samples discovered from val/ subdirectory."""
        data_root, class_names, _ = fake_dataset
        samples = _discover_samples(str(data_root / 'val'), class_names)
        assert len(samples) == 6

    def test_discovers_test_samples(self, fake_dataset):
        """Test samples discovered from test/ subdirectory."""
        data_root, class_names, _ = fake_dataset
        samples = _discover_samples(str(data_root / 'test'), class_names)
        assert len(samples) == 6

    def test_each_sample_has_valid_paths(self, fake_dataset):
        """Each (rgb_path, depth_path, label) has existing files."""
        data_root, class_names, _ = fake_dataset
        for split in ['train', 'val', 'test']:
            samples = _discover_samples(str(data_root / split), class_names)
            for rgb_path, depth_path, label in samples:
                assert os.path.exists(rgb_path)
                assert os.path.exists(depth_path)
                assert 0 <= label < len(class_names)

    def test_unpaired_rgb_raises(self, fake_dataset):
        """ValueError raised when RGB file has no matching depth."""
        data_root, class_names, _ = fake_dataset
        orphan = data_root / 'train' / 'apple' / 'orphan_f000_rgb.pt'
        torch.save(torch.zeros(3, 256, 256, dtype=torch.uint8), orphan)
        with pytest.raises(ValueError, match="Unpaired files"):
            _discover_samples(str(data_root / 'train'), class_names)

    def test_unpaired_depth_raises(self, fake_dataset):
        """ValueError raised when depth file has no matching RGB."""
        data_root, class_names, _ = fake_dataset
        orphan = data_root / 'train' / 'apple' / 'orphan_f000_depth.pt'
        torch.save(torch.zeros(1, 256, 256, dtype=torch.int16), orphan)
        with pytest.raises(ValueError, match="Unpaired files"):
            _discover_samples(str(data_root / 'train'), class_names)

    def test_unknown_folder_skipped(self, fake_dataset):
        """Folders not in class_names.txt are skipped silently."""
        data_root, class_names, _ = fake_dataset
        extra = data_root / 'train' / 'unknown_class'
        extra.mkdir()
        torch.save(torch.zeros(3, 256, 256, dtype=torch.uint8), extra / 'x_f000_rgb.pt')
        torch.save(torch.zeros(1, 256, 256, dtype=torch.int16), extra / 'x_f000_depth.pt')
        samples = _discover_samples(str(data_root / 'train'), class_names)
        assert len(samples) == 24

    def test_deterministic_ordering(self, fake_dataset):
        """Two calls return identical ordering."""
        data_root, class_names, _ = fake_dataset
        train_dir = str(data_root / 'train')
        s1 = _discover_samples(train_dir, class_names)
        s2 = _discover_samples(train_dir, class_names)
        assert s1 == s2


# ── Tests: WashingtonRGBDDataset ─────────────────────────────────────────


class TestWashingtonRGBDDataset:
    """Tests for WashingtonRGBDDataset class."""

    def test_init_train(self, fake_dataset):
        """Train dataset initializes with correct counts."""
        ds = _make_dataset(fake_dataset, split='train')
        assert len(ds) == 24
        assert ds.num_classes == 3
        assert ds.split == 'train'

    def test_init_val(self, fake_dataset):
        """Val dataset initializes correctly."""
        ds = _make_dataset(fake_dataset, split='val')
        assert ds.split == 'val'
        assert len(ds) == 6

    def test_init_test(self, fake_dataset):
        """Test dataset initializes correctly."""
        ds = _make_dataset(fake_dataset, split='test')
        assert ds.split == 'test'
        assert len(ds) == 6

    def test_invalid_split_raises(self, fake_dataset):
        """ValueError on invalid split name."""
        data_root, class_names, norm_stats = fake_dataset
        samples = _discover_samples(str(data_root / 'train'), class_names)
        with pytest.raises(ValueError, match="split must be one of"):
            WashingtonRGBDDataset(
                data_root=str(data_root),
                split='holdout',
                samples=samples,
                class_names=class_names,
                norm_stats=norm_stats,
            )

    def test_getitem_shapes_train(self, fake_dataset):
        """__getitem__ returns correct shapes and types for train."""
        ds = _make_dataset(fake_dataset, split='train', crop_size=224)
        rgb, depth, label = ds[0]
        assert rgb.shape == (3, 224, 224)
        assert depth.shape == (1, 224, 224)
        assert rgb.dtype == torch.float32
        assert depth.dtype == torch.float32
        assert isinstance(label, int)
        assert 0 <= label < 3

    def test_getitem_shapes_val(self, fake_dataset):
        """__getitem__ returns correct shapes for val (CenterCrop)."""
        ds = _make_dataset(fake_dataset, split='val', crop_size=224)
        rgb, depth, label = ds[0]
        assert rgb.shape == (3, 224, 224)
        assert depth.shape == (1, 224, 224)
        assert rgb.dtype == torch.float32
        assert depth.dtype == torch.float32

    def test_getitem_shapes_test(self, fake_dataset):
        """__getitem__ returns correct shapes for test (CenterCrop)."""
        ds = _make_dataset(fake_dataset, split='test', crop_size=224)
        rgb, depth, label = ds[0]
        assert rgb.shape == (3, 224, 224)
        assert depth.shape == (1, 224, 224)
        assert rgb.dtype == torch.float32
        assert depth.dtype == torch.float32

    def test_depth_conversion_to_meters(self, fake_dataset):
        """Depth int16 mm is converted to float32 meters."""
        ds = _make_dataset(fake_dataset, split='val', normalize=False)
        rgb, depth, label = ds[0]
        # Original is randint(0, 10000) mm = 0-10 meters
        assert depth.max() > 0.5, (
            f"depth.max()={depth.max():.4f} — likely wrong conversion"
        )
        assert depth.max() <= 10.0

    def test_depth_mm_conversion_numeric(self, tmp_path):
        """Verify specific mm value converts to correct meters value."""
        class_names = ['apple']
        depth_mean = 0.85
        norm_stats = {
            'rgb_mean': [0.5, 0.5, 0.5],
            'rgb_std': [0.2, 0.2, 0.2],
            'depth_mean': [depth_mean],
            'depth_std': [0.25],
        }

        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("apple\n")
        with open(tmp_path / 'norm_stats.json', 'w') as f:
            json.dump(norm_stats, f)

        cls_dir = tmp_path / 'train' / 'apple'
        cls_dir.mkdir(parents=True)
        rgb = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        depth = torch.full((1, 256, 256), 3000, dtype=torch.int16)
        torch.save(rgb, cls_dir / 'apple_1_1_f000_rgb.pt')
        torch.save(depth, cls_dir / 'apple_1_1_f000_depth.pt')

        samples = _discover_samples(str(tmp_path / 'train'), class_names)
        ds = WashingtonRGBDDataset(
            data_root=str(tmp_path),
            split='val',
            samples=samples,
            class_names=class_names,
            norm_stats=norm_stats,
            normalize=False,
        )
        _, out_depth, _ = ds[0]
        # CenterCrop 256->224: pixel (128,128) -> (112,112)
        assert out_depth[0, 112, 112].item() == pytest.approx(3.0)

    def test_sentinel_replaced_with_mean(self, tmp_path):
        """0-sentinel pixels are replaced with depth_mean."""
        class_names = ['apple']
        depth_mean = 0.85
        norm_stats = {
            'rgb_mean': [0.5, 0.5, 0.5],
            'rgb_std': [0.2, 0.2, 0.2],
            'depth_mean': [depth_mean],
            'depth_std': [0.25],
        }

        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("apple\n")
        with open(tmp_path / 'norm_stats.json', 'w') as f:
            json.dump(norm_stats, f)

        cls_dir = tmp_path / 'train' / 'apple'
        cls_dir.mkdir(parents=True)
        rgb = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        depth = torch.full((1, 256, 256), 5000, dtype=torch.int16)
        depth[0, 128, 128] = 0
        torch.save(rgb, cls_dir / 'apple_1_1_f000_rgb.pt')
        torch.save(depth, cls_dir / 'apple_1_1_f000_depth.pt')

        samples = _discover_samples(str(tmp_path / 'train'), class_names)
        ds = WashingtonRGBDDataset(
            data_root=str(tmp_path),
            split='val',
            samples=samples,
            class_names=class_names,
            norm_stats=norm_stats,
            normalize=False,
        )
        _, out_depth, _ = ds[0]
        # CenterCrop 256->224: (128,128) -> (112,112)
        assert out_depth[0, 112, 112] == pytest.approx(depth_mean)

    def test_normalized_output_range(self, fake_dataset):
        """With normalize=True, output is standardized (not in [0,1])."""
        ds = _make_dataset(fake_dataset, split='val', normalize=True)
        rgb, depth, label = ds[0]
        assert rgb.min() < 0.0 or rgb.max() > 1.0

    def test_unnormalized_rgb_range(self, fake_dataset):
        """With normalize=False, RGB is in [0,1] range."""
        ds = _make_dataset(fake_dataset, split='val', normalize=False)
        rgb, depth, label = ds[0]
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_labels_are_list_of_int(self, fake_dataset):
        """self.labels is list[int]."""
        ds = _make_dataset(fake_dataset, split='train')
        assert isinstance(ds.labels, list)
        assert all(isinstance(l, int) for l in ds.labels)

    def test_get_class_weights_shape(self, fake_dataset):
        """get_class_weights returns [num_classes] float32 tensor."""
        ds = _make_dataset(fake_dataset, split='train')
        weights = ds.get_class_weights()
        assert weights.shape == (3,)
        assert weights.dtype == torch.float32
        assert torch.allclose(weights, torch.ones(3), atol=0.01)

    def test_get_sample_weights_shape(self, fake_dataset):
        """get_sample_weights returns [num_samples] float64 tensor."""
        ds = _make_dataset(fake_dataset, split='train')
        weights = ds.get_sample_weights()
        assert weights.shape == (24,)
        assert weights.dtype == torch.float64
        assert (weights > 0).all()

    def test_get_class_distribution(self, fake_dataset):
        """get_class_distribution returns correct counts."""
        ds = _make_dataset(fake_dataset, split='train')
        dist = ds.get_class_distribution()
        assert set(dist.keys()) == {'apple', 'bowl', 'camera'}
        for name in ['apple', 'bowl', 'camera']:
            assert dist[name]['count'] == 8
            assert abs(dist[name]['percentage'] - 100.0 / 3) < 0.1

    def test_get_norm_stats(self, fake_dataset):
        """get_norm_stats returns the loaded dict."""
        data_root, _, norm_stats = fake_dataset
        ds = _make_dataset(fake_dataset, split='train')
        assert ds.get_norm_stats() == norm_stats

    def test_zero_mask_pixel_correspondence_train(self, fake_dataset):
        """Sentinel pixels are replaced with depth_mean even on train path."""
        data_root, class_names, norm_stats = fake_dataset
        samples = _discover_samples(str(data_root / 'train'), class_names)
        ds = WashingtonRGBDDataset(
            data_root=str(data_root),
            split='train',
            samples=samples,
            class_names=class_names,
            norm_stats=norm_stats,
            crop_size=256,
            normalize=False,
        )
        depth_mean = norm_stats['depth_mean'][0]
        ds._flip_p = 0.0

        np.random.seed(12345)
        for _ in range(5):
            _, out_depth, _ = ds[0]
            assert out_depth[0, 0, 0].item() == pytest.approx(depth_mean)
            assert out_depth[0, 4, 4].item() == pytest.approx(depth_mean)

    def test_scale_jitter_shape_invariance(self, fake_dataset):
        """Depth scale jitter changes values but not spatial dimensions."""
        ds = _make_dataset(fake_dataset, split='train', normalize=False)
        np.random.seed(42)
        for _ in range(10):
            _, depth, _ = ds[0]
            assert depth.shape == (1, 224, 224)

    def test_test_split_no_augmentation(self, fake_dataset):
        """Test split uses CenterCrop only, no augmentation (deterministic)."""
        ds = _make_dataset(fake_dataset, split='test', crop_size=224)
        results = []
        for _ in range(3):
            rgb, depth, label = ds[0]
            results.append((rgb.clone(), depth.clone()))
        assert torch.equal(results[0][0], results[1][0])
        assert torch.equal(results[0][1], results[1][1])
        assert torch.equal(results[0][0], results[2][0])

    def test_val_split_no_augmentation(self, fake_dataset):
        """Val split is deterministic (no augmentation)."""
        ds = _make_dataset(fake_dataset, split='val', crop_size=224)
        r1 = ds[0]
        r2 = ds[0]
        assert torch.equal(r1[0], r2[0])
        assert torch.equal(r1[1], r2[1])

    def test_scalar_depth_stats_work_in_dataset(self, fake_dataset_scalar_depth_stats):
        """Dataset works when norm_stats.json has scalar depth values."""
        data_root, class_names, norm_stats = fake_dataset_scalar_depth_stats
        loaded_stats = _load_norm_stats(str(data_root))
        train_dir = str(data_root / 'train')
        samples = _discover_samples(train_dir, class_names)
        ds = WashingtonRGBDDataset(
            data_root=str(data_root),
            split='train',
            samples=samples,
            class_names=class_names,
            norm_stats=loaded_stats,
            crop_size=224,
            normalize=True,
        )
        rgb, depth, label = ds[0]
        assert rgb.shape == (3, 224, 224)
        assert depth.shape == (1, 224, 224)
        assert not torch.isnan(rgb).any()
        assert not torch.isnan(depth).any()

    def test_no_nan_in_output(self, fake_dataset):
        """No NaN values in any output tensors across multiple samples."""
        ds = _make_dataset(fake_dataset, split='train', normalize=True)
        np.random.seed(42)
        for i in range(min(10, len(ds))):
            rgb, depth, label = ds[i]
            assert not torch.isnan(rgb).any(), f"NaN in RGB at index {i}"
            assert not torch.isnan(depth).any(), f"NaN in depth at index {i}"
            assert not torch.isinf(rgb).any(), f"Inf in RGB at index {i}"
            assert not torch.isinf(depth).any(), f"Inf in depth at index {i}"


# ── Tests: _apply_hole_dropout ──────────────────────────────────────────


class TestHoleDropout:
    """Tests for _apply_hole_dropout method."""

    def test_creates_zero_regions(self, fake_dataset):
        """Hole dropout sets some pixels to 0."""
        ds = _make_dataset(fake_dataset, split='train')
        depth = torch.ones(1, 224, 224, dtype=torch.float32) * 2.5
        result = ds._apply_hole_dropout(depth)
        assert (result == 0.0).any()

    def test_preserves_shape_dtype(self, fake_dataset):
        """Output has same shape and dtype as input."""
        ds = _make_dataset(fake_dataset, split='train')
        depth = torch.ones(1, 224, 224, dtype=torch.float32) * 2.5
        result = ds._apply_hole_dropout(depth)
        assert result.shape == (1, 224, 224)
        assert result.dtype == torch.float32

    def test_modifies_in_place(self, fake_dataset):
        """_apply_hole_dropout modifies the input tensor in-place."""
        ds = _make_dataset(fake_dataset, split='train')
        depth = torch.ones(1, 224, 224, dtype=torch.float32) * 2.5
        result = ds._apply_hole_dropout(depth)
        assert result is depth


# ── Tests: get_washington_dataloaders ────────────────────────────────────


class TestGetWashingtonDataloaders:
    """Tests for get_washington_dataloaders factory."""

    def test_returns_four_elements(self, fake_dataset):
        """Factory returns (train_loader, val_loader, test_loader, num_classes)."""
        data_root, _, _ = fake_dataset
        result = get_washington_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        assert len(result) == 4
        train_loader, val_loader, test_loader, num_classes = result
        assert isinstance(num_classes, int)
        assert num_classes == 3
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_returns_five_elements_with_class_weights(self, fake_dataset):
        """Factory returns 5-tuple when use_class_weights=True."""
        data_root, _, _ = fake_dataset
        result = get_washington_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
            use_class_weights=True,
        )
        assert len(result) == 5
        _, _, _, num_classes, class_weights = result
        assert class_weights.shape == (num_classes,)

    def test_batch_shapes_train(self, fake_dataset):
        """Train batches have correct shapes."""
        data_root, _, _ = fake_dataset
        train_loader, _, _, _ = get_washington_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        rgb, depth, labels = next(iter(train_loader))
        assert rgb.shape == (4, 3, 224, 224)
        assert depth.shape == (4, 1, 224, 224)
        assert labels.shape == (4,)

    def test_batch_shapes_val(self, fake_dataset):
        """Val batches have correct shapes."""
        data_root, _, _ = fake_dataset
        _, val_loader, _, _ = get_washington_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        rgb, depth, labels = next(iter(val_loader))
        assert rgb.shape == (4, 3, 224, 224)
        assert depth.shape == (4, 1, 224, 224)
        assert labels.shape == (4,)

    def test_batch_shapes_test(self, fake_dataset):
        """Test batches have correct shapes."""
        data_root, _, _ = fake_dataset
        _, _, test_loader, _ = get_washington_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        rgb, depth, labels = next(iter(test_loader))
        assert rgb.shape == (4, 3, 224, 224)
        assert depth.shape == (4, 1, 224, 224)
        assert labels.shape == (4,)

    def test_val_deterministic(self, fake_dataset):
        """Val loader produces identical outputs on repeated iteration."""
        data_root, _, _ = fake_dataset
        _, val_loader, _, _ = get_washington_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        batch1 = next(iter(val_loader))
        batch2 = next(iter(val_loader))
        assert torch.equal(batch1[0], batch2[0])
        assert torch.equal(batch1[1], batch2[1])

    def test_test_deterministic(self, fake_dataset):
        """Test loader produces identical outputs on repeated iteration."""
        data_root, _, _ = fake_dataset
        _, _, test_loader, _ = get_washington_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        batch1 = next(iter(test_loader))
        batch2 = next(iter(test_loader))
        assert torch.equal(batch1[0], batch2[0])
        assert torch.equal(batch1[1], batch2[1])

    def test_all_splits_have_all_classes(self, fake_dataset):
        """Train, val, and test all contain samples from all classes."""
        data_root, _, _ = fake_dataset
        train_loader, val_loader, test_loader, _ = get_washington_dataloaders(
            data_root=str(data_root),
            batch_size=30,
            num_workers=0,
            seed=42,
        )
        for loader, name in [(train_loader, 'train'), (val_loader, 'val'),
                              (test_loader, 'test')]:
            labels = set()
            for _, _, batch_labels in loader:
                labels.update(batch_labels.tolist())
            assert len(labels) == 3, f"{name} missing classes: got {labels}"

    def test_num_workers_zero_no_crash(self, fake_dataset):
        """num_workers=0 works without prefetch_factor error."""
        data_root, _, _ = fake_dataset
        train_loader, val_loader, test_loader, _ = get_washington_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        next(iter(train_loader))
        next(iter(val_loader))
        next(iter(test_loader))

    def test_reproducible_loading(self, fake_dataset):
        """Same seed produces same val outputs."""
        data_root, _, _ = fake_dataset
        _, val1, _, _ = get_washington_dataloaders(
            data_root=str(data_root), batch_size=6, num_workers=0, seed=42
        )
        _, val2, _, _ = get_washington_dataloaders(
            data_root=str(data_root), batch_size=6, num_workers=0, seed=42
        )
        b1 = next(iter(val1))
        b2 = next(iter(val2))
        assert torch.equal(b1[0], b2[0])
        assert torch.equal(b1[2], b2[2])

    def test_missing_train_dir_raises(self, tmp_path):
        """FileNotFoundError raised when train/ dir missing."""
        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("apple\n")
        with open(tmp_path / 'norm_stats.json', 'w') as f:
            json.dump({
                'rgb_mean': [0.5, 0.5, 0.5], 'rgb_std': [0.2, 0.2, 0.2],
                'depth_mean': [0.85], 'depth_std': [0.25],
            }, f)
        with pytest.raises(FileNotFoundError, match="train/"):
            get_washington_dataloaders(data_root=str(tmp_path), num_workers=0)

    def test_empty_train_raises(self, tmp_path):
        """ValueError raised when train dir has no samples."""
        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("apple\n")
        with open(tmp_path / 'norm_stats.json', 'w') as f:
            json.dump({
                'rgb_mean': [0.5, 0.5, 0.5], 'rgb_std': [0.2, 0.2, 0.2],
                'depth_mean': [0.85], 'depth_std': [0.25],
            }, f)
        (tmp_path / 'train').mkdir()
        with pytest.raises(ValueError, match="No training samples"):
            get_washington_dataloaders(data_root=str(tmp_path), num_workers=0)

    def test_missing_val_test_returns_none(self, tmp_path):
        """Missing val/ and test/ dirs result in None loaders, not errors."""
        class_names = ['apple']
        with open(tmp_path / 'class_names.txt', 'w') as f:
            f.write("apple\n")
        with open(tmp_path / 'norm_stats.json', 'w') as f:
            json.dump({
                'rgb_mean': [0.5, 0.5, 0.5], 'rgb_std': [0.2, 0.2, 0.2],
                'depth_mean': [0.85], 'depth_std': [0.25],
            }, f)
        _create_split_samples(tmp_path, class_names, 'train',
                               objs_per_class=2, frames_per_obj=2)
        # No val/ or test/ dirs created
        train_loader, val_loader, test_loader, num_classes = get_washington_dataloaders(
            data_root=str(tmp_path), batch_size=4, num_workers=0, seed=42
        )
        assert train_loader is not None
        assert val_loader is None
        assert test_loader is None
        assert num_classes == 1

    def test_balanced_sampling_false(self, fake_dataset):
        """balanced_sampling=False uses shuffle instead of WeightedRandomSampler."""
        data_root, _, _ = fake_dataset
        result = get_washington_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
            balanced_sampling=False,
        )
        assert len(result) == 4
        train_loader, val_loader, test_loader, num_classes = result
        assert num_classes == 3
        assert train_loader.sampler.__class__.__name__ != 'WeightedRandomSampler'
        rgb, depth, labels = next(iter(train_loader))
        assert rgb.shape == (4, 3, 224, 224)

    def test_custom_crop_size(self, fake_dataset):
        """Custom crop_size propagates to output tensor shapes."""
        data_root, _, _ = fake_dataset
        train_loader, val_loader, test_loader, _ = get_washington_dataloaders(
            data_root=str(data_root),
            batch_size=4,
            num_workers=0,
            seed=42,
            crop_size=128,
        )
        rgb, depth, _ = next(iter(train_loader))
        assert rgb.shape == (4, 3, 128, 128)
        assert depth.shape == (4, 1, 128, 128)

        rgb, depth, _ = next(iter(val_loader))
        assert rgb.shape == (4, 3, 128, 128)

        rgb, depth, _ = next(iter(test_loader))
        assert rgb.shape == (4, 3, 128, 128)
