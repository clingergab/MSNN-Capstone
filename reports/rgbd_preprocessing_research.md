# Washington RGB-D Object Dataset: Preprocessing Practices for Classification

## 1. Executive Summary

This report surveys how researchers preprocess the Washington RGB-D Object Dataset
(eval set) for object classification, focusing on image resolution, resizing strategies,
and evaluation protocols. The overwhelming consensus across the literature is:

- **Crops are resized to 256x256, then cropped to 227x227 or 224x224** depending on the CNN backbone (CaffeNet/AlexNet vs VGG/ResNet).
- **Upscaling ~80px crops to 256x256 is standard practice**, not unusual at all.
- **Researchers use the pre-cropped eval set images**, not the full 640x480 frames.
- **Aspect ratio is preserved** by scaling the long side to 256 and padding/tiling the short side, rather than direct warping.

---

## 2. Dataset Background

The Washington RGB-D Object Dataset (Lai et al., 2011) contains:
- **51 categories**, 300 object instances
- **~41,877 RGB-D frames** captured with a Kinect (640x480 native resolution)
- Each frame provides: RGB image, depth image, and segmentation mask
- The **evaluation set** provides tightly cropped bounding box images around objects,
  subsampled every 5th video frame
- Crop dimensions are variable and small, **typically 50-120px** on the longer side,
  with most falling in the **~73-89px range**

Source: [RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/dataset.html)

---

## 3. Resolution and Resizing Practices by Paper

### 3.1 Eitel et al. (2015) - "Multimodal Deep Learning for Robust RGB-D Object Recognition"
**Architecture:** Two-stream CaffeNet (AlexNet variant) with late fusion
**Input resolution:** 227x227 (cropped from 256x256)
**Preprocessing pipeline:**
1. Scale the image so the **longest side = 256 pixels** (preserving aspect ratio)
2. **Tile border pixels** along the shorter dimension to reach 256x256 (NOT warping, NOT zero-padding)
3. Randomly crop 227x227 subimages during training
4. Random horizontal flipping for augmentation
5. For depth: apply jet colormap encoding (single-channel depth -> 3-channel RGB-like image)

**Key insight:** The authors explicitly noted that "standard image warping was detrimental
to object recognition performance" due to shape information loss. This motivated their
border-tiling approach over simple resize.

Source: [arXiv:1507.06821](https://arxiv.org/abs/1507.06821)

### 3.2 Schwarz et al. (2015) - "RGB-D Object Recognition and Pose Estimation Based on Pre-trained CNN Features"
**Architecture:** CaffeNet (used as fixed feature extractor, no fine-tuning)
**Input resolution:** 227x227 (standard CaffeNet input)
**Preprocessing:** Rendered objects from canonical perspective; colorized depth by
distance from object center. Used standard CaffeNet preprocessing pipeline (resize to
256, crop to 227).

Source: [IEEE ICRA 2015](https://www.ais.uni-bonn.de/papers/ICRA_2015_Schwarz_RGB-D-Objects_Transfer-Learning.pdf)

### 3.3 Madai-Tahy et al. (2016) - "Revisiting Deep Convolutional Neural Networks for RGB-D Based Object Recognition"
**Architecture:** CaffeNet-based two-stream
**Input resolution:** 256x256 -> 227x227 crop
**Preprocessing:**
- **Border replication** to square the image: "the outer pixels of the longer side are
  replicated along the axis of the shorter side until the entire image is quadratic"
- Then resize to 256x256
- Crop to 227x227
- For depth: explored jet colormap, surface normals, and other colorization methods

Source: [Springer ICANN 2016](https://link.springer.com/chapter/10.1007/978-3-319-44781-0_4)

### 3.4 Zia et al. (2017) - "RGB-D Object Recognition Using Deep Convolutional Neural Networks"
**Architecture:** Deep CNN (ICCV 2017 workshop)
**Input resolution:** 227x227
**Preprocessing:**
1. Resize so the **long side = 227 pixels**
2. **Pad with black pixels** on the short side to make it square (227x227)
3. Center the original image in the padded result
4. For depth: HHA encoding (Horizontal disparity, Height above ground, Angle of surface normal)

**Key difference from Eitel:** Used zero-padding instead of border tiling, and resized
directly to 227 rather than going through 256 first.

Source: [ICCV 2017 Workshop](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w17/html/Zia_RGB-D_Object_Recognition_ICCV_2017_paper.html)

### 3.5 Later VGG/ResNet-based approaches (2018-2020)
**Architecture:** VGG-f, VGG-16, ResNet-18/50
**Input resolution:** 224x224 (standard for VGG/ResNet)
**Preprocessing:** Standard ImageNet pipeline:
1. Resize shortest side to 256
2. Center crop to 224x224 (inference) or random crop (training)
3. ImageNet mean subtraction
4. For depth: surface normals (3-channel) or colorjet encoding

Source: [Springer 2020 - ResNet for RGB-D](https://link.springer.com/chapter/10.1007/978-3-030-49556-5_15)

### 3.6 Loghmani et al. (2019) - "Recurrent Convolutional Fusion for RGB-D Object Recognition"
**Architecture:** Pre-trained ResNet with RNN fusion
**Input resolution:** 224x224 (ResNet standard)
**Preprocessing:** Standard ResNet preprocessing with randomized pooling for
multi-layer feature extraction.

Source: [arXiv:1806.01673](https://arxiv.org/abs/1806.01673)

### 3.7 Bo et al. (2012) - "Unsupervised Feature Learning for RGB-D Based Object Recognition"
**Architecture:** Hierarchical Matching Pursuit (HMP), not CNN-based
**Input resolution:** Variable (hand-crafted features, not requiring fixed input size)
**Preprocessing:** Kernel descriptors computed at multiple scales on the original crop
sizes. This is the pre-deep-learning approach that did NOT require resizing.

Source: [UW Technical Report](https://research.cs.washington.edu/istc/lfb/paper/iser12.pdf)

---

## 4. Summary Table

| Paper                     | Year | Backbone        | Target Resolution | Squaring Method          | Depth Encoding    |
|---------------------------|------|-----------------|-------------------|--------------------------|-------------------|
| Bo et al.                 | 2012 | HMP (non-CNN)   | Original size     | N/A                      | Raw depth         |
| Schwarz et al.            | 2015 | CaffeNet        | 256->227x227      | Standard CaffeNet        | Colorized depth   |
| Eitel et al.              | 2015 | CaffeNet x2     | 256->227x227      | Border tiling            | Jet colormap      |
| Madai-Tahy et al.         | 2016 | CaffeNet        | 256->227x227      | Border replication       | Surface normals   |
| Zia et al.                | 2017 | Deep CNN        | 227x227           | Black pixel padding      | HHA encoding      |
| Loghmani et al.           | 2019 | ResNet          | 224x224           | Standard ImageNet        | Surface normals   |
| Later ResNet/VGG work     | 2020 | ResNet-50/VGG   | 224x224           | Standard ImageNet        | Various           |

---

## 5. Answers to Specific Questions

### Q1: What resolution do papers typically resize these crops to?

**Answer: 256x256 (intermediate) -> 227x227 or 224x224 (final input)**

The CaffeNet/AlexNet era (2015-2017) universally used **227x227** (cropped from 256x256).
The VGG/ResNet era (2018+) shifted to **224x224** (cropped from 256x256). No papers
in the surveyed literature used 128x128 or kept the original ~80px size for CNN-based
approaches.

### Q2: Do people pad to square first, or just resize directly?

**Answer: They pad/tile to square first, then resize. Direct warping is avoided.**

Three distinct approaches are used:
1. **Border tiling** (Eitel et al.): Replicate edge pixels to fill the shorter dimension.
   This is the most cited approach and reportedly gives best results.
2. **Border replication** (Madai-Tahy et al.): Similar to border tiling -- replicate outer
   pixels until the image is square.
3. **Black pixel padding** (Zia et al.): Pad with zeros to make square, centering the
   original image.

Direct aspect-ratio-distorting resize (warping) is explicitly warned against by Eitel et al.
as "detrimental to object recognition performance."

### Q3: Do researchers use the crops as-is or extract larger context from full images?

**Answer: Researchers use the pre-cropped evaluation set images.**

All surveyed papers for the **category recognition** task use the tightly cropped bounding
box images from the eval set. The full 640x480 images with loc.txt are used for the
**object detection** task (a different evaluation), not for classification/recognition.

The evaluation set documentation explicitly states these crops are "exactly as used in the
object recognition evaluation."

### Q4: What are the standard evaluation protocols?

**Answer: Leave-one-instance-out with 10 random trials.**

The standard protocol:
1. For each of the 51 categories, randomly select one object instance to hold out for testing
2. Train on remaining instances, test on held-out instances
3. Repeat for **10 random trials** (the exact splits are provided with the dataset)
4. Report **mean accuracy +/- standard deviation** across the 10 trials
5. Each trial has approximately **~35,000 training images** and **~7,000 test images**

This is NOT traditional k-fold cross-validation. It tests generalization to **unseen object
instances** within known categories, which is a harder test than random image splits.

### Q5: Notable papers using RGB-D with depth - what resolution?

All CNN-based papers in this survey used depth and all resized to the same resolution as
their RGB stream:

- **Eitel et al. (2015):** 256->227x227, jet colormap on depth
- **Schwarz et al. (2015):** 256->227x227, distance-based colorization
- **Madai-Tahy et al. (2016):** 256->227x227, surface normals encoding
- **Zia et al. (2017):** 227x227, HHA encoding
- **ResNet-based (2019-2020):** 224x224, surface normals or colorjet

The depth image undergoes the SAME spatial preprocessing (resize, pad, crop) as the RGB
image, then is converted from single-channel to 3-channel via one of several encoding
methods (jet colormap, HHA, surface normals).

### Q6: Is upscaling ~80px crops to 256x256 common or unusual?

**Answer: This is completely standard and universal practice.**

Every CNN-based paper on this dataset upscales the small crops (typically 50-120px) to
256x256 (or directly to 224/227). This represents a **~3-4x upscaling factor**, which is
large but necessary for compatibility with ImageNet-pretrained networks.

The key considerations when upscaling:
- **Border tiling/replication is preferred** over direct resize to avoid distortion
- The **aspect ratio is preserved** during the initial scaling step
- **Transfer learning from ImageNet** requires these standard input sizes
- The upscaling artifacts are partially mitigated by the CNN's learned robustness to
  scale variations (from ImageNet pre-training)

No papers in the survey attempted to use a smaller input size (e.g., 64x64 or 128x128)
to reduce the upscaling factor. The consensus is that matching the ImageNet pre-training
resolution is more important than minimizing upscaling artifacts.

---

## 6. Recommended Preprocessing Pipeline

Based on the literature consensus, the recommended pipeline for this dataset is:

```
1. Load RGB crop (variable size, e.g., 73x89)
2. Load depth crop (same dimensions)
3. Load mask (same dimensions)

4. For RGB:
   a. Scale longest side to 256, preserving aspect ratio
   b. Tile/replicate border pixels to make 256x256 square
   c. During training: random crop to 224x224 (ResNet) or 227x227 (AlexNet)
   d. During eval: center crop to 224x224 or 227x227
   e. Normalize with ImageNet mean/std

5. For Depth:
   a. Apply same spatial transforms as RGB (resize, pad, crop)
   b. Encode to 3 channels via one of:
      - Jet/colorjet colormap (simplest, competitive performance)
      - Surface normals (3 channels: nx, ny, nz)
      - HHA encoding (horizontal disparity, height, angle)
   c. Normalize (method-dependent)

6. For Mask:
   a. Apply same spatial transforms using nearest-neighbor interpolation
```

---

## 7. Depth Encoding Methods Comparison

| Method          | Channels | Complexity | Performance | Notes                           |
|-----------------|----------|------------|-------------|---------------------------------|
| Jet colormap    | 3        | Trivial    | Good        | Most popular, negligible overhead |
| Surface normals | 3        | Moderate   | Good-Best   | Requires normal estimation       |
| HHA encoding    | 3        | High       | Good        | Requires camera intrinsics       |
| Raw replicated  | 3        | Trivial    | Baseline    | Just copy depth to 3 channels    |

---

## 8. Sources Consulted

1. Lai, K., Bo, L., Ren, X., & Fox, D. (2011). "A Large-Scale Hierarchical Multi-View RGB-D Object Dataset." ICRA 2011. [Link](https://research.cs.washington.edu/istc/lfb/paper/icra11a.pdf)
2. Bo, L., Ren, X., & Fox, D. (2012). "Unsupervised Feature Learning for RGB-D Based Object Recognition." ISER 2012. [Link](https://research.cs.washington.edu/istc/lfb/paper/iser12.pdf)
3. Schwarz, M. et al. (2015). "RGB-D Object Recognition and Pose Estimation Based on Pre-trained CNN Features." ICRA 2015. [Link](https://www.ais.uni-bonn.de/papers/ICRA_2015_Schwarz_RGB-D-Objects_Transfer-Learning.pdf)
4. Eitel, A. et al. (2015). "Multimodal Deep Learning for Robust RGB-D Object Recognition." IROS 2015. [Link](https://arxiv.org/abs/1507.06821)
5. Madai-Tahy, L. et al. (2016). "Revisiting Deep Convolutional Neural Networks for RGB-D Based Object Recognition." ICANN 2016. [Link](https://link.springer.com/chapter/10.1007/978-3-319-44781-0_4)
6. Zia, S. et al. (2017). "RGB-D Object Recognition Using Deep Convolutional Neural Networks." ICCV 2017 Workshop. [Link](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w17/html/Zia_RGB-D_Object_Recognition_ICCV_2017_paper.html)
7. Loghmani, M.R. et al. (2019). "Recurrent Convolutional Fusion for RGB-D Object Recognition." [Link](https://arxiv.org/abs/1806.01673)
8. RGB-D Object Dataset Official Website. [Link](https://rgbd-dataset.cs.washington.edu/dataset.html)
9. RGB-D Object Dataset Evaluation Set. [Link](https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_eval/)
10. Various ResNet/VGG implementations for RGB-D (2019-2020). [Link](https://link.springer.com/chapter/10.1007/978-3-030-49556-5_15)
