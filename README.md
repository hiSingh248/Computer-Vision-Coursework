# Computer-Vision-Coursework
Assignments from Computer Vision Coursework

**H1:Histograms, Filters, Deconvolution, Blending**

The goal in this assignment is to get you acquainted with filtering in the spatial domain as well as in the frequency domain.
Laplacian Blending using Image Pyramids is a very good intro to working and thinking in frequencies, and Deconvolution is a neat trick.
You tasks for this assignment are:
- Perform Histogram Equalization on the given input image.
- Perform Low-Pass, High-Pass and Deconvolution on the given input image.
- Perform Laplacian Blending on the two input images (blend them together).

**HW2: Image Alignment, Panoramas**

Your goal is to create 2 panoramas:
- Using homographies and perspective warping on a common plane (3 images).
- Using cylindrical warping (many images).

**HW3: Detection and Tracking**

Your goal is to:
- Detect the face in the first frame of the movie
- Using pre-trained Viola-Jones detector
- Track the face throughout the movie using:
  - CAMShift
  - Particle Filter
  - Face detector + Kalman Filter
- Bonus (20pt): Face Detector + Optical Flow tracker.

**HW4: Segmentation**

Your goal is to perform semi-automatic binary segmentation based on SLIC superpixels and graph-cuts.
- Given an image and sparse markings for foreground and background
- Calculate SLIC over image
- Calculate color histograms for all superpixels
- Calculate color histograms for FG and BG
- Construct a graph that takes into account superpixel-to-superpixel interaction (smoothness term), as well as superpixel-FG/BG interaction (match term)
- Run a graph-cut algorithm to get the final segmentation
- Bonus: Make it interactive: Let the user draw the markings, recalculate only the FG-BG histograms, construct the graph and get a segmentation from the max-flow graph-cut, show the result immediately to the user.

**HW5: Structured Light**

Your goal is to reconstruct a scene from multiple structured light scannings of it.

- Calibrate projector with the “easy” method.
  - Use ray-plane intersection
  - Get 2D-3D correspondence and use stereo calibration
- Get the binary code for each pixel - this you should do, but it's super easy.
- Correlate code with (x,y) position - we provide a "codebook" from binary code -> (x,y)
- With 2D-2D correspondence, Perform stereo triangulation (existing function) to get a depth map

**HW6: CNNs and Transfer Learning**

Your goal is to:
- Train an MNIST CNN classifier on just the digits: 1, 4, 5 and 9
- Architecture (suggested, you may change it):
  - "conv1": conv2d 3x3x4, stride=1, ReLU, padding = "SAME"
  - "conv2": conv2d 3x3x8, stride=2, ReLU, padding = "SAME"
  - "pool": pool 2x2
  - "fc1": fc 16
  - "fc2": fc 10
  - "softmax": xentropy loss, fc-logits = 4
  - Optimizer: ADAM
  - 5 epochs, 10 batch size
- Use your trained model’s weights on the lower 4 layers to train a classifier for the rest of MNIST (excluding 1,4,5 and 9)
  - Create new layers for the top (5 and 6)
  - Try to run as few epochs as possible to get a good classification (> 99% on test)
  - Try a session with freezing the lower layers weights, and also a session of just fine-tuning the weights.
  - Use (for speed) a constraint on the optimizer for freezing:
    -*train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="fc2_023678|softmax_023678")*
    -*training_op = optimizer.minimize(loss, var_list=train_vars)*
- Report:
  - Test loss curve on MNIST-1459
  - Test loss curve on transferred MNIST-023678:
    -with fine-tuning everything
    -with frozen layers up to fc2 (and not including)
  - Final execution graph (provided code)
 - Bonus 1:
   - Apply dropout regularization after conv1, conv2 and fc1
 - Bonus 2:
   - Visualize the filter maps (activations) for conv1, conv2 and pool

