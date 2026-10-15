# Drishti: Comprehensive Technical Foundations & Mathematical Derivations

This document is engineered to provide an exhaustive, beginner-to-advanced deep dive into the physical, mathematical, biological, and algorithmic principles governing the Drishti Assistive Wearable. Every topic is explained from first principles—starting with intuitive physical concepts, moving through formal mathematical proofs, and concluding with edge implementation mechanics.

---

# Table of Contents
1. [The Human Problem: Visual Impairment & Sensory Substitution](#1-the-human-problem-visual-impairment--sensory-substitution)
2. [Optics & Perspective Geometry (Photons to 3D Rays)](#2-optics--perspective-geometry-photons-to-3d-rays)
3. [Time-of-Flight (ToF) Physics & Ray-Cone Spatial Fusion](#3-time-of-flight-tof-physics--ray-cone-spatial-fusion)
4. [Deep Learning Edge Quantization & The Roofline Model](#4-deep-learning-edge-quantization--the-roofline-model)
5. [Indian Currency Note (INR) Recognition](#5-indian-currency-note-inr-recognition)
6. [Multilingual Indic Optical Character Recognition (OCR)](#6-multilingual-indic-optical-character-recognition-ocr)
7. [Vision-Language Models (VLM) for Scene Narration](#7-vision-language-models-vlm-for-scene-narration)
8. [Psychoacoustics & 3D Spatial Audio Synthesis (ITD & ILD)](#8-psychoacoustics--3d-spatial-audio-synthesis-itd--ild)
9. [Edge Privacy & Caregiver Telemetry Architecture](#9-edge-privacy--caregiver-telemetry-architecture)

---

# 1. The Human Problem: Visual Impairment & Sensory Substitution

### 1.1 The Navigational Challenge
When a sighted person walks through an unfamiliar room, their visual system performs continuous, parallel spatial mapping:
- Floor surfaces, steps, and drop-offs are detected instantly.
- Distances to obstacles are perceived via binocular stereopsis and motion parallax.
- Signs, room labels, and currency notes are read effortlessly.

For a visually impaired individual, traditional white canes provide tactile feedback only for objects in physical contact with the cane tip ($1.0\text{ to } 1.5\text{ meters}$ forward at ground level). Head-height obstructions (open cabinet doors, tree branches), moving vehicles, and unreadable signage remain constant hazards.

### 1.2 Sensory Substitution: Vision to Sound
*Sensory substitution* is the scientific principle of translating information normally acquired by one sensory modality (vision) into signals that can be interpreted by another intact modality (hearing).

However, vision and audition operate on fundamentally different informational bandwidths:
- **Visual Cortex Bandwidth**: $\approx 10^7 \text{ bits/second}$. The human retina contains over 100 million photoreceptors operating in parallel across a 2D spatial plane.
- **Auditory Comprehension Bandwidth**: $\approx 30\text{ to } 50 \text{ bits/second}$ ($\approx 150\text{ words/minute}$). Human speech processing is strictly serial—we hear one word after another over time.

### 1.3 The Danger of Auditory Clutter & Cognitive Load
If an assistive wearable uses continuous text-to-speech to narrate everything in front of the user (*"Chair ahead, table on left, door on right, person walking..."*):
1. **Auditory Blindness**: The user's ears are overwhelmed by robotic speech, drowning out ambient environmental sounds (footsteps, traffic, echoes) that visually impaired people rely upon for orientation.
2. **Latency Fatalities**: Human reaction time to an audio cue is $\approx 150\text{ ms}$. If an obstacle alert requires a 3-second spoken sentence (*"Caution, there is a low table one meter in front of you"*), the user will physically collide with the object before the sentence finishes playing.

### 1.4 The Drishti Two-Tier Solution
To solve this fundamental biological bottleneck, Drishti strictly decouples perception into:
1. **Fast Perception Stream (Continuous, sub-100ms)**: Converts physical obstacles into intuitive, non-verbal **binaural spatial sound cues** (clicks and tones that sound like they are physically originating from the exact 3D location of the obstacle).
2. **On-Demand Cognitive Stream (Triggered)**: Natural language speech (OCR, Currency, Scene Narration) is engaged **only when explicitly requested** by the user via a tactile button or voice command.

---

# 2. Optics & Perspective Geometry (Photons to 3D Rays)

### 2.1 From Photons to Pixels: How Digital Image Sensors Work
Light consists of packets of electromagnetic energy called photons. When sunlight or indoor lamps illuminate an object, photons reflect off the object's surface toward the camera lens.

1. **Photodiode Array**: The camera sensor (e.g., Sony IMX708 in Raspberry Pi Camera Module 3) consists of an array of microscopic silicon photodiodes (e.g., $4608 \times 2592$ pixels).
2. **Photoelectric Effect**: When an incident photon with energy $E = h \nu$ strikes silicon, it excites an electron from the valence band into the conduction band, generating an electron-hole pair.
3. **Charge Accumulation & ADC**: During the exposure time, accumulated electrical charge is converted into an analog voltage and digitized by an Analog-to-Digital Converter (ADC) into an integer value between $0$ (pure black) and $255$ (maximum brightness) across Red, Green, and Blue color channels (via a Bayer color filter matrix).

### 2.2 The Pinhole Camera Model
To mathematically model how a 3D point in the physical world appears on a 2D digital image, we use the **pinhole camera model**.

Imagine a box with a tiny pinhole at the origin $\mathbf{O} = [0, 0, 0]^T$. The image sensor plane is located at a distance $f$ (the focal length) behind the pinhole along the optical $z$-axis.

Let a physical point in 3D camera coordinates be:
$$\mathbf{P}_c = \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$

By similar triangles, the light ray passing from $\mathbf{P}_c$ through the pinhole intersects the 2D image plane at coordinates $(x, y)$:
$$\frac{x}{f} = \frac{X_c}{Z_c} \implies x = f \frac{X_c}{Z_c}$$
$$\frac{y}{f} = \frac{Y_c}{Z_c} \implies y = f \frac{Y_c}{Z_c}$$

Notice that dividing by $Z_c$ (metric depth) causes distant objects to appear smaller on the image plane. This division is called **perspective division**.

```
          Y_c ^
              |        Physical Object (X_c, Y_c, Z_c)
              |           *
              |          /|
              |         / |
              |        /  |
              |       /   |
              |      /    |
              |     /     |
              |    /      |
  ----+-------O---/-------+---------> Z_c (Optical Axis)
      |       |  /        |
      | Image | /         |
      | Plane |/          |
      |  (u,v)*           |
      |       |           |
```

### 2.3 The Camera Intrinsic Matrix ($\mathbf{K}$)
In a digital sensor, coordinates are measured in discrete integer **pixel columns ($u$) and rows ($v$)**, starting from the top-left corner $(0, 0)$.

To convert metric sensor coordinates $(x, y)$ in millimeters into pixel coordinates $(u, v)$:
$$u = f_x \frac{X_c}{Z_c} + c_x$$
$$v = f_y \frac{Y_c}{Z_c} + c_y$$

where:
- $f_x = \frac{f}{\mu_x}$ is the horizontal focal length in pixel units ($\mu_x$ is the physical pixel width in mm).
- $f_y = \frac{f}{\mu_y}$ is the vertical focal length in pixel units ($\mu_y$ is the physical pixel height in mm).
- $(c_x, c_y)$ is the **principal point**, the pixel location where the optical axis pierces the sensor (approximately the center of the image).

In projective homogeneous coordinates:
$$\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \mathbf{P}_c = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$
where $\lambda = Z_c$ is the scalar depth.

### 2.4 Lens Distortion Modeling (Brown-Conrady)
Real physical lenses bend light rays unevenly, creating geometric distortion:
- **Radial Distortion (Barrel / Pincushion)**: Occurs because light rays bend more at the edges of the curved glass lens than at the optical center.
- **Tangential Distortion**: Occurs when the lens elements are not perfectly parallel to the silicon sensor plane during manufacturing assembly.

Let normalized undistorted coordinates be:
$$x_n = \frac{X_c}{Z_c} = \frac{u - c_x}{f_x}, \quad y_n = \frac{Y_c}{Z_c} = \frac{v - c_y}{f_y}$$
$$r^2 = x_n^2 + y_n^2$$

The distorted normalized coordinates $(x_d, y_d)$ are calculated using the Brown-Conrady polynomial:
$$x_d = x_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + \underbrace{2 p_1 x_n y_n + p_2 (r^2 + 2 x_n^2)}_{\text{tangential distortion}}$$
$$y_d = y_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + \underbrace{p_1 (r^2 + 2 y_n^2) + 2 p_2 x_n y_n}_{\text{tangential distortion}}$$

where $k_1, k_2, k_3$ are radial distortion coefficients and $p_1, p_2$ are tangential distortion coefficients. In Drishti, all detected bounding box coordinates are undistorted using inverse calibration matrices prior to spatial calculations.

### 2.5 Back-Projecting 2D Bounding Boxes to 3D Unit Rays
When the YOLO object detector identifies an obstacle (e.g., a chair) with pixel bounding box $[u_{min}, v_{min}, u_{max}, v_{max}]$, its visual centroid is:
$$u_0 = \frac{u_{min} + u_{max}}{2}, \quad v_0 = \frac{v_{min} + v_{max}}{2}$$

To find the physical line in 3D space along which this object lies, we compute the inverse projection through the camera intrinsic matrix:
$$\mathbf{v}_{ray} = \mathbf{K}^{-1} \begin{bmatrix} u_0 \\ v_0 \\ 1 \end{bmatrix} = \begin{bmatrix} \frac{u_0 - c_x}{f_x} \\ \frac{v_0 - c_y}{f_y} \\ 1 \end{bmatrix}$$

Normalizing this vector gives the **unit direction ray** $\hat{\mathbf{r}}$:
$$\hat{\mathbf{r}} = \frac{\mathbf{v}_{ray}}{\|\mathbf{v}_{ray}\|_2}$$

The physical horizontal azimuth angle $\theta_{az}$ and vertical elevation angle $\phi_{el}$ of the obstacle relative to the user's straight-ahead gaze are:
$$\theta_{az} = \arctan\left( \frac{u_0 - c_x}{f_x} \right)$$
$$\phi_{el} = \arctan\left( \frac{v_0 - c_y}{f_y} \right)$$

---

# 3. Time-of-Flight (ToF) Physics & Ray-Cone Spatial Fusion

### 3.1 The Scale Ambiguity of Monocular Vision
A fundamental limitation of a single 2D camera is **scale ambiguity**:
Dividing by $Z_c$ means that an infinite number of 3D points lying along the ray $\hat{\mathbf{r}}$ produce the exact same pixel $(u, v)$. A small toy car $0.3\text{ m}$ away produces the exact same image as a real car $30\text{ m}$ away.

For a visually impaired user navigating real-world spaces, knowing that an obstacle exists is useless without knowing its **exact metric distance**. A chair 4 meters away is safe; a chair 0.6 meters away requires an immediate stop.

### 3.2 Physical Principles of Direct Time-of-Flight (dToF)
To resolve metric depth with millimetric precision, Drishti pairs the camera with the STMicroelectronics VL53L1X Time-of-Flight sensor.

```
       [ VCSEL Laser Diode ] ───(940nm Infrared Pulse)───► [ Obstacle Surface ]
                                                                   │
                                                                   │ (Reflected Photons)
                                                                   ▼
       [ SPAD Avalanche Array ] ◄──────────────────────────────────+
```

1. **Laser Emission**: A Vertical-Cavity Surface-Emitting Laser (VCSEL) emits periodic bursts of infrared light at a wavelength of $\lambda = 940\text{ nm}$ (completely invisible and safe to the human eye).
2. **Photon Flight**: The light pulses travel forward through the air at the speed of light:
   $$c \approx 2.99792 \times 10^8 \text{ m/s} \quad (\approx 30 \text{ cm per nanosecond})$$
3. **Reflective Arrival & SPAD Array**: Photons reflect off obstacles and strike a Single-Photon Avalanche Diode (SPAD) array. When a single photon hits a reverse-biased SPAD above its breakdown voltage, it triggers an avalanche breakdown current pulse.
4. **Distance Equation**: The round-trip transit time $\Delta t$ is measured by picosecond Time-to-Digital Converters (TDCs):
   $$d = \frac{c \cdot \Delta t}{2}$$
   *(The factor of 2 accounts for the two-way journey to the object and back).*

### 3.3 Conical Field of View & Multi-Zone RoI
The VL53L1X sensor emits and receives light within a **conical acceptance cone** with a full Field of View (FoV) of:
$$2\theta_{FoV} \approx 27^\circ \implies \theta_{FoV} \approx 13.5^\circ \text{ (half-angle)}$$

The sensor's $16 \times 16$ SPAD array can be dynamically programmed into distinct Regions of Interest (RoIs), such as Left ($4 \times 16$), Center ($8 \times 16$), and Right ($4 \times 16$), providing multi-zone distance readings across the cone.

### 3.4 Ray-Cone Metric Fusion Algorithm
Let an object detected by the camera have bounding box $[u_{min}, v_{min}, u_{max}, v_{max}]$.
Using the intrinsic matrix $\mathbf{K}$, we calculate the horizontal angular span of the bounding box:
$$\theta_{min} = \arctan\left(\frac{u_{min} - c_x}{f_x}\right), \quad \theta_{max} = \arctan\left(\frac{u_{max} - c_x}{f_x}\right)$$

We evaluate the geometric intersection between the bounding box angular footprint and the active ToF cone:
$$\mathcal{S}_{overlap} = \left[\theta_{min}, \theta_{max}\right] \cap \left[-\theta_{FoV}, +\theta_{FoV}\right]$$

If $\mathcal{S}_{overlap} \neq \emptyset$, the physical object is located within the active laser cone. The metric distance $d_{ToF}$ reported by the sensor is projected along the centroid ray $\hat{\mathbf{r}}$ to yield the **true 3D metric coordinate vector**:
$$\mathbf{P}_{obstacle} = \begin{bmatrix} X_{obj} \\ Y_{obj} \\ Z_{obj} \end{bmatrix} = d_{ToF} \begin{bmatrix} \sin\theta_{az} \cos\phi_{el} \\ \sin\phi_{el} \\ \cos\theta_{az} \cos\phi_{el} \end{bmatrix}$$

### 3.5 Geometric Fallback Depth Estimation
When an obstacle lies outside the $27^\circ$ laser cone (e.g., at the extreme left or right edge of the camera's $120^\circ$ wide lens), Drishti uses geometric perspective estimation:
$$Z_{est} \approx \frac{f_y \cdot H_{real}}{h_{pixels}}$$
where $H_{real}$ is the statistical prior height of the classified object (e.g., $1.7\text{ m}$ for a standing human, $0.85\text{ m}$ for a chair) and $h_{pixels} = v_{max} - v_{min}$ is the bounding box pixel height.

---

# 4. Deep Learning Edge Quantization & The Roofline Model

### 4.1 What an Object Detector Does
An object detection model (such as YOLO nano) processes an input image $\mathbf{I} \in \mathbb{R}^{3 \times 640 \times 640}$ through dozens of convolutional layers. Each layer applies thousands of 2D filter kernels to detect edges, corners, textures, shapes, and semantic objects:
$$Y[c, i, j] = \sum_{k=0}^{C_{in}-1} \sum_{m=-K}^{K} \sum_{n=-K}^{K} W[c, k, m, n] \cdot X[k, i+m, j+n] + B[c]$$

A single forward pass of YOLO nano requires over **4.5 billion arithmetic operations** (FLOPs).

### 4.2 The Edge Compute Bottleneck: Why FP32 Fails on Edge CPUs
In standard machine learning frameworks (PyTorch, TensorFlow), every weight and activation is stored as a 32-bit IEEE 754 Floating-Point number (`FP32`):
- 1 bit sign, 8 bits exponent, 23 bits mantissa.
- 1 float requires 4 bytes of memory.
- A 3-million-parameter network requires $3 \times 10^6 \times 4 \text{ bytes} = 12 \text{ MB}$ just to store weights.

On resource-constrained hardware like the Raspberry Pi 5 (Quad-core ARM Cortex-A76 @ 2.4 GHz):
1. **Memory Bandwidth Saturation**: Transferring 12 MB of weights plus gigabytes of intermediate activation tensors to and from LPDDR4X RAM 30 times every second saturates the external memory bus.
2. **Thermal Dissipation**: High-precision floating-point arithmetic consumes substantial power, triggering thermal throttling ($> 80^\circ\text{C}$) and reducing clock frequencies.

### 4.3 Affine Uniform Quantization: Deriving FP32 to INT8
Quantization maps continuous real numbers $x \in [\alpha, \beta]$ into 8-bit signed integers $q \in [-128, 127]$.

The fundamental affine relation is:
$$x \approx S \cdot (q - Z)$$
$$q = \text{clamp}\left( \left\lfloor \frac{x}{S} \right\rceil + Z, \; -128, \; 127 \right)$$

where:
- $S \in \mathbb{R}^+$ is the **scale factor**:
  $$S = \frac{\beta - \alpha}{q_{max} - q_{min}} = \frac{\beta - \alpha}{255}$$
- $Z \in \mathbb{Z}$ is the **zero-point** integer that ensures real $0.0$ maps exactly to an integer (critical so that zero-padding in convolutional padding does not cause numerical distortion):
  $$Z = \text{round}\left( -\frac{\alpha}{S} \right) + q_{min}$$

For **symmetric quantization** (commonly used for convolution weights where the dynamic distribution is centered around zero):
$$\alpha = -\beta \implies Z = 0, \quad S = \frac{\max(|x|)}{127}$$

```
Real FP32 Scale:    -15.0 .............. 0.0 ................. +15.0
                      |                    |                     |
                      v                    v                     v
INT8 Quantized:     -128 ................  0 .................. +127
```

### 4.4 Fast Integer Convolution Arithmetic
Consider a convolution layer computing $Y = W X$. Substituting the quantized representations:
$$S_Y (q_Y - Z_Y) = \sum \left( S_W (q_W - Z_W) \right) \cdot \left( S_X (q_X - Z_X) \right)$$

Assuming symmetric weights ($Z_W = 0$):
$$q_Y = Z_Y + \underbrace{\left( \frac{S_W S_X}{S_Y} \right)}_{M} \sum q_W \cdot (q_X - Z_X)$$

The term $M = \frac{S_W S_X}{S_Y}$ is a constant real multiplier strictly between $0$ and $1$. It is decomposed into fixed-point integer arithmetic:
$$M \approx 2^{-n} M_0 \quad \text{where } M_0 \in \mathbb{Z}, n \in \mathbb{N}$$

Now the edge processor executes the entire neural network using:
- Pure 8-bit integer multiplications (`q_W * q_X`),
- 32-bit integer accumulation (`int32`),
- A bit-shift right by $n$ bits (`>> n`) to apply $M$,
- Completely bypassing all floating-point units.

### 4.5 The Hardware Roofline Model
The attainable computational throughput on any edge processor is governed by the **Roofline Model**:
$$\text{Attainable Performance [GFLOP/s]} = \min\left( \Pi_{peak}, \; \beta_{mem} \times I \right)$$
where:
- $\Pi_{peak}$ is peak processor computation speed.
- $\beta_{mem}$ is sustained memory bandwidth in GB/s.
- $I = \frac{\text{Operations}}{\text{Bytes Transferred}}$ is the **arithmetic intensity**.

```
Performance ^
(GFLOP/s)   |                    +------------------- Peak Compute Limit (\Pi_peak)
            |                   /
            |                  /
            |                 /   <-- Compute-Bound Region
            |                /
            |               /
            |              /
            |             /  <-- Memory-Bandwidth Bound Region
            |            /
            |           /
            +----------+----------------------------> Arithmetic Intensity (I)
                       I* (Balance Point)
```

By switching from FP32 to INT8:
- Weight footprint shrinks by **$75\%$** (from 4 bytes to 1 byte per weight).
- Arithmetic intensity $I$ increases by $4\times$.
- The application moves out of the memory-bandwidth-choked regime into the compute-efficient regime, achieving $30\text{ FPS}$ on Raspberry Pi 5 CPU and $> 60\text{ FPS}$ on the Hailo-8L NPU.

---

# 5. Indian Currency Note (INR) Recognition

### 5.1 The Accessibility Problem of Paper Banknotes
Paper currency notes in the Indian Rupee (INR) system—specifically the *Mahatma Gandhi New Series*—contain subtle tactile features (intaglio-printed raised geometric shapes and bleed lines on the left and right borders):
- ₹100: Triangle with 4 bleed lines.
- ₹200: Circle with 4 bleed lines and 2 internal circles.
- ₹500: Circle with 5 bleed lines.

In practice, circulating paper currency rapidly deteriorates:
1. Friction, moisture, and creasing flatten the raised tactile intaglio ink within months.
2. Visually impaired users frequently report difficulty distinguishing between soiled ₹20, ₹50, and ₹100 notes by touch alone, leaving them vulnerable to deliberate fraud during commercial transactions.

### 5.2 Visual Features of the Mahatma Gandhi New Series
Drishti implements an automated banknote classification engine based on distinct visual invariant features:

| Denomination | Base Color | Dimensions (Width $\times$ Height) | Reverse Motif |
| :--- | :--- | :--- | :--- |
| **₹10** | Chocolate Brown | $123 \text{ mm} \times 63 \text{ mm}$ (Aspect: 1.95) | Konark Sun Temple Wheel |
| **₹20** | Greenish Yellow | $129 \text{ mm} \times 63 \text{ mm}$ (Aspect: 2.04) | Ellora Caves |
| **₹50** | Fluorescent Blue| $135 \text{ mm} \times 66 \text{ mm}$ (Aspect: 2.04) | Hampi with Chariot |
| **₹100**| Lavender | $142 \text{ mm} \times 66 \text{ mm}$ (Aspect: 2.15) | Rani ki Vav (Stepwell) |
| **₹200**| Bright Yellow | $146 \text{ mm} \times 66 \text{ mm}$ (Aspect: 2.21) | Sanchi Stupa |
| **₹500**| Stone Grey | $150 \text{ mm} \times 66 \text{ mm}$ (Aspect: 2.27) | Red Fort (Delhi) |

### 5.3 Classification Architecture
The banknote classifier uses a fast, multi-stage physical and optical verification pipeline:
1. **High-Frequency Edge Density & Intaglio Texture Gating**:
   - Computes local spatial intensity gradients across the candidate image:
     $$G_x(x, y) = I(x+1, y) - I(x-1, y), \quad G_y(x, y) = I(x, y+1) - I(x, y-1)$$
     $$|\nabla I(x, y)| = |G_x(x, y)| + |G_y(x, y)|$$
   - The mean gradient density $\mu_{edge} = \frac{1}{HW} \sum |\nabla I|$ must satisfy $\mu_{edge} \ge 14.0$ to pass. Smooth background surfaces (human faces, shirts, walls) have $\mu_{edge} < 8.0$ and are immediately rejected, preventing false positive announcements.
2. **Geometric Aspect Ratio Verification**:
   - Evaluates the bounding ratio $\text{AR} = \frac{\text{Width}}{\text{Height}}$.
   - Enforces Mahatma Gandhi New Series physical envelope bounds: $1.60 \le \text{AR} \le 2.60$.
3. **BGR-to-HSV Chromatic Ink Distribution & Disambiguation**:
   - Converts standard camera BGR pixel buffers directly to Hue, Saturation, Value color space without external dependencies.
   - Evaluates circular angular difference $\Delta \theta = \min(|H - \theta_{ref}|, 360^\circ - |H - \theta_{ref}|)$ against dedicated denomination profiles:
     - ₹10: $\theta_{ref} = 18^\circ$ (Chocolate Brown)
     - ₹20: $\theta_{ref} = 42^\circ$ (Greenish Yellow, $\text{AR} = 2.05$)
     - ₹50: $\theta_{ref} = 195^\circ$ (Fluorescent Blue)
     - ₹100: $\theta_{ref} = 265^\circ$ (Lavender)
     - ₹200: $\theta_{ref} = 58^\circ$ (Bright Yellow, $\text{AR} = 2.21$)
     - ₹500: $\theta_{ref} = 110^\circ$ (Stone Grey / Greenish)
   - Resolves potential ₹20 vs ₹200 ambiguity through dedicated angular tolerance bands ($\le 12^\circ$) and aspect ratio error weighting ($0.70 \times \text{Score}_{hue} + 0.30 \times \text{Score}_{aspect}$).
4. **Immediate Spoken Verification**:
   - If confidence exceeds the threshold ($\ge 0.50$), the speech engine speaks the verified denomination (e.g. *"Fifty rupees"*).

---

# 6. Multilingual Indic Optical Character Recognition (OCR)

### 6.1 The Two-Stage OCR Pipeline
Optical Character Recognition translates visual pixel strips of text into machine-readable digital character strings. It consists of two sequential phases:
1. **Text Detection**: Locating bounding polygons around words or text lines within a complex background scene.
2. **Text Recognition**: Converting the pixel region inside each polygon into an ordered sequence of Unicode characters.

```
Input Image ──► [ DBNet Text Detection ] ──► Bounding Polygons ──► [ CRNN + CTC Recognition ] ──► Unicode Text
```

### 6.2 Differentiable Binarization (DBNet) for Text Detection
Traditional binarization applies a rigid step function to probability heatmaps:
$$B_{i,j} = \begin{cases} 1 & \text{if } P_{i,j} \ge T \\ 0 & \text{otherwise} \end{cases}$$
The standard step function is non-differentiable: its derivative is zero everywhere and undefined at the threshold, meaning neural networks cannot backpropagate gradients through it during training.

DBNet solves this by replacing the hard threshold with a **differentiable approximate binarization function**:
$$\hat{B}_{i,j} = \frac{1}{1 + \exp\left( -s \cdot (P_{i,j} - T_{i,j}) \right)}$$
where:
- $P_{i,j}$ is the predicted probability that pixel $(i,j)$ belongs to text.
- $T_{i,j}$ is an adaptive threshold map learned dynamically by the network.
- $s$ is an amplification scale factor ($s = 50$).

This allows the network to learn sharp, precise text polygon boundaries even under uneven shadows, wrinkled medicine packaging, or curved road signboards.

### 6.3 Sequence Recognition via CTC Loss
In Latin or Indic text reading, segmenting each character into an isolated bounding box is error-prone because letters connect smoothly or overlap.

**Connectionist Temporal Classification (CTC)** maps an entire image feature sequence $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_T)$ to a text string $\mathbf{l} = (l_1, \dots, l_U)$ ($U \le T$) without character-level segmentation.

1. At each time step $t$, the recurrent network outputs a probability distribution over the character vocabulary $\Sigma \cup \{\epsilon\}$ (where $\epsilon$ is a special blank token).
2. A path $\pi = (\pi_1, \dots, \pi_T)$ is a sequence of frame-level labels.
3. The collapse operator $\mathcal{B}$ removes adjacent duplicate characters and blanks:
   $$\mathcal{B}(\text{c } \text{c } \epsilon \text{ a } \text{a } \text{t}) = \text{"cat"}$$
4. The probability of target string $\mathbf{l}$ is the sum over all possible alignment paths that collapse into $\mathbf{l}$:
   $$P(\mathbf{l} | \mathbf{X}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} \prod_{t=1}^T P(\pi_t | \mathbf{x}_t)$$
5. The model is trained by minimizing the negative log-likelihood:
   $$\mathcal{L}_{CTC} = -\ln P(\mathbf{l} | \mathbf{X})$$

### 6.4 Linguistic Morphologies of Indic Scripts
Unlike Latin alphabets (which are strictly linear and left-to-right), Indic scripts (such as Devanagari for Hindi, Tamil, and Malayalam) are **Abugidas** (alphasyllabaries):

1. **Base Consonants**: Every consonant has an inherent vowel sound (usually short /a/).
2. **Dependent Vowel Signs (Matras)**: When a vowel other than /a/ follows a consonant, it attaches as a modifier glyph (Matra) that may visually render:
   - *After* the consonant: Hindi क + ा = का
   - *Before* the consonant: Hindi क + ि = कि (the matra appears to the left of the letter, even though it is pronounced after it)
   - *Above* the consonant: Hindi क + े = के
   - *Below* the consonant: Hindi क + ु = कु
3. **Conjunct Consonants (Samyuktaksharas)**: When two consonants occur without an intervening vowel, they fuse into a complex ligature:
   - Hindi: क् + ष = क्ष
   - Malayalam: ക + ഷ = ക്ഷ
4. **Unicode Normalization Form C (NFC)**:
   In raw text extraction, a character can be stored in two ways:
   - Composed (NFC): A single precomposed code point (`U+0915` + `U+093F`).
   - Decomposed (NFD): Multiple separate base characters and combining marks.
   Drishti automatically enforces `unicodedata.normalize('NFC', text)` to ensure all Indic text is canonicalized before being passed to the speech synthesizer.

---

# 7. Vision-Language Models (VLM) for Scene Narration

### 7.1 Architecture of Edge Vision-Language Models
When the user requests a comprehensive scene description (*"What is around me?"*), Drishti activates a compact Vision-Language Model (SmolVLM / Moondream2).

The model consists of three core components:

```
[ Input Camera Image (384x384) ]
               │
               ▼
   [ Vision Transformer (ViT) ]
   (Decomposes into 16x16 patches)
               │
               ▼ Visual Token Embeddings V \in \mathbb{R}^{N x D_v}
   [ Multimodal Linear Projector ]
               │
               ▼ Projected Tokens \in \mathbb{R}^{N x D_{text}}
   [ Autoregressive Decoder LLM ] ◄─── System Navigational Prompt
               │
               ▼
[ 2-Sentence Navigational Summary ]
```

1. **Vision Transformer (ViT)**: Decomposes the input image $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$ into non-overlapping spatial patches of size $P \times P$ ($14 \times 14$ or $16 \times 16$). Each patch is flattened and linearly projected into a vector of dimension $D_v$.
2. **Projector**: A lightweight Multi-Layer Perceptron (MLP) or cross-attention layer maps the visual tokens from vision dimension $D_v$ into the language model's latent embedding space $D_{text}$.
3. **Autoregressive Language Decoder**: A decoder-only transformer model (e.g., SmolLM2 with 500M parameters). It treats visual tokens as an image prompt prefix, attending across both visual tokens and text tokens using causal multi-head self-attention.

### 7.2 The Autoregressive Generation Loop & KV Caching
The language model generates output text one token at a time:
$$P(w_{t+1} | w_1, \dots, w_t, \mathbf{V}) = \text{Softmax}\left( \mathbf{W}_{vocab} \mathbf{h}_t \right)$$

In naive generation, computing self-attention requires re-calculating Key ($\mathbf{K}$) and Value ($\mathbf{V}$) matrices for all previous tokens at every step, resulting in $O(T^2)$ computational complexity.

**Key-Value (KV) Caching** stores previously computed keys and values in memory:
$$\mathbf{K}_{cached} = [\mathbf{K}_{cached}, \mathbf{K}_{new}], \quad \mathbf{V}_{cached} = [\mathbf{V}_{cached}, \mathbf{V}_{new}]$$
This reduces per-token decoding latency to $O(T)$, allowing SmolVLM to generate descriptions at $8\text{ to } 12\text{ tokens/second}$ on the Raspberry Pi 5.

### 7.3 Navigational Prompt Conditioning
General VLMs generate verbose, descriptive prose: *"A brightly lit living room with a carpet on the floor, some paintings on the wall..."*

This verbosity is dangerous for navigation. Drishti conditions the VLM with a strict system prompt:
```
System: You are Drishti, an assistive navigator for a visually impaired user.
Rules:
1. Identify immediate obstacles and their metric distance first.
2. State directional layout (left, right, straight ahead).
3. Warn of physical hazards (steps, drop-offs, low clearance).
4. Strictly limit response to 2 actionable sentences.
```

---

# 8. Psychoacoustics & 3D Spatial Audio Synthesis (ITD & ILD)

### 8.1 The Duplex Theory of Spatial Hearing
How does the human brain determine whether a sound originates from the left, right, in front, or behind, using only two ears?

In 1907, Lord Rayleigh formulated the **Duplex Theory**, demonstrating that human horizontal sound localization relies on two distinct physical phenomena:
1. **Interaural Time Difference (ITD)** for low frequencies ($f < 1.5\text{ kHz}$).
2. **Interaural Level Difference (ILD)** for high frequencies ($f > 1.5\text{ kHz}$).

```
                             Front (0 rad)
                                  ^
                                  |
               Acoustic Wave      |
                  \               |
                   \              |
                    \             |
                     v            |
              [ Left Ear ]    [ Head ]    [ Right Ear ]
                  (Near)       (Shadow)      (Far)
                     |                       |
                     +---- ITD (\Delta t) ---+
                     +---- ILD (\Delta L) ---+
```

### 8.2 Interaural Time Difference (ITD)
Because sound travels through air at a finite speed ($c \approx 343\text{ m/s}$ at $20^\circ\text{C}$), a sound wave originating from an azimuth angle $\theta$ reaches the closer (ipsilateral) ear before the farther (contralateral) ear.

Using **Woodworth's Spherical Head Model** (modeling the human head as a sphere of radius $r \approx 0.0875\text{ m}$):
- For a sound in the far field, the extra distance the acoustic wave must travel to reach the farther ear is:
  $$\Delta d(\theta) = r (\theta + \sin\theta)$$

The time delay between the two ears is:
$$\Delta t_{ITD}(\theta) = \frac{r}{c} (\theta + \sin\theta)$$

Let us calculate the maximum physical delay when sound arrives directly from the right ($\theta = 90^\circ = \frac{\pi}{2} \text{ rad}$):
The human auditory brainstem (specifically the Medial Superior Olive) detects interaural phase shifts as small as **$10 \;\mu\text{s}$**, allowing humans to localize azimuth with $1^\circ$ precision.

### 8.3 Continuous Fractional Sample Delay Interpolation
In digital discrete-time processing with sampling rate $f_s = 44,100\text{ Hz}$, the sampling interval is $T_s = \frac{1}{f_s} \approx 22.67 \;\mu\text{s}$.
An ITD delay of $\tau = 655.4 \;\mu\text{s}$ corresponds to a non-integer sample shift:
$$D = \tau \times f_s = 0.0006554 \times 44,100 \approx 28.903 \text{ samples}$$

Rounding $D$ to the nearest integer ($29$ samples) introduces a phase discretization error of up to $\pm 11.3 \;\mu\text{s}$—exceeding the human brainstem's $10 \;\mu\text{s}$ threshold and causing audible angular localization jitter or "jumping" as an obstacle moves continuously across the field of view.

Drishti implements **continuous sub-sample fractional delay filtering** via linear interpolation. Decomposing the total delay into integer and fractional components:
$$D = D_{\text{int}} + d, \quad D_{\text{int}} = \lfloor D \rfloor, \quad 0 \le d < 1$$

The delayed continuous discrete-time output is evaluated at arbitrary fractional indices:
$$y[n] = x(n - D) = (1 - d) \cdot x[n - D_{\text{int}}] + d \cdot x[n - D_{\text{int}} - 1]$$

This continuous formulation yields sub-sample phase precision, smooth auditory motion trajectories without discretization artifacts, and negligible computational overhead ($O(N)$ vectorized linear interpolation).

### 8.4 Interaural Level Difference (ILD)
The speed of sound in air is $c = f \lambda$. At $1.5\text{ kHz}$, the acoustic wavelength is:
$$\lambda = \frac{343\text{ m/s}}{1500\text{ Hz}} \approx 0.228 \text{ m} = 22.8 \text{ cm}$$

Since the human head diameter ($2r \approx 17.5\text{ cm}$) is comparable to or larger than acoustic wavelengths above $1.5\text{ kHz}$, the head acts as a solid physical obstacle, casting an **acoustic shadow**.

The sound wave diffracts poorly around the skull, attenuating the acoustic sound pressure level at the farther ear:
$$\text{ILD}(\theta) = 20 \log_{10}\left( \frac{P_{ipsilateral}}{P_{contralateral}} \right) \approx \gamma(f) \cdot \sin\theta \quad [\text{dB}]$$
where $\gamma(f)$ scales from $6\text{ dB}$ at $2\text{ kHz}$ to over $20\text{ dB}$ at $8\text{ kHz}$.

### 8.4 Distance Sonification & Proximity Modulation
To intuitively convey metric distance $d$ without speaking words:
1. **Pulse Repetition Frequency (PRF)**: Analogous to a submarine sonar ping, the pulse rate accelerates as the user approaches an obstacle:
   $$f_{pulse}(d) = \text{clamp}\left( \frac{k_{rate}}{d}, \; 1.0\text{ Hz}, \; 12.0\text{ Hz} \right)$$
   At 3 meters, the user hears 1 ping per second; at 0.5 meters, the user hears 10 rapid pings per second.
2. **Frequency Pitch Shifting**:
   - Caution Zone ($2.0\text{ m} \le d \le 4.0\text{ m}$): $440\text{ Hz}$ tone (musical note A4).
   - Warning Zone ($1.0\text{ m} \le d < 2.0\text{ m}$): $660\text{ Hz}$ tone (musical note E5).
   - Critical Danger Zone ($d < 1.0\text{ m}$): $880\text{ Hz}$ tone (musical note A5).

### 8.5 Bone-Conduction Transducers
Standard in-ear or over-ear headphones block the ear canal (*attenuation of ambient sound*), which is dangerous for visually impaired individuals who rely on natural environmental acoustics (hearing oncoming traffic, footsteps, echoes from walls).

Drishti mandates **bone-conduction transducers** resting on the temporal bones (zygomatic arch) directly in front of the ears. Mechanical acoustic vibrations transmit sound directly to the inner cochlea through cranial bone conduction, leaving the ear canal **100% open to ambient surroundings**.

---

# 9. Edge Privacy & Caregiver Telemetry Architecture

### 9.1 The Ethical Necessity of Edge-Side Privacy
Wearable cameras capture everyone and everything in the user's surroundings—including bystanders' faces, vehicle license plates, and private documents. Transmitting unredacted video streams to cloud servers creates grave privacy and surveillance violations.

Drishti enforces an **offline-first, zero-cloud default policy**:
- All continuous detection, OCR, and VLM operations execute locally on-device.
- Zero video frames are ever recorded or transmitted during normal operation.

### 9.2 Real-Time Gaussian Face Redaction
When the user explicitly triggers an opt-in emergency SOS or caregiver location check-in, the snapshot passes through an on-device privacy filter before leaving the device.

Detected face regions are convolved with a 2D Gaussian kernel:
$$I_{redacted}(x, y) = \sum_{i=-K}^{K} \sum_{j=-K}^{K} I(x+i, y+j) \cdot G_\sigma(i, j)$$
where the discrete Gaussian kernel is:
$$G_\sigma(i, j) = \frac{1}{2\pi \sigma^2} \exp\left( -\frac{i^2 + j^2}{2\sigma^2} \right)$$

With kernel window radius $K = 15$ and variance $\sigma = 7$, individual facial identities are mathematically and irreversibly destroyed while preserving high-level context (e.g., posture, ambient room illumination).

### 9.3 Mosquitto MQTT Protocol & Security Architecture
Drishti communicates with the caregiver Flutter companion app using the lightweight **MQTT (Message Queuing Telemetry Transport)** protocol:

```
[ Drishti Wearable ] ──(TLS 1.3 / Port 8883)──► [ Mosquitto MQTT Broker ] ──► [ Caregiver App ]
```

1. **Transport Layer Security (TLS 1.3)**: All MQTT payloads are encrypted using asymmetric RSA-2048 or ECDSA certificates with AES-256-GCM symmetric transport encryption.
2. **Quality of Service (QoS Level 1)**: Emergency alerts use QoS 1 (*At least once delivery*). The client buffers the alert until the broker returns a packet acknowledgment (`PUBACK`), guaranteeing that emergency distress calls are not dropped during spotty cellular coverage.
3. **Last Will and Testament (LWT)**: When Drishti connects to the MQTT broker, it registers an LWT payload on topic `drishti/device/{id}/status`. If the wearable battery dies or the cellular connection abruptly drops, the broker automatically publishes an `OFFLINE_EMERGENCY` notification to the caregiver.
