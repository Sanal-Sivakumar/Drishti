# From prototype to an industry-ready assistive device

This is a future architecture and process proposal, not a statement of current capability. Final components must be selected through hazard analysis, supplier qualification, environmental testing, availability planning, and regulatory review.

## 1. Freeze intended use before choosing hardware

Define exactly what the product claims. “Provides optional awareness cues” and “guides independent navigation” create very different hazards and regulatory obligations. In India, CDSCO states that devices—including software—intended to assist with an injury or disability fall within the medical-device framework, and current medical-device software guidance should be reviewed with qualified regulatory counsel.

Do not advertise obstacle avoidance, safe path, drop-off protection, or independent mobility until those claims are validated and authorized for the target market.

## 2. Recommended product architecture

### Safety channel: deterministic and independently monitored

- **Primary depth:** calibrated stereo/global-shutter depth or an industrial depth module producing a dense map.
- **Near-field redundancy:** multizone ToF, not a single ray. A sensor such as VL53L8CX provides 8×8 zones, multitarget ranging, up to 60 Hz, and up to 4 m under specified conditions.
- **Downward coverage:** independently angled multizone ToF or compact depth sensor for curbs, steps and drop-offs.
- **Motion/orientation:** a calibrated IMU for head motion, sensor orientation, fall/impact context, and temporal stabilization.
- **Compute watchdog:** a small safety MCU supervising heartbeat, sensor freshness, buttons, battery, temperature, audio/haptic actuator health, and fail-safe state independently of Linux.
- **Critical output:** wired/open-ear stereo plus a separate haptic actuator. A critical event should remain perceivable if either audio or haptics fails.
- **Controls:** tactile power, repeat, mode and emergency buttons with distinguishable shapes; audible/haptic startup and fault codes.

The safety channel must not depend on a generative model, network connection, caregiver service, GPS, or a single RGB detector.

### Context channel: helpful but non-authoritative

- RGB detector and segmentation for semantic context.
- OCR and currency recognition on demand.
- VLM for optional scene explanation and user questions.
- Telemetry only with consent and explicit data minimization.

VLM output must remain lower priority, cancellable, freshness-bounded, and unable to assert an unverified path or distance.

## 3. Candidate development platforms

### Continue prototype work

- Laptop + current ESP32/VL53L0X for debugging serial protocols and UI.
- Raspberry Pi 5 for early wearable power/thermal experiments.

### Product-oriented prototypes

- **Compute Module 5** with eMMC and a custom carrier when power, size and cost dominate and optimized non-generative perception meets latency targets. Raspberry Pi publishes CM5 design/compliance material and offers 2–16 GB variants.
- **Jetson Orin Nano/NX production module** when accelerated detector, segmentation, stereo, and VLM workloads justify higher power and thermal cost. Use a production module/carrier—not a developer kit—in a product.
- **Depth module example:** OAK-D-class stereo modules combine global-shutter stereo and an IMU and can offload depth/vision. Verify minimum range, field of view, sunlight behavior, synchronization, thermal limits, and lifecycle before selection.

Do not select a board only from TOPS. Measure sustained p95/p99 latency, battery runtime, enclosure temperature, boot time, supply lifetime, secure-boot/update support, and recoverability.

## 4. Mechanical, electrical, and human factors

- Rigidly mount and factory-calibrate every camera/range sensor; detect mechanical shifts.
- Use keyed, locking connectors and strain relief.
- Design for rain/sweat/dust, drops, vibration, ESD, EMC, temperature, and direct sunlight appropriate to intended use.
- Add monitored battery protection, fuel gauge, safe charging, brownout handling, and a user-replaceable or serviceable battery strategy.
- Preserve ambient hearing. Open-ear/bone-conduction output needs per-user calibration and a hard acoustic level limit.
- Perform formative and summative usability work with blind/low-vision participants and orientation-and-mobility specialists.
- Make every failure understandable without a screen: “not ready” must have a distinct, persistent cue.

## 5. Software/product process

If the intended use makes Drishti a medical device, build the lifecycle and evidence set with professional regulatory support. Relevant frameworks include:

- [ISO 14971:2019](https://www.iso.org/standard/72704.html): device risk management across the lifecycle;
- [ISO 13485:2016](https://www.iso.org/iso-13485-medical-devices.html): medical-device quality management;
- [IEC 62304](https://webstore.iec.ch/en/publication/22794): medical-device software lifecycle;
- [IEC 62366-1](https://webstore.iec.ch/en/publication/21863): safety-related usability engineering;
- [IEC 81001-5-1](https://webstore.iec.ch/en/publication/63293): health-software cybersecurity lifecycle;
- [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework): voluntary AI risk-management practices;
- [CDSCO Medical Device Software guidance](https://www.cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/Guidance-document-on-Medical-Device-Software-under-MDR-2017.pdf): current Indian regulatory expectations;
- [India Digital Personal Data Protection Act, 2023](https://www.meity.gov.in/static/uploads/2024/02/Digital-Personal-Data-Protection-Act-2023.pdf): personal-data obligations, subject to current commencement rules and legal advice.

The applicable standards, classification, testing laboratories and submission path depend on intended use and target markets. This document is engineering guidance, not legal or certification advice.

## 6. Evidence required before release

- traceable user needs, system requirements and software requirements;
- hazard analysis, foreseeable misuse, FMEA/FTA and verified risk controls;
- versioned datasets with consent, provenance, labeling QA and demographic/environmental coverage;
- locked model artifacts, hashes, training records and change-control rules;
- unit, integration, hardware-in-loop, fault-injection and regression evidence;
- accuracy, latency, false-warning, missed-hazard and unsupported-claim results with confidence intervals;
- cybersecurity threat model, SBOM, signed updates, rollback, vulnerability intake and support policy;
- usability engineering and supervised human-factor evidence;
- environmental, electrical, EMC, battery, acoustic and mechanical test reports;
- manufacturing calibration, end-of-line test, serial traceability and complaint/CAPA process;
- post-market monitoring and field-update plan.

## 7. Product gate

Do not progress from supervised indoor research to unsupervised use merely because average accuracy or a demo looks good. Release requires the predefined residual-risk decision, regulatory authorization where applicable, manufacturing controls, and evidence that critical functions remain safe under sensor faults, stale data, thermal throttling, low battery, audio failure, glare, darkness, rain, network loss, and model uncertainty.

