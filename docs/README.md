# Drishti documentation index

Start here when resuming work on another computer or with another AI coding session.

| Document | Purpose |
|---|---|
| [`CURRENT_SYSTEM.md`](CURRENT_SYSTEM.md) | What runs today, end-to-end data flow, and honest limitations |
| [`LAPTOP_WORKFLOW.md`](LAPTOP_WORKFLOW.md) | Laptop + ESP32/camera setup, Git workflow, and AI-session handoff |
| [`LATENCY.md`](LATENCY.md) | Current measurements, latency risks, profiling, and optimization order |
| [`INDUSTRY_ROADMAP.md`](INDUSTRY_ROADMAP.md) | Proposed product hardware, safety architecture, regulatory process, and validation |
| [`NEXT_STEPS.md`](NEXT_STEPS.md) | Ordered engineering backlog and exit criteria |
| [`../technical_details.md`](../technical_details.md) | Module-level technical contracts |
| [`../troubleshoot.md`](../troubleshoot.md) | Known failures and diagnostics |
| [`../TEAM_HANDOVER.md`](../TEAM_HANDOVER.md) | Short engineering handover |
| [`../evaluation/README.md`](../evaluation/README.md) | Dataset and metric protocol |
| [`../firmware/drishti_tof/README.md`](../firmware/drishti_tof/README.md) | ESP32/VL53L0X firmware protocol |

## Status vocabulary

- **Implemented:** present in code and covered by automated tests.
- **Verified locally:** exercised on the current development laptop.
- **Hardware-verified:** measured on the exact assembled camera, range sensor, audio device, and compute target.
- **Validated:** tested against a predefined protocol and acceptance criteria with retained evidence.
- **Product-ready:** released under an appropriate quality, risk, cybersecurity, manufacturing, and post-market process.

Drishti is currently an implemented and locally tested research prototype. It is not hardware-validated or product-ready.

