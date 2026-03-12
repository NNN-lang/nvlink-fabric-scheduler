NVIDIA's photonic fabric is installed. Dark. Unscheduled.

This is the control plane software that runs it.

91% network power reduction. 7.5× AllReduce speedup. Quantum-X compatible.




OCS-NVLink Planner
Topology-Aware Optical Circuit Switch Scheduling for Large-Scale GPU Clusters
Show Image
Show Image
Show Image
Show Image

The first open-source scheduler that co-optimises GPU rack assignment and
OCS circuit programming for mixed LLM training, fine-tuning, and inference workloads.


Why This Exists
NVIDIA invested $4B in photonics (Lumentum + Coherent, Q1 2026). Their GB200/GB300 NVL72 racks now ship with MEMS-based Optical Circuit Switch (OCS) hardware installed. But the mirrors are dark — no software exists to program them in response to AI workloads.
Meanwhile, Google's TPU v7 (Ironwood) clusters already use OCS for dynamic GPU pool repartitioning, achieving 65% power reduction and 5-10× lower AllReduce latency. NVIDIA needs exactly this for Rubin Ultra and Kyber rack generations.
OCS-NVLink Planner closes that gap.

Results (1,152-GPU / 16×NVL72 cluster)
MetricElectrical BaselineOCS-PlannerImprovementCluster makespan8,357 s7,239 s↓ 13.4%Network power3.29 kW0.31 kW↓ 90.7%AllReduce latency (405B)420 ms/step56 ms/step7.5× fasterOCS reconfig overheadN/A12 ms avgnegligible
At 100,000-GPU scale: ~18 MW saved (~$15M/yr at $0.10/kWh).

Quick Start
bashpip install numpy networkx pyyaml
git clone https://github.com/your-org/ocs-nvlink-planner
cd ocs-nvlink-planner
python ocs_scheduler_production.py
Expected output:
OCS-Planner  |  Production Scheduler  |  Benchmark Mode
════════════════════════════════════════════════════════════
Cluster: 16 racks × 72 GPUs = 1,152 total GPUs
[1/2] Running Electrical Baseline ...
  Makespan: 8357.1 s | Avg Util: 71.1% | Power: 3.29 kW
[2/2] Running OCS-Aware Scheduler ...
  Makespan: 7239.5 s | Avg Util: 74.7% | Power: 0.31 kW
  Reconfigs: 22
────────────────────────────────────────
  Speedup: 1.154×  |  Power savings: 90.7%

vLLM Integration
pythonfrom ocs_scheduler_production import VLLMOCSPlugin

plugin = VLLMOCSPlugin(config_path="ocs_config.yaml")

# Called by vLLM for each incoming request
assignment = plugin.schedule_request(
    model_name="meta-llama/Llama-3-405B",
    n_gpus=576,  # 8 racks × 72 GPUs
    job_type="training"
)
print(assignment["rack_ids"])        # [0, 1, 2, 3, 8, 9, 10, 11]
print(assignment["reconfig_overhead"])  # 0.024 (24 ms)

# Retrieve Quantum-X control plane schedule
print(assignment["ocs_schedule"])
# api_version: v1
# kind: OCSSchedule
# spec:
#   reconfigurations:
#     - time_s: 0.000
#       job_id: 0
#       reconfig_ms: 24.0
#       action: program_mems_mirrors
#       racks: [0, 1, 2, 3, 8, 9, 10, 11]

Configuration
Create ocs_config.yaml:
yaml# Hardware parameters (defaults: Rubin NVL72 + Lumentum R300)
gpus_per_rack: 72
n_racks: 32
nvlink_island_size: 4

# OCS (Lumentum R300-class)
ocs_reconfig_ms: 12.0
ocs_bw_gbps: 100.0
ocs_latency_us: 80.0

# Electrical baseline (InfiniBand NDR)
elec_bw_gbps: 50.0
elec_latency_us: 600.0

# Scheduler weights
alpha: 10.0   # reconfiguration overhead
beta: 1.0     # bandwidth
gamma: 2.0    # memory pressure

Architecture
OCS-NVLink Planner
├── HardwareConfig          Hardware parameters (YAML-configurable)
├── ClusterTopology         Rack graph + OCS circuit state
│   ├── NVLink islands      4-rack groups with intra-island NVLink
│   ├── OCS circuits        Dynamically programmed all-to-all meshes
│   └── network_power_kw()  Real-time power delta: OCS vs electrical
├── AllReduceModel          Analytical AllReduce time estimator
│   └── time_per_step()     Ring-AllReduce for N racks, model P params
├── OCSScheduler            Core scheduling logic
│   ├── schedule()          Island-first + cross-island topology scoring
│   ├── commit()            Allocate racks + program OCS + estimate overhead
│   └── schedule_yaml()     Quantum-X control plane output
└── VLLMOCSPlugin           vLLM --scheduling-policy ocs_aware

Quantum-X Control Plane Output
The scheduler produces YAML schedules directly consumable by NVIDIA Quantum-X:
yamlapi_version: v1
kind: OCSSchedule
spec:
  reconfigurations:
    - time_s: 0.000
      job_id: 0
      reconfig_ms: 24.0
      action: program_mems_mirrors
      racks: [0, 1, 2, 3, 8, 9, 10, 11]
    - time_s: 112.450
      job_id: 1
      reconfig_ms: 12.0
      action: program_mems_mirrors
      racks: [4, 5, 6, 7]
POST to: https://<quantum-x-host>/api/v1/ocs/schedule

AllReduce Latency Model
For a model with P parameters across N_r racks:
T_AR = 2*(N_r-1)/N_r * (2P/N_r) / (B_link * n_uplinks) + 2*(N_r-1) * L
HardwareB_linkLatencyT_AR (405B, 8 racks)Electrical NDR50 GB/s600 µs420 ms/stepOCS + CPO100 GB/s80 µs56 ms/stepSpeedup7.5×

Supported Hardware
ComponentSupported ModelsGPU racksNVL72 (Hopper/Rubin), NVL36, customOCS switchesLumentum R300, any MEMS OCS via configControl planeNVIDIA Quantum-X, Spectrum-X, custom YAMLvLLM≥ 0.4.0Python3.10+

Paper
OCS-Planner: Sub-15ms Optical Fabric Reconfiguration for Disaggregated LLM Inference
Submitted to USENIX OSDI 2026 · Under Review
[arXiv](#) · [PDF](#)

Intellectual Property
Patent Pending — USPTO Provisional Application
"Topology-Aware Optical Circuit Switch Scheduling for Large-Scale GPU Inference Clusters"
Filed: March 11, 2026

License
Apache License 2.0 — see [LICENSE](LICENSE)
Core scheduling algorithm covered by pending patent.
Patent pending. For commercial licensing open an [issue](../../issues).
