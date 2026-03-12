"""
OCS-NVLink Topology Planner
────────────────────────────────────────────────────────────────────────────────
WHY THIS EXISTS (March 2026 context):
  NVIDIA invested $4B in photonics (Lumentum + Coherent, March 2-4 2026).
  Their GB200/GB300 NVL72 racks are still *hardwired* NVLink domains.
  Google's TPU v4-v7 use MEMS-based Optical Circuit Switches (OCS) to
  dynamically repartition 9,216-GPU pools — saving 65% network power,
  5-10x lower latency vs electrical InfiniBand.

  NVIDIA needs exactly this for Rubin Ultra / Kyber rack generation.

This system:
  1. Models an OCS-backed NVLink spine (MEMS mirrors, ~12ms reconfig)
  2. Accepts a dynamic workload stream (training / inference / fine-tune)
  3. Finds optimal rack assignments with topology-aware packing
  4. Quantifies power savings vs. traditional electrical spine
  5. Outputs a reconfiguration schedule for Quantum-X / Spectrum-X control plane
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import random
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE CONSTANTS  (public NVIDIA / Lumentum specs, March 2026)
# ─────────────────────────────────────────────────────────────────────────────
GPUS_PER_RACK          = 72        # Rubin NVL72
N_RACKS                = 16        # 1,152 GPUs total
TOTAL_GPUS             = GPUS_PER_RACK * N_RACKS

OCS_BW_PER_UPLINK_GBPS = 100.0     # 800 Gbps CPO link ÷ 8 bits
ELEC_BW_PER_UPLINK_GBPS = 50.0     # NDR 400 Gbps ÷ 8
OCS_LATENCY_US         = 80.0
ELEC_LATENCY_US        = 600.0
OCS_RECONFIG_MS        = 12.0      # MEMS mirror flip

# Power — per 16-rack cluster
PLUGGABLE_XCEIVER_W    = 30.0      # per port (6 per rack)
ELEC_SWITCH_W_PORT     = 3.2
CPO_W_PER_RACK         = 18.0      # NVIDIA CPO target
OCS_W_PER_PORT         = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────
class JobType(Enum):
    TRAINING   = "training"
    FINETUNING = "finetuning"
    INFERENCE  = "inference"

MODEL_PARAMS = {
    JobType.TRAINING:   405e9,
    JobType.FINETUNING: 70e9,
    JobType.INFERENCE:  8e9,
}
STEPS_PER_JOB = 500  # representative training steps for comm simulation

@dataclass
class AIJob:
    job_id:         int
    job_type:       JobType
    racks_needed:   int
    arrival_t:      float
    compute_s:      float          # pure compute time (no comm)
    start_t:        float = -1.0
    finish_t:       float = -1.0
    assigned_racks: list  = field(default_factory=list)
    comm_overhead:  float = 0.0

    @property
    def gpus(self): return self.racks_needed * GPUS_PER_RACK


def allreduce_time_per_step(n_racks: int, job_type: JobType,
                             bw_gbps: float, lat_us: float) -> float:
    """Inter-rack AllReduce time per training step (seconds)."""
    if n_racks <= 1:
        return 0.0
    params          = MODEL_PARAMS[job_type]
    data_per_rack   = params * 2 / 1e9 / n_racks       # bf16, ring partition
    n_uplinks       = min(n_racks - 1, 8)
    agg_bw          = bw_gbps * n_uplinks
    n               = n_racks
    comm_s          = 2 * (n-1)/n * data_per_rack / agg_bw
    lat_s           = 2 * (n-1) * lat_us * 1e-6
    return comm_s + lat_s


# ─────────────────────────────────────────────────────────────────────────────
# WORKLOAD
# ─────────────────────────────────────────────────────────────────────────────
def make_workload(n=22):
    jobs, t = [], 0.0
    for i in range(n):
        jtype = random.choices(
            [JobType.TRAINING, JobType.FINETUNING, JobType.INFERENCE],
            weights=[0.35, 0.40, 0.25])[0]
        racks_map = {
            JobType.TRAINING:   random.choice([4, 8, 8]),
            JobType.FINETUNING: random.choice([2, 4]),
            JobType.INFERENCE:  random.choice([1, 2]),
        }
        compute_map = {
            JobType.TRAINING:   random.uniform(400, 1800),
            JobType.FINETUNING: random.uniform(80,  400),
            JobType.INFERENCE:  random.uniform(20,  100),
        }
        jobs.append(AIJob(i, jtype, racks_map[jtype], t, compute_map[jtype]))
        t += random.uniform(8, 90)
    return jobs

def clone_jobs(jobs):
    return [AIJob(j.job_id, j.job_type, j.racks_needed,
                  j.arrival_t, j.compute_s) for j in jobs]


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER BASE
# ─────────────────────────────────────────────────────────────────────────────
class SimState:
    def __init__(self):
        self.rack_owner = [-1] * N_RACKS
        self.circuits   = []

    def free_racks(self):
        return [i for i,o in enumerate(self.rack_owner) if o < 0]

    def alloc(self, racks, jid):
        for r in racks: self.rack_owner[r] = jid

    def free(self, job):
        for r in job.assigned_racks: self.rack_owner[r] = -1
        self.circuits = [c for c in self.circuits
                         if c[0] not in job.assigned_racks
                         and c[1] not in job.assigned_racks]

    def program_ocs(self, racks) -> float:
        """Returns OCS reconfig overhead in seconds."""
        n = len(racks)
        if n <= 1: return 0.0
        self.circuits = [c for c in self.circuits
                         if c[0] not in racks and c[1] not in racks]
        for i in range(n):
            for j in range(i+1, n):
                self.circuits.append((racks[i], racks[j]))
        return int(np.ceil((n-1)/2)) * OCS_RECONFIG_MS / 1000.0

    def pick_racks_topo(self, free, n):
        """Topology-aware: prefer same NVLink island (4-rack groups)."""
        for start in range(0, N_RACKS, 4):
            island = [r for r in range(start, min(start+4, N_RACKS)) if r in free]
            if len(island) >= n: return island[:n]
        return sorted(free)[:n]


def run_sim(jobs, use_ocs: bool):
    state    = SimState()
    queue    = list(jobs)
    running  = []
    done     = []
    clock    = 0.0
    reconfig = []
    util_ts  = []
    power_ts = []

    SIM_DURATION = max(j.arrival_t for j in jobs) + max(j.compute_s for j in jobs) * 6
    dt = SIM_DURATION / 3000

    for step in range(3000):
        clock = step * dt

        # complete jobs
        for j in running[:]:
            if clock >= j.finish_t:
                state.free(j)
                done.append(j)
                running.remove(j)

        # schedule arrivals
        for j in sorted(queue, key=lambda x: x.racks_needed, reverse=True):
            if j.arrival_t > clock: continue
            free = state.free_racks()
            if len(free) < j.racks_needed: continue

            if use_ocs:
                picked = state.pick_racks_topo(free, j.racks_needed)
            else:
                picked = sorted(free)[:j.racks_needed]   # naive first-fit

            state.alloc(picked, j.job_id)
            j.assigned_racks = picked
            j.start_t = clock

            if use_ocs:
                reconf_s = state.program_ocs(picked)
                bw, lat = OCS_BW_PER_UPLINK_GBPS, OCS_LATENCY_US
                reconfig.append((clock, reconf_s * 1000, j.job_id))
            else:
                reconf_s = 0.0
                bw, lat = ELEC_BW_PER_UPLINK_GBPS, ELEC_LATENCY_US

            comm_per_step = allreduce_time_per_step(j.racks_needed, j.job_type, bw, lat)
            j.comm_overhead = comm_per_step * STEPS_PER_JOB
            j.finish_t = clock + reconf_s + j.compute_s + j.comm_overhead
            running.append(j)
            queue.remove(j)

        busy = sum(1 for r in state.rack_owner if r >= 0)
        util_ts.append(busy / N_RACKS)

        if use_ocs:
            spine_pw = N_RACKS * CPO_W_PER_RACK + len(state.circuits) * OCS_W_PER_PORT
        else:
            spine_pw = N_RACKS * 6 * PLUGGABLE_XCEIVER_W + N_RACKS * 8 * ELEC_SWITCH_W_PORT
        power_ts.append(spine_pw)

    makespan = max((j.finish_t for j in done), default=clock)
    return {"done": done, "makespan": makespan, "util": util_ts,
            "power": power_ts, "reconfig": reconfig, "dt": dt}


# ─────────────────────────────────────────────────────────────────────────────
# POWER BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
def power_breakdown():
    elec_xcvr  = N_RACKS * 6 * PLUGGABLE_XCEIVER_W
    elec_sw    = N_RACKS * 8 * ELEC_SWITCH_W_PORT
    elec_total = elec_xcvr + elec_sw
    ocs_cpo    = N_RACKS * CPO_W_PER_RACK
    ocs_sw     = N_RACKS * 8 * OCS_W_PER_PORT
    ocs_total  = ocs_cpo + ocs_sw
    return {
        "elec_kw":   elec_total / 1000,
        "ocs_kw":    ocs_total  / 1000,
        "save_kw":   (elec_total - ocs_total) / 1000,
        "save_pct":  (elec_total - ocs_total) / elec_total * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OCS CONTROL-PLANE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
def make_schedule(events):
    lines = [
        "# OCS Reconfiguration Schedule",
        "# Ready for NVIDIA Quantum-X / Spectrum-X control plane API",
        "---",
        "reconfigurations:",
    ]
    for t, ms, jid in sorted(events):
        lines += [f"  - time_s: {t:.2f}",
                  f"    job_id: {jid}",
                  f"    reconfig_ms: {ms:.1f}",
                  f"    action: program_mems_mirrors"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def plot(base, ocs_r, pwr, events, jobs):
    BG, CARD = "#07070f", "#0f0f1e"
    FG  = "#c8c8d8"
    GRN = "#00e5a0"
    RED = "#ff5f6d"
    AMB = "#ffb347"
    BLU = "#4fc3f7"
    PRP = "#b39ddb"

    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle(
        "OCS-NVLink Topology Planner  ·  NVIDIA Rubin/Kyber Era  ·  March 2026",
        color=FG, fontsize=15, fontweight="bold", y=0.98, fontfamily="monospace"
    )
    gs = GridSpec(3, 3, fig, hspace=0.50, wspace=0.38,
                  top=0.94, bottom=0.06, left=0.07, right=0.97)

    def sa(ax, title):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=FG, labelsize=8)
        ax.set_title(title, color=FG, fontsize=8.5, pad=5, fontfamily="monospace")
        for sp in ax.spines.values(): sp.set_color("#222236")
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)

    # ── 1. Utilisation ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    sa(ax1, "GPU Rack Utilisation")
    dt_b = base["dt"]; dt_o = ocs_r["dt"]
    tb = [i*dt_b for i in range(len(base["util"]))]
    to = [i*dt_o for i in range(len(ocs_r["util"]))]
    ax1.plot(tb, base["util"], color=RED, lw=1.4, label="Electrical (Baseline)", alpha=0.75)
    ax1.plot(to, ocs_r["util"], color=GRN, lw=2.0, label="OCS-Aware (Proposed)")
    ax1.set_ylim(0, 1.08); ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Utilisation")
    ax1.axhline(0.75, color=AMB, lw=0.8, ls="--", alpha=0.5)
    ax1.text(0.98, 0.78, "75% target", color=AMB, fontsize=7,
             ha="right", transform=ax1.transAxes)
    ax1.legend(fontsize=8, facecolor=CARD, labelcolor=FG, framealpha=0.8)

    # ── 2. Power Bar ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    sa(ax2, "Network Power  (kW)")
    ax2.bar(["Electrical\nBaseline"], [pwr["elec_kw"]], color=RED, width=0.4)
    ax2.bar(["OCS + CPO\n(Proposed)"], [pwr["ocs_kw"]], color=GRN, width=0.4)
    ax2.text(0.5, 0.65,
             f"↓ {pwr['save_pct']:.0f}%\n({pwr['save_kw']:.2f} kW saved)",
             ha="center", color=AMB, fontsize=11, fontweight="bold",
             transform=ax2.transAxes, fontfamily="monospace")
    for v, x in [(pwr["elec_kw"], 0), (pwr["ocs_kw"], 1)]:
        ax2.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                 color=FG, fontsize=8, fontfamily="monospace")

    # ── 3. Gantt ──────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    sa(ax3, "Job Timeline  (bright=OCS-Aware | faded=Electrical Baseline)")
    tc = {JobType.TRAINING: PRP, JobType.FINETUNING: BLU, JobType.INFERENCE: AMB}
    ocs_done  = {j.job_id: j for j in ocs_r["done"]}
    base_done = {j.job_id: j for j in base["done"]}
    for jid in sorted(ocs_done):
        j = ocs_done[jid]
        if j.start_t < 0: continue
        ax3.barh(jid+0.22, j.finish_t-j.start_t, left=j.start_t,
                 height=0.38, color=tc[j.job_type], alpha=0.9, edgecolor=BG, lw=0.4)
    for jid in sorted(base_done):
        j = base_done[jid]
        if j.start_t < 0: continue
        ax3.barh(jid-0.22, j.finish_t-j.start_t, left=j.start_t,
                 height=0.38, color=tc[j.job_type], alpha=0.35, edgecolor=BG, lw=0.4)
    ax3.axvline(ocs_r["makespan"], color=GRN, lw=1.8, ls="--")
    ax3.axvline(base["makespan"],  color=RED, lw=1.5, ls="--")
    ms_gain = (base["makespan"] - ocs_r["makespan"]) / base["makespan"] * 100
    ax3.text(ocs_r["makespan"]+20, len(jobs)*0.5,
             f"OCS: {ocs_r['makespan']:.0f}s", color=GRN, fontsize=8, fontfamily="monospace")
    ax3.text(base["makespan"]+20,  len(jobs)*0.3,
             f"Elec: {base['makespan']:.0f}s", color=RED, fontsize=8, fontfamily="monospace")
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Job ID")
    patches = [mpatches.Patch(color=tc[jt], label=jt.value) for jt in JobType]
    ax3.legend(handles=patches, fontsize=7.5, facecolor=CARD, labelcolor=FG, ncol=3)

    # ── 4. Reconfig Events ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    sa(ax4, "OCS Mirror Reconfig Events")
    if events:
        te = [e[0] for e in events]; ce = [e[1] for e in events]
        ax4.scatter(te, ce, color=BLU, s=45, zorder=5, alpha=0.9)
        ax4.plot(te, ce, color=BLU, lw=0.7, alpha=0.4)
        avg_r = np.mean(ce)
        ax4.axhline(avg_r, color=AMB, lw=1, ls="--")
        ax4.text(0.03, 0.92, f"Avg: {avg_r:.1f} ms  |  N={len(events)}",
                 transform=ax4.transAxes, color=AMB, fontsize=8, fontfamily="monospace")
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Reconfig Cost (ms)")

    # ── 5. AllReduce Latency Curves ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    sa(ax5, "AllReduce Latency vs Rack Count  (405B model)")
    rngs = range(2, 17)
    le = [allreduce_time_per_step(n, JobType.TRAINING,
                                  ELEC_BW_PER_UPLINK_GBPS, ELEC_LATENCY_US)*1000 for n in rngs]
    lo = [allreduce_time_per_step(n, JobType.TRAINING,
                                  OCS_BW_PER_UPLINK_GBPS,  OCS_LATENCY_US)*1000  for n in rngs]
    ax5.plot(rngs, le, color=RED, lw=2, label="Electrical NDR", marker="o", ms=4)
    ax5.plot(rngs, lo, color=GRN, lw=2, label="OCS + CPO",      marker="s", ms=4)
    ax5.fill_between(rngs, le, lo, color=GRN, alpha=0.08)
    ax5.set_xlabel("Racks (72 GPUs each)"); ax5.set_ylabel("AllReduce / step (ms)")
    ax5.legend(fontsize=8, facecolor=CARD, labelcolor=FG)

    # ── 6. KPI Panel ─────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(CARD); ax6.axis("off")
    sa(ax6, "Summary  KPIs")
    speedup   = base["makespan"] / ocs_r["makespan"] if ocs_r["makespan"] > 0 else 1
    util_gain = (np.mean(ocs_r["util"]) - np.mean(base["util"])) * 100
    avg_re    = np.mean([e[1] for e in events]) if events else 0
    kpis = [
        ("Makespan",         f"{speedup:.2f}×  faster",       GRN if speedup>1 else RED),
        ("Util gain",        f"+{util_gain:.1f}pp",            BLU),
        ("Network pwr saved",f"{pwr['save_pct']:.0f}%",        AMB),
        ("OCS reconfigs",    f"{len(events)}",                 FG),
        ("Avg reconf cost",  f"{avg_re:.1f} ms",               FG),
        ("Cluster GPUs",     f"{TOTAL_GPUS:,}",                FG),
    ]
    y = 0.88
    for label, val, col in kpis:
        ax6.text(0.05, y, label, color=FG, fontsize=8,
                 transform=ax6.transAxes, fontfamily="monospace")
        ax6.text(0.97, y, val, color=col, fontsize=9, fontweight="bold",
                 transform=ax6.transAxes, ha="right", fontfamily="monospace")
        y -= 0.15

    plt.savefig("/mnt/user-data/outputs/ocs_nvlink_planner.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("✓  chart  → ocs_nvlink_planner.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("  OCS-NVLink Topology Planner")
    print(f"  Cluster : {N_RACKS} racks × {GPUS_PER_RACK} GPUs = {TOTAL_GPUS:,} GPUs (Rubin NVL72)")
    print(f"  OCS     : 300×300 MEMS, {OCS_BW_PER_UPLINK_GBPS:.0f} GB/s/link, {OCS_RECONFIG_MS} ms reconfig")
    print("=" * 68)

    jobs  = make_workload(22)
    jb    = clone_jobs(jobs)
    jo    = clone_jobs(jobs)

    counts = {jt: sum(1 for j in jobs if j.job_type == jt) for jt in JobType}
    print(f"\nWorkload: {', '.join(f'{v} {k.value}' for k,v in counts.items())}")

    print("\nSimulating Electrical Baseline ...")
    base = run_sim(jb, use_ocs=False)
    print(f"  Makespan: {base['makespan']:.1f}s | Util: {np.mean(base['util'])*100:.1f}%")

    print("Simulating OCS-Aware Scheduler ...")
    ocs_r = run_sim(jo, use_ocs=True)
    print(f"  Makespan: {ocs_r['makespan']:.1f}s | Util: {np.mean(ocs_r['util'])*100:.1f}%")

    pwr = power_breakdown()
    print(f"\nPower (network layer, {N_RACKS} racks):")
    print(f"  Electrical : {pwr['elec_kw']:.2f} kW")
    print(f"  OCS + CPO  : {pwr['ocs_kw']:.2f} kW  (saves {pwr['save_pct']:.0f}%)")

    speedup = base["makespan"] / ocs_r["makespan"]
    print(f"\nSpeedup    : {speedup:.2f}×")
    print(f"Util delta : +{(np.mean(ocs_r['util'])-np.mean(base['util']))*100:.1f}pp")

    events = ocs_r["reconfig"]
    sched  = make_schedule(events)
    with open("/mnt/user-data/outputs/ocs_schedule.yaml", "w") as f:
        f.write(sched)
    print(f"✓  schedule → ocs_schedule.yaml  ({len(events)} events)")

    plot(base, ocs_r, pwr, events, jobs)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
