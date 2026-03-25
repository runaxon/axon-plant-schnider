# Schnider Propofol PK/PD Simulator

Example repository for the [Axon](https://runaxon.com) tech blog series on profiling and optimizing physiological models in Python.

---

## What this is

A pure-Python implementation of the **Schnider (1998/1999) propofol pharmacokinetic/pharmacodynamic model** — the same model used in clinical Target Controlled Infusion (TCI) pumps — extended with a virtual patient cohort generator, a closed-loop PID controller, and a grid search optimizer.

The repository accompanies a blog article that walks through:
1. Implementing the physiological model
2. Building a virtual patient cohort via Latin Hypercube Sampling
3. Designing a PID controller and scoring it across the cohort
4. Profiling the bottleneck and transpiling the ODE hot path to C
5. Re-running the grid search with the accelerated simulator

---

## The model

Propofol distributes through the body via a **3-compartment mammillary PK model**: a central compartment (plasma) exchanging drug with a fast peripheral compartment (well-perfused tissue) and a slow peripheral compartment (fat/muscle). Drug effect in the brain is captured by a separate **effect-site compartment** linked to plasma via the equilibration rate constant $k_{e0}$.

The observable output is **BIS** (Bispectral Index) — a processed EEG signal scaled 0–100 that quantifies anesthetic depth. BIS is mapped from effect-site concentration via a sigmoidal Hill equation. The clinical target for general anesthesia is BIS 40–60.

All PK parameters are derived from patient demographics (age, weight, height, sex) using the Schnider population model.

![Single patient simulation: Cp, Ce, BIS, and infusion rate over 60 minutes](schnider_demo.png)

---

## Code structure

```
physio/
    core.py       # Patient, generic RK4 integrator, simulate(), SimulationResult
    schnider.py   # params_from_patient(), derivatives(), outputs()
    cohort.py     # Latin Hypercube Sampling cohort generator + JSON persistence

controller/
    pid.py        # Discrete-time PID with anti-windup and output clamping

demo_schnider.py  # Single patient: induction + maintenance + washout
eval_cohort.py    # Cohort evaluation harness + loss function
grid_search.py    # Grid search over PID gains + 3D loss landscape plot

cohorts/          # Persisted cohort JSON files
results/          # Grid search results
```

`derivatives()` in `physio/schnider.py` is the ODE right-hand side — a pure arithmetic function with no Python object overhead. It is the transpilation target for the C optimization stage.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Single patient demo

```bash
python demo_schnider.py
```

Simulates a 70 kg / 40 yr male receiving a standard induction bolus + 30 min maintenance + washout. Saves `schnider_demo.png`.

---

## Virtual cohort

Patient demographics are sampled via **Latin Hypercube Sampling** over the ranges below, which reflect the adult surgical population studied in Schnider (1998):

| Variable | Range |
|----------|-------|
| Age | 18 – 80 years |
| Weight | 50 – 120 kg |
| Height | 150 – 195 cm |
| Sex | 50% male / 50% female |

LHS guarantees uniform coverage across every dimension — no clustering, no gaps — with far fewer samples than pure random sampling.

```bash
# Generate and persist a cohort
python eval_cohort.py --generate --n 200 --seed 99 --out cohorts/n200_seed99.json

# Evaluate open-loop
python eval_cohort.py --cohort cohorts/n200_seed99.json
```

![Open-loop cohort evaluation: BIS and Ce spread across 200 patients](cohort_eval.png)

The 3.5× spread in metabolic clearance (CL1: 1.3–4.5 L/min) across the cohort means the same fixed infusion schedule produces meaningfully different anesthetic depth in different patients. The controller must compensate.

---

## Loss function

Controller performance is scored by a normalized cohort loss:

$$\tilde{L} = \frac{1}{N} \sum_{n=1}^{N} \left( \frac{ISE_n}{ISE_{ref}} + \tilde{P}_n \right)$$

- **ISE** is the Integral Squared Error between BIS and target (50), computed over the maintenance window only (t = 2–30 min). Normalized by $ISE_{ref} = 50^2 \times 28 = 70{,}000$ so the score is in [0, 1] regardless of window length or cohort size.
- **Induction penalty** fires if BIS has not crossed below 60 within 2 minutes — the clinical standard for propofol induction. Adds ~7.14 per failed patient.

| Scenario | Loss |
|----------|------|
| Open-loop baseline | 0.1000 |
| Optimized PID | 0.0000179 |
| Improvement | **5,600×** |

---

## PID controller

The controller observes BIS at each time step and outputs an infusion rate. Error is defined as `measurement - setpoint` (positive when BIS is above target, driving higher infusion).

```python
from controller.pid import PIDController

pid = PIDController(kp=4000, ki=600, kd=2000, setpoint=50, dt=0.1)
pid.reset()

rate = pid.step(bis_measurement)  # returns µg/min
```

Key implementation details:
- **Anti-windup**: integral is frozen when output is saturated (at max or min rate)
- **Derivative on measurement**: avoids derivative kick if setpoint changes mid-case
- **Output clamped** to [0, max_rate] — negative infusion rates are physically impossible

---

## Grid search

```bash
python grid_search.py --cohort cohorts/n200_seed99.json
```

Searches 512 candidates ($8^3$ grid) centered on the known optimum region. Each candidate runs all 200 patients in closed-loop.

**512 candidates × 200 patients = 102,400 simulations — 185 seconds in pure Python.**

Best gains found: `kp=4000, ki=600, kd=2000`

![3D scatter of PID gain search colored by normalized loss](results/grid_search.png)

---

## Closed-loop cohort evaluation

```bash
python eval_cohort.py --cohort cohorts/n200_seed99.json --pid 4000,600,2000
```

![Closed-loop PID cohort evaluation: BIS held within 0.7 points of target across all 200 patients](cohort_eval_pid.png)

With the optimized gains, all 200 patients are held within **0.7 BIS points** of the target (median nadir 49.8, P10–P90: 49.3–49.9). Compare to the open-loop spread of 15.7–20.9.

---

## Use the model directly

```python
from physio.core import Patient
from physio.schnider import simulate

patient = Patient(age=55, weight=80, height=175, sex='female')

schedule = [
    (0.0,  160_000.0),  # induction (µg/min)
    (1.0,    8_000.0),  # maintenance
    (45.0,       0.0),  # stop
]

result = simulate(patient, schedule, duration=90.0, dt=0.1)

# result.time  — list of time points (min)
# result.cp    — plasma concentration (µg/mL)
# result.ce    — effect-site concentration (µg/mL)
# result.bis   — BIS score (0–100)
```

---

## References

- Schnider TW et al. *The influence of age on propofol pharmacokinetics.* Anesthesiology 1998; 88(5):1170–82
- Schnider TW et al. *The influence of method of administration and covariates on the pharmacokinetics of propofol in adult volunteers.* Anesthesiology 1999; 90(6):1502–16
- James WPT. *Research on obesity.* HMSO, London, 1976 (LBM formula)
- McKay MD, Beckman RJ, Conover WJ. *A comparison of three methods for selecting values of input variables in the analysis of output from a computer code.* Technometrics 1979; 21(2):239–45
