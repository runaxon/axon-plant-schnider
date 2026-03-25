# Schnider Propofol PK/PD Simulator

Example repository for the [Axon](https://runaxon.com) tech blog series on profiling and optimizing physiological models in Python.

## What this is

A pure-Python implementation of the **Schnider (1998/1999) propofol pharmacokinetic/pharmacodynamic model** — the same model used in clinical Target Controlled Infusion (TCI) pumps.

The model simulates how propofol distributes through the body and produces its anesthetic effect, given a patient's demographics and an infusion schedule.

**Model components:**

- 3-compartment mammillary PK model (central + fast/slow peripheral)
- Effect-site compartment linked via ke0 (plasma-to-brain equilibration)
- BIS pharmacodynamic endpoint via a sigmoidal Hill equation
- Patient-specific parameters derived from age, weight, height, and sex (Schnider 1998)

**Code structure:**

```
physio/
    core.py       # Patient, generic RK4 integrator, simulate(), SimulationResult
    schnider.py   # Schnider-specific: params_from_patient(), derivatives(), outputs()
demo_schnider.py  # Induction + maintenance + washout example
```

`derivatives()` in `schnider.py` is the ODE right-hand side — a pure arithmetic function with no Python object overhead, designed as the transpilation target for the C optimization stage of the blog series.

![Schnider Propofol PK/PD simulation](schnider_demo.png)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the demo

```bash
python demo_schnider.py
```

Simulates a 70 kg / 40 yr male patient receiving:
- Induction: 140 mg/min for 1 min (~2 mg/kg)
- Maintenance: 7 mg/min (~100 µg/kg/min) for 29 min
- Washout: infusion stopped, passive elimination observed

Prints peak Cp, peak Ce, and nadir BIS, then saves a plot to `schnider_demo.png`.

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
