from physio.core import Patient
from physio.schnider import params_from_patient, run_mass_balance_test

patient = Patient(sex='male', weight=70, height=175, age=40)
params = params_from_patient(patient)

run_mass_balance_test(params, infusion_rate=8000.0, duration=60, dt=0.001)
