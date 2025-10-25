import stim
import pymatching
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors

def surface_code_circuit(p, d):
    return stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    rounds=d,
    distance=d,
    after_clifford_depolarization=p, # Chance of noise after a clifford gate
    after_reset_flip_probability=p,   # Chance of reset to 1 After a qubit is reset to 0
    before_measure_flip_probability=p, # Chance of flip right before measuring a qubit
    before_round_data_depolarization=p) # Before each round of error correction starts, each data qubit might


def plot_logical_error_rates():
    num_shots = 100_000
    for d in [5]:
        xs = []
        ys = []
        for noise in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]:
            circuit = surface_code_circuit(noise, d)
            num_errors_sampled = count_logical_errors(circuit, num_shots)
            xs.append(noise)
            ys.append(num_errors_sampled / num_shots)
        plt.plot(xs, ys, label="d=" + str(d))
    plt.loglog()
    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate per shot")
    plt.legend()
    plt.show()

plot_logical_error_rates()


d = 5
p = 0.004

num_shots = 100_000
circuit = surface_code_circuit(p, d)
num_errors = count_logical_errors(circuit, num_shots)

print(f"Distance {d} surface code with physical error rate {p} has logical error rate {1-num_errors / num_shots} over {num_shots} shots.")

import numpy as np

ml_decoder=decoder

def surface_code_circuit(p, d):
    return stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    rounds=d,
    distance=d,
    after_clifford_depolarization=p, # Chance of noise after a clifford gate
    after_reset_flip_probability=p,   # Chance of reset to 1 After a qubit is reset to 0
    before_measure_flip_probability=p, # Chance of flip right before measuring a qubit
    before_round_data_depolarization=p) # Before each round of error correction starts, each data qubit might

def logical_error_rate_ml(circuit: stim.Circuit, num_shots: int, ml_decoder) -> int:
    ml_decoder.eval()
    sampler = circuit.compile_detector_sampler()
    detections_test, flips_test = sampler.sample(10**5, separate_observables=True)
    detections_test = detections_test.astype(int) * 2 - 1
    detections_test, flips_test = torch.Tensor(detections_test), torch.Tensor(flips_test.astype(int).flatten())
    error_rate = 1-torch.mean(((ml_decoder(detections_test) > 0.5) == flips_test[:,None]).float())
    return float(error_rate)

def plot_logical_error_rates_ml(ml_decoder):
    num_shots = 100000
    for d in [5]:
        xs = []
        ys_ml = []
        ys_mwpm = []
        for noise in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]:
            circuit = surface_code_circuit(noise, d)
            # num_errors_sampled = count_logical_errors(circuit, num_shots)
            ml_ler = logical_error_rate_ml(circuit, num_shots, ml_decoder)
            mwpm_ler = count_logical_errors(circuit, num_shots)/num_shots
            print(f"Noise={noise:.3f} | ML LER={ml_ler:.6f} | MWPM LER={mwpm_ler:.6f}")
            xs.append(noise)
            ys_ml.append(ml_ler)
            ys_mwpm.append(mwpm_ler)
        plt.plot(xs, ys_ml, label="ML d=" + str(d))
        plt.plot(xs, ys_mwpm, label="MWPM d=" + str(d))
    plt.loglog()
    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate per shot")
    plt.legend()
    plt.show()

