import stim
import torch
from torch import nn

# defining the rounds, distance etc.

p=0.004 # enter your preffered noise value

d=5 #enter your preffered code distance

surface_code_circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    rounds=d,
    distance=d,
    after_clifford_depolarization=p, # Chance of noise after a clifford gate
    after_reset_flip_probability=p,   # Chance of reset to 1 After a qubit is reset to 0
    before_measure_flip_probability=p, # Chance of flip right before measuring a qubit
    before_round_data_depolarization=p) # Before each round of error correction starts, each data qubit might randomly suffer an X, Y, or Z error

train_size = 10**7

sampler = surface_code_circuit.compile_detector_sampler()
detections, flips = sampler.sample(train_size, separate_observables=True) 


detections = detections.astype(int) * 2 -1
detections, flips = torch.Tensor(detections), torch.Tensor(flips.astype(int).flatten())

num_input = detections.shape[1]

decoder = nn.Sequential(
    nn.Linear(num_input, 256),
    #nn.BatchNorm1d(1024),
    nn.SiLU(),
    nn.Linear(256, 512), 
    #nn.BatchNorm1d(1024),
    nn.SiLU(),
    nn.Linear(512, 1024), # instead of slowly expadning and contracting at the end expand immediately contract slow
    #nn.BatchNorm1d(1024), #normalizes each of the 1024 channels independently across the batch
    nn.SiLU(),
    nn.Linear(1024, 1),
    #nn.BatchNorm1d(1024),
    nn.Sigmoid()
)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)


from tqdm.auto import *

def train_loop(measurements, flips, bs=200): 
    decoder.train()
    running_avg = 0
    with trange(train_size//bs) as pbar:
        for batch in pbar:
            X = measurements[batch*bs:(batch+1)*bs]
            y = flips[batch*bs:(batch+1)*bs][:,None]
            #pred = decoder(X.reshape((-1, 5*12)))
            pred = decoder(X)
            loss = loss_fn(pred, y)
            acc = torch.mean(((pred>0.5) == y).float())
            running_avg = acc * 0.01 + running_avg*0.99 if running_avg != 0 else acc
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch % 100 == 0:
                pbar.set_description(f'{running_avg:>4f} {loss:.4f}')

# training loops can modify or remove loops depending on how much training is required

train_loop(detections, flips, bs=512)
train_loop(detections, flips, bs=512)
train_loop(detections, flips, bs=4096)
train_loop(detections, flips, bs=4096)
train_loop(detections, flips, bs=8000)
train_loop(detections, flips, bs=8000)
train_loop(detections, flips, bs=8000)
train_loop(detections, flips, bs=8000)
train_loop(detections, flips, bs=8000)
train_loop(detections, flips, bs=8000)
train_loop(detections, flips, bs=8000)
train_loop(detections, flips, bs=8000)
