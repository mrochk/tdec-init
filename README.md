# TDec-Init
Warm starting neural dynamics models from few Jacobian samples.

<img src="image.png">

## Instructions

Setup
```bash
git clone --recurse-submodules git@github.com:mrochk/tdec-init.git
cd tdec-init
uv sync . 
```

Run and plot the (1) *identification accuracy* experiment
```bash
uv run experiments/run_identification.py
uv run experiments/plot_identification.py
```

Run and plot the (2) *closed-loop performance* experiment
```bash
uv run experiments/run_closed_loop.py
uv run experiments/plot_closed_loop.py
```

