# TQDM Progress Bar CLI Fix

## Problem

The `em43_doubling_new.py` script was not showing tqdm progress bars when run from the command line. This was because the `TqdmMonitor` was auto-detecting the environment and potentially using notebook-style tqdm instead of CLI-style tqdm.

## Root Cause

The `TqdmMonitor` class has auto-detection logic that tries to determine whether to use:
- `tqdm.notebook.tqdm` for Jupyter environments
- `tqdm.tqdm` for CLI environments

However, the auto-detection wasn't working perfectly in all CLI scenarios.

## Solution

### 1. Fixed Original File

Modified `examples/sandbox/em43_doubling_new.py`:

```python
# OLD (auto-detection)
tqdm_monitor = TqdmMonitor(N_GENERATIONS)

# NEW (force CLI mode)
tqdm_monitor = TqdmMonitor(N_GENERATIONS, force_notebook=False)
```

This explicitly forces the use of CLI-style tqdm progress bars.

### 2. Created CLI-Optimized Version

Created `examples/sandbox/em43_doubling_cli.py` with:
- Command-line argument parsing
- Reasonable default parameters for quick testing
- Explicit CLI tqdm configuration
- Better error handling and user feedback

## Usage Examples

### Fixed Original Script
```bash
# Now shows proper CLI progress bar
python examples/sandbox/em43_doubling_new.py
```

### New CLI-Optimized Script
```bash
# Quick test with small parameters
python examples/sandbox/em43_doubling_cli.py --population 100 --generations 10 --inputs 1-5

# Full test with larger parameters
python examples/sandbox/em43_doubling_cli.py --population 2000 --generations 100 --inputs 1-30

# Quiet mode (only progress bar, no detailed telemetry)
python examples/sandbox/em43_doubling_cli.py --quiet --population 500 --generations 20
```

## Progress Bar Features

The CLI progress bars now show:
- **Progress**: Visual bar with percentage
- **Speed**: Generations per second
- **ETA**: Estimated time remaining
- **Best Fitness**: Current best fitness score
- **Mean Fitness**: Current population mean fitness

Example output:
```
Training:  40%|████████████████████▌                    | 2/5 [00:00<00:00, 43.18gen/s, Best: 0.3783, Mean: 0.3015]
```

## TqdmMonitor Parameters

The `TqdmMonitor` class supports these parameters:

```python
TqdmMonitor(
    total_generations=100,    # Total generations for progress bar
    update_every=1,          # Update frequency (every N generations)
    force_notebook=False     # Force CLI mode (True for Jupyter)
)
```

### Auto-Detection Logic

When `force_notebook=None` (default), the monitor:
1. Checks if running in IPython/Jupyter environment
2. Checks if `tqdm.notebook` is available
3. Falls back to CLI tqdm if either check fails

### Manual Override

- `force_notebook=False`: Always use CLI tqdm
- `force_notebook=True`: Always use notebook tqdm (requires ipywidgets)

## Testing

Both scripts have been tested and confirmed to show proper progress bars:

```bash
# Test the fixed original
python examples/sandbox/em43_doubling_new.py

# Test the CLI-optimized version
python examples/sandbox/em43_doubling_cli.py --population 100 --generations 5 --inputs 1-3
```

## Recommendations

1. **For CLI usage**: Use `force_notebook=False` in `TqdmMonitor`
2. **For Jupyter**: Use `JupyterMonitor` or `force_notebook=True`
3. **For development**: Use the CLI-optimized script with small parameters for quick testing
4. **For production**: Use the original script with full parameters

## Related Files

- `examples/sandbox/em43_doubling_new.py` - Fixed original script
- `examples/sandbox/em43_doubling_cli.py` - New CLI-optimized script
- `emergent_models/training/monitor.py` - TqdmMonitor implementation

The fix ensures that users get proper visual feedback when running EM-4/3 training from the command line, making it much easier to monitor training progress and estimate completion times.
