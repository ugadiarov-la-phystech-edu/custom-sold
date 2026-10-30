# Tests

Run everything from the repository root, in the `mof-text` conda environment (`update_mof_text_env.sh` installs pytest).

```bash
pytest -m "not bug" --cov            # Regression suite: must pass, with 100% line coverage of sold/
pytest -m bug                        # Known bugs: fail until fixed (see BUGS_AND_PITFALLS.md)
pytest                               # Everything: only the bug tests fail
pytest -m "not bug" --cov --cov-branch --cov-fail-under=0   # Also show branch coverage
pytest -m "not mujoco and not slow"  # Quick run without MuJoCo rendering or full script runs
```

The regression suite takes under a minute on CPU. Tests never use the GPU, except `test_27` in `bugs/`, which needs one to create a GPU checkpoint.

## Layout

| Folder | Contents |
|---|---|
| `unit/` | One file per source area: models, datasets, environments, utils, training loop, `SOLDModule`, script functions. Plus environment checks (`test_environment.py`: LFS checkpoints, pinned versions). |
| `integration/` | The four Hydra entry points run in-process with the real `configs/` at toy sizes (`slow`), and the real multi-object-fetch environments (`mujoco`). |
| `golden/` | Outputs of the released checkpoints and of SOLD's losses compared with stored tensors in `golden/data/`. |
| `bugs/` | One test per item in `BUGS_AND_PITFALLS.md`, named `test_<number>_...`. |
| `conftest.py`, `fakes.py` | Tiny SAVi/SOLD builders, a deterministic pixel environment (`FakePixel-v0`, used through SOLD's real wrapper stack), and fake `dm_control`, `gymnasium`, `tensordict` and `wandb` modules. |

## When a test fails during a refactor

- **Unit or integration test:** behavior changed. If the change is intended, update the test in the same commit.
- **Golden test:** outputs or state-dict keys changed. Renamed modules break loading of the released checkpoints. If the numeric change is intended, regenerate the references and review the diff:
  ```bash
  PYTHONPATH=sold:. python -m tests.golden.references
  ```
- **A bug test starts passing:** the bug is fixed. Remove the entry from `BUGS_AND_PITFALLS.md` and turn the test into a regular test (drop the `bug` marker, move it next to the related unit tests). Regression tests that document the buggy behavior (for example the `(3, 6, 8)` shape in `unit/test_envs.py`) are commented with the bug number and must be updated at the same time.

Generate golden references only on a machine you trust. Unstable hardware can silently produce wrong results under multithreaded load.
