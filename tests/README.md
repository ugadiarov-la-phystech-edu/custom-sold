# Tests

Run everything from the repository root, in the `mof-text` conda environment (`update_mof_text_env.sh` installs pytest).

```bash
pytest --cov                         # Must pass, with 100% line coverage of sold/
pytest --cov --cov-branch --cov-fail-under=0   # Also show branch coverage
pytest -rx                           # List the known bugs (xfail) with their BUGS_AND_PITFALLS.md numbers
pytest -m "not mujoco and not slow"  # Quick run without MuJoCo rendering or full script runs
```

To run the suite on a remote GPU machine without touching its working copy (settings in the script's header, or in a gitignored `scripts/remote_test.env`):

```bash
scripts/remote_test.sh -q --cov
```

The suite takes under a minute on CPU. Tests never use the GPU, except `test_27` in `unit/test_fixed_bugs.py`, which needs one to create a GPU checkpoint.

## Known bugs

Each open code issue in `BUGS_AND_PITFALLS.md` has a test next to the related unit tests that asserts the correct behavior. It is marked `@pytest.mark.xfail(strict=True, reason="BUGS_AND_PITFALLS.md #<number>: ...")`, so it is reported as `xfailed` and does not fail the run. Find them with `grep -rn "xfail" tests`.

## Layout

| Folder | Contents |
|---|---|
| `unit/` | One file per source area: models, datasets, environments, utils, training loop, `SOLDModule`, script functions. Plus environment checks (`test_environment.py`: LFS checkpoints, pinned versions) and regression tests for fixed items of `BUGS_AND_PITFALLS.md` (`test_fixed_bugs.py`, named `test_<number>_...`). |
| `integration/` | The four Hydra entry points run in-process with the real `configs/` at toy sizes (`slow`), and the real multi-object-fetch environments (`mujoco`). |
| `golden/` | Outputs of the released checkpoints and of SOLD's losses compared with stored tensors in `golden/data/`. Also a tiny SOLD checkpoint in the pre-refactoring format (`golden/data/legacy_tiny_sold.ckpt`, see `references.py`), which must keep loading, giving the same outputs and resuming. |
| `conftest.py`, `fakes.py` | Tiny SAVi/SOLD builders, a deterministic pixel environment (`FakePixel-v0`, used through SOLD's real wrapper stack), and fake `dm_control`, `gymnasium`, `tensordict` and `wandb` modules. |

## When a test fails during a refactor

- **Unit or integration test:** behavior changed. If the change is intended, update the test in the same commit.
- **Golden test:** outputs or state-dict keys changed. Renamed modules break loading of the released checkpoints. If the numeric change is intended, regenerate the references and review the diff:
  ```bash
  PYTHONPATH=sold:. python -m tests.golden.references
  ```
- **Legacy checkpoint tests (`test_legacy_sold_checkpoint*`, `test_sold_checkpoint_loads_every_weight[legacy]`):** the checkpoint format changed (a moved module or a changed `SOLDModule` constructor), so existing checkpoints no longer load. Add a compatibility shim; never regenerate `legacy_tiny_sold.ckpt`.
- **`XPASS(strict)` failure:** a known bug is fixed, so its xfail test now passes and fails the run. Remove the `xfail` marker and mark the item fixed in `BUGS_AND_PITFALLS.md`. Regression tests that document the buggy behavior (for example the `(3, 6, 8)` shape in `unit/test_envs.py`) are commented with the bug number and must be updated at the same time.

Generate golden references only on a machine you trust. Unstable hardware can silently produce wrong results under multithreaded load.
