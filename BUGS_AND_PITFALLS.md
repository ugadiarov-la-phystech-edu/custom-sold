# Bugs and Pitfalls

Findings from a full read of this repository (all Python sources, configs, README and Apptainer files) and a few checks on the `victrix` machine.
Line numbers refer to commit `f233960`.

Severity levels:

- **Blocker**: prevents running out of the box.
- **Bug**: wrong behavior or a crash in a code path that is normally used.
- **Pitfall**: correct code, but surprising behavior or a trap for future changes.
- **Paper deviation**: the code differs from the SOLD paper (see `PAPER_NOTES.md`); fixing it changes how existing checkpoints behave.
- **(fixed)**: fixed in the working tree, not yet committed. Its regression test sits next to the related unit tests under a `# BUGS_AND_PITFALLS.md #<number> (fixed).` comment.

Each open code issue has a test next to the related unit tests that asserts the correct behavior. It is marked `@pytest.mark.xfail(strict=True)` with the item number in its reason, so `pytest` reports it as `xfailed`; list them with `pytest -rx`, and see `tests/README.md`.
A fix turns it into a strict `XPASS` failure, which is the cue to remove the marker. Fixed issues keep their test as a regular regression test.

## Summary

| # | Severity | Area | Issue |
|---|---|---|---|
| 1 | Blocker | Checkpoints | `.ckpt` files are Git LFS pointer stubs, not weights |
| 2 | Blocker (fixed) | Config | SAVi dataset path is hardcoded to the author's machine |
| 3 | Blocker | Environment | Required packages are missing from local conda envs |
| 3a | Blocker (fixed) | Dependencies | Lightning 2.5.3 and newer crash SOLD's custom progress bar |
| 3b | Blocker | Dependencies | Rich 14.1 and newer crash Lightning 2.5.2 and older |
| 3c | Blocker | Dependencies | setuptools 81 and newer break `lightning-bolts` |
| 4 | Bug (fixed) | Resume | Resuming calls a replay buffer method that does not exist |
| 5 | Bug (fixed) | Checkpoints | Pruning old replay buffers can delete the newest one |
| 6 | Bug (fixed) | Evaluation | Evaluating the latest checkpoint likely crashes |
| 7 | Bug (fixed) | Evaluation | Evaluation never updates the last action |
| 8 | Bug (fixed) | Environments | `survival` suite imports a module that does not exist |
| 9 | Pitfall | Environments | Reach episodes are half the configured length |
| 10 | Pitfall | Disk | Each replay buffer copy is about 15 GB |
| 11 | Pitfall (fixed) | Reproducibility | The seed never reaches the environment |
| 12 | Pitfall (fixed) | Memory | A new MuJoCo environment is created on every reset |
| 13 | Pitfall (fixed) | Data | Actions are shifted one step between SAVi and SOLD |
| 14 | Pitfall (fixed) | Data | Random crop never uses the last possible window |
| 15 | Pitfall (fixed) | Dependencies | Every suite requires `multi_object_fetch` |
| 16 | Pitfall (fixed) | Hydra | Scripts use different working-directory behavior |
| 17 | Minor (fixed) | Code quality | Smaller issues |
| 18 | Bug (fixed) | Logging | The last metrics record never reaches `metrics.jsonl` |
| 19 | Pitfall (fixed) | Data | The replay buffer never samples the last window of an episode |
| 20 | Minor (fixed) | Code quality | A wrong-size attention mask raises `AttributeError` instead of `ValueError` |
| 21 | Minor (fixed) | Code quality | `AttentionMask.show` always titles the plot `head=0` |
| 22 | Bug (fixed) | Training | Training crashes with `RecursionError` after about 975 evaluations |
| 23 | Bug (fixed) | Config | `configs/autoencoder/cnn.yaml` cannot be instantiated |
| 24 | Pitfall (fixed) | Environments | Non-square image sizes come out transposed |
| 25 | Bug (fixed) | Data | `RingBufferDataset` crashes when created with predefined fields |
| 26 | Bug | Training | The actor's entropy bonus ignores per-step discounts |
| 27 | Bug (fixed) | Checkpoints | SAVi checkpoints saved on a GPU cannot be loaded on a CPU-only machine |
| 28 | Bug | Actor | The actor's mean can never exceed about ±0.76 |
| 29 | Bug | Imagination | Imagined actions are not limited to the action space |
| 30 | Bug | Dynamics | Teacher forcing lets the dynamics model see the frame it predicts |
| 31 | Paper deviation | SAVi | Slot Attention scales by the wrong dimension |
| 32 | Paper deviation | Transformers | ALiBi slopes differ from the original formula |
| 33 | Paper deviation | Rewards, values | The two-hot mean is not the expectation of Eq. 5 |
| 34 | Bug (fixed) | Critic | The target critic accumulated unused gradients |
| 35 | Bug (fixed) | Imagination | `batch_size: 1` crashed the λ-return computation |
| 36 | Pitfall (fixed) | Performance | The attention mask was rebuilt in every transformer layer |
| 37 | Pitfall (fixed) | Performance | The actor's bounds were copied to the GPU on every call |
| 38 | Pitfall | Evaluation | `evaluate_sold.py` replays the scenes training started with |
| 39 | Pitfall | Evaluation | Evaluation during training shares the environment with training |

---

## Blockers

### 1. Checkpoints are Git LFS pointer stubs

**Status on `victrix`:** resolved. `git-lfs` was installed into the `mof-text` conda env and the checkpoints were pulled.

`.gitattributes` tracks `checkpoints/**/*.ckpt` with Git LFS.
A clone without LFS contains 130-byte text files such as:

```
version https://git-lfs.github.com/spec/v1
oid sha256:81ab607bcabb6dcc17bf7f3df5170e920decb67791aa96fbe4de4a02e7e6c1e7
size 170447222
```

`configs/train_sold.yaml:59` points at `../checkpoints/autoencoder/savi/reach_red/latest.ckpt`, so `train_sold.py` fails while unpickling the stub.
`git-lfs` is not installed system-wide on `victrix`.

**Fix:**

```bash
conda install -n mof-text -c conda-forge git-lfs   # or: sudo apt install git-lfs
git lfs install --local
git lfs pull
```

`update_mof_text_env.sh --lfs-pull` does this automatically.

Alternatively, pre-train SAVi yourself and point `model.autoencoder.checkpoint_path` at the result.

### 2. SAVi dataset path is hardcoded (fixed)

**Status:** fixed (uncommitted). `dataset.path` and `dataset.data_dir` in `configs/train_autoencoder.yaml` are now mandatory (`???`) and must be given on the command line.

`configs/train_autoencoder.yaml:15-16`:

```yaml
path: ../experiments/generate_dataset/npz_reach_red
data_dir: 2025-04-08_11-29-03
```

This is a dated output folder from the author's machine.
`generate_dataset.py` writes to `../experiments/generate_dataset/<experiment>/<timestamp>`, so after generating a dataset you must override both values:

```bash
python train_autoencoder.py dataset.path=../experiments/generate_dataset/my_dataset dataset.data_dir=<timestamp>
```

Note also that `generate_dataset.yaml` defaults to `save_format: png`, while the example path name suggests NPZ. Both formats are supported.

### 3. Missing packages in local conda envs

Status on `victrix`:

| Package | `mof` | `mof-text` |
|---|---|---|
| torch, torchvision, lightning, lightning-bolts, hydra-core, termcolor, rich, matplotlib, tensorboardX, av, wandb | present, but see 3a and 3b | present, pinned to working versions |
| dm_control (only for the `dmcontrol` suite) | missing | missing |
| tensordict (only for the `survival` suite) | missing | missing |

`apptainer/environment.yml` lists SOLD's dependencies without versions, and it does not include `dm_control` or `tensordict`.
An unpinned install today pulls versions that hit 3a, 3b and 3c.

**Fix:** run `update_mof_text_env.sh` from the repository root. It installs the versions below, which pass a full run of all four scripts on `victrix`.

| Package | Version |
|---|---|
| torch / torchvision | 2.9.0 / 0.24.0 |
| lightning | 2.5.2 |
| lightning-bolts / pytorch-lightning | 0.7.0 / 1.9.5 |
| rich | 14.0.0 |
| setuptools | below 81 |
| hydra-core / omegaconf | 1.3.2 / 2.3.0 |

### 3a. Lightning 2.5.3 and newer crash SOLD's custom progress bar (fixed)

**Status:** fixed (uncommitted). `OnlineProgressBar.on_train_start` now calls Lightning's `on_train_start` and creates the train task immediately, so Lightning's own `on_train_batch_end` finds it. Not yet tried with Lightning 2.5.3 or newer, so keep the pin until it is.

`OnlineProgressBar` (`sold/utils/training.py`) overrides `on_train_start` without calling the parent, and creates the Rich progress display lazily in `_update`.
Starting with Lightning 2.5.3, `RichProgressBar.on_train_batch_end` creates the train task itself if it does not exist yet.
That runs before `_update` ever creates the display, so the first training batch fails:

```
File ".../lightning/pytorch/callbacks/progress/rich_progress.py", in _add_task
    assert self.progress is not None
AssertionError
```

The `mof` env has Lightning 2.5.5 and is affected.

**Fix:** pin `lightning==2.5.2`. A code fix would call `super().on_train_start(...)` or create the display in `OnlineProgressBar.on_train_start`.

### 3b. Rich 14.1 and newer crash Lightning 2.5.2 and older

Rich 14.1 replaced the console's internal `_live` attribute with a `_live_stack` list.
Lightning added a compatibility shim only in later releases, so with Lightning 2.5.2 training fails at start:

```
File ".../rich/console.py", in clear_live
    self._live_stack.pop()
IndexError: pop from empty list
```

**Fix:** pin `rich==14.0.0` together with `lightning==2.5.2`.

### 3c. setuptools 81 and newer break `lightning-bolts`

`lightning-bolts` 0.7.0 imports `pkg_resources`, which recent setuptools releases no longer ship.
Both training scripts fail, because the autoencoder config uses the bolts learning-rate scheduler and `train_sold.py` loads the autoencoder through it:

```
ModuleNotFoundError: No module named 'pkg_resources'
hydra.errors.InstantiationException: Error locating target 'pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR'
```

**Fix:** pin `setuptools<81`.

---

## Bugs

### 4. Resuming training crashes (fixed)

**Status:** fixed (uncommitted). `on_load_checkpoint` calls `load_from_disk_storage`.

`sold/utils/training.py:264` calls:

```python
self.replay_buffer.load_from_files(...)
```

`RingBufferDataset` defines this method as `load_from_disk_storage` (`sold/datasets/ring_buffer.py:202`).
Resuming from any checkpoint whose saved replay buffer folder still exists raises `AttributeError`.

**Fix:** rename the call to `load_from_disk_storage`.

### 5. Pruning old replay buffers can delete the newest one (fixed)

**Status:** fixed (uncommitted). Every replay buffer copy except the one for the current checkpoint is deleted, so the order of the names no longer matters.

`sold/utils/training.py:243-246`:

```python
shutil.copytree(..., replay_buffer_dir)
old_replay_buffer_dirs = sorted(list(self.checkpoints_dir.glob("replay_buffer-steps=*")))[:-1]
for old_replay_buffer_dir in old_replay_buffer_dirs:
    shutil.rmtree(old_replay_buffer_dir)
```

The folders are sorted as strings, so `replay_buffer-steps=10500-...` sorts before `replay_buffer-steps=9000-...`.
Once the step count gains a digit, the newest copy is deleted and an older one is kept.
The checkpoint then stores a `replay_buffer_path` that no longer exists, and resuming silently starts with an empty buffer.

**Fix:** sort numerically, for example:

```python
import re
key = lambda p: int(re.search(r"steps=(\d+)", p.name).group(1))
old_replay_buffer_dirs = sorted(self.checkpoints_dir.glob("replay_buffer-steps=*"), key=key)[:-1]
```

Or simply delete every folder except `replay_buffer_dir`.

### 6. Evaluating the latest checkpoint likely crashes (fixed)

**Status:** fixed (uncommitted). The replay buffer is restored only when a trainer is attached (resuming with `trainer.fit`), not in `load_from_checkpoint` without one.

`evaluate_sold.py:55` calls `SOLDModule.load_from_checkpoint(checkpoint, env=env)` without a trainer.
That triggers `on_load_checkpoint` (`sold/utils/training.py:260`).
If the checkpoint's replay buffer folder still exists, which is true for the newest checkpoint of a training run, it calls `self.train_dataloader()`.
That method reads `self.logger.log_dir`, and `self.logger` is `None` outside training.
Even past that point, it would hit bug 4.

**Fix:** skip replay buffer loading when `self._trainer is None`, or pass a flag from `evaluate_sold.py`.

### 7. Evaluation never updates the last action (fixed)

**Status:** fixed (uncommitted). `evaluate_sold.play_episode` writes each action into `sold.last_action`, like training does.

`OnlineModule.play_episode` writes each action into `self.last_action` (`sold/utils/training.py:214`), which `select_action` passes to the SAVi encoder (`sold/train_sold.py:311`).
The copy of `play_episode` in `evaluate_sold.py:24` stores the action in a local variable instead, so the encoder always receives the initial NaN action.

This has no effect with the default `TransformerPredictor`, which ignores actions.
It would corrupt slots with `ActionConditionalTransformerPredictor`, because the NaN would propagate.

**Fix:** use `sold.last_action[:] = sold.select_action(...)`.

### 8. `survival` suite imports a missing module (fixed)

**Status:** fixed (uncommitted). Selecting `suite: survival` without `envs/from_survival.py` now raises a `ValueError` that says the module is missing. The suite still works if the module is added; `gymnasium` and `tensordict` remain undeclared dependencies.

`sold/envs/__init__.py:20` imports `envs.from_survival`, which is not in the repository.
`sold/envs/wrappers/to_tensordict.py` also needs `gymnasium` and `tensordict`, which are not listed as dependencies.
Selecting `suite: survival` raises `ModuleNotFoundError`.

---

## Pitfalls

### 9. Reach episodes are half the configured length

`multi_object_fetch` registers its environments with gym time limits of 50 simulator steps for Reach and 100 for Push and Pick.
`gym.make` wraps them in gym's own `TimeLimit`, and SOLD then applies `ActionRepeat(2)` on top.

| Environment | gym limit | Agent steps with action repeat 2 | `max_episode_steps` in config |
|---|---|---|---|
| Reach | 50 | 25 | 50 |
| Push, Pick | 100 | 50 | 50 |

So Reach episodes end after 25 agent steps, regardless of `max_episode_steps: 50` in `train_sold.yaml:54` and `evaluate_sold.yaml:14`.
The config value still sizes the transformer attention masks, so nothing crashes, but episode length is not what the config suggests.

### 10. Each replay buffer copy is about 15 GB

`buffer_capacity: 1250000` (`configs/train_sold.yaml:23`) at 64×64×3 bytes per frame is about 15.4 GB of memory-mapped data.
With `save_replay_buffer: True`, the whole buffer is copied at every evaluation (`sold/utils/training.py:243`), which happens every 1500 steps.
Expect heavy disk I/O, and two copies on disk at times, or more with bug 5.

### 11. The seed never reaches the environment (fixed)

**Status:** fixed (uncommitted). `make_env` takes `seed: Optional[int] = None` and seeds the environment when a seed is given. `train_sold.yaml` and `evaluate_sold.yaml` pass `seed: ${seed}`; each distractor variant gets `seed + number of distractors`. `generate_dataset.yaml` stays unseeded on purpose, because it builds one environment per episode and a fixed seed would make every episode identical.

`make_env` accepts `seed` (`sold/envs/from_gym.py:8`, `sold/envs/from_mof.py:18`), but the gym and multi-object-fetch paths never use it.
MuJoCo environments seed themselves randomly, so runs are not reproducible even with a fixed `seed` in the config.

### 12. A new MuJoCo environment is created on every reset (fixed)

**Status:** fixed (uncommitted). `VariableDistractorsWrapper` creates one environment per distractor count on first use and reuses it; `close()` closes all of them.

`VariableDistractorsWrapper.reset` (`sold/envs/from_mof.py:14`) builds a fresh environment on every episode and never closes the previous one.
Each one holds a MuJoCo simulation and possibly a rendering context.
Over millions of steps this may grow memory use.
Caching one environment per distractor count would avoid it.

### 13. Actions are shifted one step between SAVi and SOLD (fixed)

**Status:** fixed (uncommitted). The SAVi datasets now return actions aligned like SOLD's replay buffer: entry t is the action that led to frame t, and entry 0 is NaN (it is dropped before use). Episode files on disk are unchanged. With the default `TransformerPredictor`, which ignores actions, SAVi training is unaffected.

- In SOLD's replay buffer, `action[t]` is the action that led to `obs[t]`, and `action[0]` is NaN.
- In the SAVi datasets, `actions[t]` is the action taken after `obs[t]`.

Both pipelines call the autoencoder with `actions[:, 1:]` (`sold/train_autoencoder.py:38`), so SAVi pre-training sees actions one step later than SOLD does.
This is harmless with the default `TransformerPredictor`, which ignores actions.
It matters as soon as an action-conditioned predictor is used.

### 14. Random crop never uses the last possible window (fixed)

**Status:** fixed (uncommitted). The random start is drawn from `0 … len(images) − sequence_length` inclusive.

`sold/datasets/image.py:65` and `:83` use:

```python
np.random.randint(0, len(images) - self.sequence_length)
```

The upper bound is exclusive, so the last valid start index is never sampled.
If an episode has exactly `sequence_length` frames, the call raises `ValueError`.
Use `len(images) - self.sequence_length + 1`.

### 15. Every suite requires `multi_object_fetch` (fixed)

**Status:** fixed (uncommitted). `Pixels` imports `multi_object_fetch` lazily and falls back to the generic rendering path when it is not installed.

`sold/envs/wrappers/pixels.py:2` imports `multi_object_fetch.env` at module level.
The `gym` and `dmcontrol` suites therefore fail if `multi_object_fetch` and `mujoco-py` are not installed.

### 16. Scripts use different working-directory behavior (fixed)

**Status:** fixed (uncommitted). `evaluate_sold.py` now uses `version_base=None` like the other scripts, so it no longer changes into the output folder; relative checkpoint paths are still resolved against the launch folder. `_self_` was added to its defaults list.

`train_sold.py`, `train_autoencoder.py` and `generate_dataset.py` set `version_base=None`, so Hydra does not change the working directory.
`evaluate_sold.py:47` omits `version_base`, so Hydra uses legacy behavior and changes into the output folder.
That script handles it with `get_original_cwd()`, but relative paths behave differently between scripts.

All scripts must be launched from the `sold` folder, because imports such as `from datasets...` and `from envs...` rely on it being on `sys.path`.
The local `datasets` package also shadows the Hugging Face `datasets` library inside this project.

### 17. Smaller issues (fixed)

**Status:** fixed (uncommitted). All five items are fixed. Note for the third: checkpoints trained with `finetune_autoencoder: True` before this fix contain four optimizer states, so resuming them with the new code fails with an optimizer-count mismatch.

- `SAVi._initialize_parameters` (`sold/modeling/autoencoder/savi/autoencoder.py:29`) reads `self.corrector.dim_slots`, which does not exist. The branch is currently unreachable.
- `select_action` clamps with `action_space.low[0]` and `high[0]` (`sold/train_sold.py:326`), which assumes every action dimension has the same bounds.
- `autoencoder_optimizer` is created outside `configure_optimizers` (`sold/train_sold.py:69`), so its state is not saved in checkpoints when `finetune_autoencoder: True`.
- SOLD's training step runs the autoencoder forward pass with gradients enabled even when `finetune_autoencoder: False`, which wastes GPU memory.
- `sold/utils/visualizations.py` repeats its import block.

---

## Found while writing the test suite

### 18. The last metrics record never reaches `metrics.jsonl` (fixed)

**Status:** fixed (uncommitted). `ExtendedTensorBoardLogger.finalize` writes the buffered record.

`ExtendedTensorBoardLogger.log_metrics` (`sold/utils/logging.py:84`) buffers metrics and writes a record only when a later step arrives.
Nothing flushes the buffer at the end, because the logger does not override `finalize` or `save`.
The metrics of the final step, including the final evaluation, are only in TensorBoard.

**Fix:** override `finalize` to call `_flush_metrics()` and close the file.

### 19. The replay buffer never samples the last window of an episode (fixed)

**Status:** fixed (uncommitted). The offset is drawn uniformly from `0 … max_offset` inclusive.

`RingBufferDataset.sample` (`sold/datasets/ring_buffer.py:164-166`) computes `max_offset = episode_length - sequence_length` and then `offset = (max_offset * rand).long()`.
Because `rand < 1`, the offset is at most `max_offset - 1`, so the final transitions of every episode (often the rewarding ones) are sampled less often than the others, and the very last window never.
This is the replay-buffer twin of item 14.

**Fix:** `torch.randint(0, max_offset + 1, ...)` per sequence.

### 20. A wrong-size attention mask raises `AttributeError` (fixed)

**Status:** fixed (uncommitted). The message uses `self._full_mask.size()`.

The error message in `AttentionMask.forward` (`sold/modeling/attention_mask.py:37`) formats `self.mask`, which does not exist.
The intended `ValueError` becomes an `AttributeError`.

### 21. `AttentionMask.show` always titles the plot `head=0` (fixed)

**Status:** fixed (uncommitted). The title shows the requested head.

`sold/modeling/attention_mask.py:48` uses `f"... (head={0})"` instead of `{head}`.

### 22. Training crashes with `RecursionError` after about 975 evaluations (fixed)

**Status:** fixed (uncommitted). `get_attention_weights` removes its `forward` patch (in a `finally` block) after collecting the weights. Runs started before this fix still accumulate wrappers until they are restarted on the new code.

`get_attention_weights` (`sold/utils/visualizations.py:177`) calls `enable_attention_weights` on every `nn.MultiheadAttention` of the actor and the reward predictor.
That function replaces `module.forward` with a wrapper around the current `forward` and never restores it.
SOLD calls it once after every evaluation (`sold/train_sold.py:197` and `:265`), so each evaluation adds one more nested wrapper, and every later forward pass of those modules goes through all of them.
With Python's default recursion limit of 1000, the actor's forward pass raises `RecursionError` after about 975 evaluations (verified in isolation; inside Lightning the stack is deeper, so it happens somewhat earlier).

- With `eval_freq: 1500` (the default) that is roughly 1.4M environment steps. The released Reach checkpoint stops at 1.2M.
- With `eval_freq: 10000` it is roughly 9.5M steps.
- Before the crash, every actor call also pays for the growing wrapper chain and computes attention weights it does not need.

**Fix:** wrap only once (for example, keep the original `forward` and restore it after the hook is removed), or compute the weights by calling `self_attn(..., need_weights=True)` directly.
Resuming from a checkpoint resets the chain, because the wrappers are not saved.

### 23. `configs/autoencoder/cnn.yaml` cannot be instantiated (fixed)

**Status:** fixed (uncommitted). `image_size` was removed from the encoder section.

The config passes `image_size` to `modeling.autoencoder.cnn.encoder.CnnEncoder`, whose constructor (`sold/modeling/autoencoder/cnn/encoder.py:9`) has no such parameter.
`train_autoencoder.py autoencoder=cnn` fails with `TypeError`.

**Fix:** remove `image_size` from the encoder section (the decoder still needs it).

### 24. Non-square image sizes come out transposed (fixed)

**Status:** fixed (uncommitted). PIL is called with `(width, height)`.

`Pixels._get_obs` (`sold/envs/wrappers/pixels.py:26-27`) resizes non-MuJoCo frames with PIL, whose `resize` takes `(width, height)`.
For `image_size: [H, W]` with `H != W`, observations have shape `(3, W, H)` while `observation_space` says `(3, H, W)`.
Square sizes, as used everywhere in the configs, are unaffected.

### 25. `RingBufferDataset` crashes when created with predefined fields (fixed)

**Status:** fixed (uncommitted). `save_path` is set before the parent constructor runs.

`EpisodeDataset.__init__` calls `_initialize_from_fields` when `info.fields` is set, but `RingBufferDataset.__init__` (`sold/datasets/ring_buffer.py:112-113`) assigns `self.save_path` only after calling it, so construction raises `AttributeError`.
SOLD never passes fields today, so this is latent.

### 26. The actor's entropy bonus ignores per-step discounts

`imagine_ahead` returns entropies of shape `(batch, horizon, 1)`, while `self.discounts` has shape `(1, horizon)`.
In `compute_actor_loss` (`sold/train_sold.py:285`) the product broadcasts to `(batch, horizon, horizon)`, so every entropy is weighted by the mean discount instead of the discount of its own step.
The effect is small with the default weight of `3e-4`, but the loss is not what the code intends.

**Fix:** `entropies.squeeze(-1)` before multiplying, or stack entropies to `(batch, horizon)` in `imagine_ahead`.

### 27. SAVi checkpoints saved on a GPU cannot be loaded on a CPU-only machine (fixed)

**Status:** fixed (uncommitted). `load_autoencoder` passes `map_location="cpu"`; Lightning moves the model to its device afterwards.

`load_autoencoder` (`sold/train_autoencoder.py:56-58`) calls `AutoencoderModule.load_from_checkpoint` without `map_location`, so tensors are restored onto the device they were saved from.
On a machine without CUDA, `train_sold.py` and `evaluate_sold.py` fail while loading the SAVi model.

**Fix:** pass `map_location="cpu"`; Lightning moves the model to the right device afterwards.

---

## Found by a code review against the paper

Items 28–37 come from `/code-review high .` with `PAPER_NOTES.md` as the specification.
Line numbers refer to commit `d7238d5`, before the fixes.

### 28. The actor's mean can never exceed about ±0.76

**Status:** open. A fix was applied and then rolled back, because it changes the behavior of existing checkpoints and runs. The bounds stay non-persistent buffers (#37).

`GaussianPredictor.forward` (`sold/modeling/sold/prediction.py:81-83`) clamps the raw mean to the action bounds `[-1, 1]` and then applies `tanh`.
The mean therefore never exceeds `tanh(1) ≈ 0.76`, and its gradient is zero outside the clamp.
In evaluation the actor uses the distribution's mode, so the arm can never move at full speed, and a mean that drifts past ±1 stops learning.

**Fix:** squash the mean into the bounds, `low + (high − low) · (tanh(x) + 1) / 2`, which equals `tanh(x)` for `[-1, 1]`.
Existing checkpoints would load unchanged, but their actor outputs change wherever the raw mean is outside `[-1, 1]`: up to 0.099 on the released Reach checkpoint, so the golden reference `tests/golden/data/sold_reach_red.pt` would have to be regenerated.

### 29. Imagined actions are not limited to the action space

`imagine_ahead` (`sold/train_sold.py:222-228`) feeds `action_dist.rsample()` to the dynamics model unclamped.
With a standard deviation of up to 1.0, imagined actions reach ±2 or more, while the dynamics model is trained only on real actions, which `select_action` clamps to `[-1, 1]`.
The actor's gradients then come from dynamics predictions far outside the training distribution, which invites it to exploit model errors.

**Fix options:** clamp with a straight-through estimator, or squash the sampled action (for example, a tanh-transformed distribution). Either changes the training dynamics.

### 30. Teacher forcing lets the dynamics model see the frame it predicts

**Status:** open. A fix (`batched_processing = False` by default in `AutoregressiveWrapper`, so teacher forcing runs step by step) was applied and then rolled back.

`OCVPSeqLayer.forward` (`sold/modeling/sold/dynamics.py:126-141`) applies temporal attention without a causal mask; `time_mask` is never passed.
In the batched teacher-forcing path (`AutoregressiveWrapper.predict_slots`, `dynamics.py:22-25`), the inputs include the ground-truth frames being predicted, so each prediction can attend to its own target.
The default `teacher_forcing: False` path is unaffected, because its inputs contain only past frames.

**Fix:** pass a causal mask to `time_encoder_block`, or raise an error when `teacher_forcing: True`.

### 31. Slot Attention scales by the wrong dimension

`Corrector` (`sold/modeling/autoencoder/savi/corrector.py:16`) uses `feature_dim ** -0.5` (64 → 1/8) to scale `q · k`, but q and k have `slot_dim` (128) channels.
Eq. 1 and SAVi scale by the square root of the q·k dimension, so attention logits are √2 larger and slots compete more sharply than intended.
Changing it alters every pretrained SAVi checkpoint's behavior.

### 32. ALiBi slopes differ from the original formula

`AlibiMask.get_slopes` (`sold/modeling/attention_mask.py:84-94`) starts the geometric sequence at `2 ** (-2 ** -(log2(n) - 2))`; Press et al. use `- 3`.
With 8 heads the slopes run from 2^-0.5 to 2^-4 instead of 2^-1 to 2^-8, so every head penalizes distance more strongly and none attends almost uniformly over long histories.
Changing it alters the behavior of trained actors, critics and reward models.

### 33. The two-hot mean is not the expectation of Eq. 5

`TwoHotEncodingDistribution.mean` (`sold/modeling/distributions.py:38-40`) returns `symexp(E[bins])`, the DreamerV3/sheeprl convention.
Eq. 5 defines `r̂ = softmax(f)ᵀ · symexp(bins)`, the expectation in raw space.
For a spread-out distribution the code's value is lower: with half the mass on 0 and half on 1, the code gives about 0.41 instead of 0.5. Reward and value estimates, and so the λ-returns, are biased low.

### 34. The target critic accumulated unused gradients (fixed)

`critic_target` (`sold/train_sold.py:50`) is a deep copy of the critic whose parameters still required gradients.
The actor loss backpropagated through `predicted_values_targ` (`:243`) into it on every step: wasted compute, and `.grad` tensors that were never used or reset.

**Fix (applied):** `critic_target.requires_grad_(False)`; it is updated only by the EMA.

### 35. `batch_size: 1` crashed the λ-return computation (fixed)

`.mean.squeeze()` (`sold/train_sold.py:232-233, 243`) also removed the batch dimension when it was 1, so `compute_lambda_returns` raised `IndexError`.

**Fix (applied):** `.squeeze(-1)`.

### 36. The attention mask was rebuilt in every transformer layer (fixed)

`SlotAggregationTransformer.forward` (`sold/modeling/sold/slot_aggregation_transformer.py:46-47`) sliced and repeated the `(batch · heads, tokens, tokens)` mask once per layer instead of once per call.
This was avoidable allocation on the hot path: 15+ actor calls per training step.

**Fix (applied):** the mask is built once before the layer loop.

### 37. The actor's bounds were copied to the GPU on every call (fixed)

`GaussianPredictor` kept `lower_bound` and `upper_bound` as plain CPU tensors and called `.to(device)` on every forward pass (`sold/modeling/sold/prediction.py:79-81`): two host-to-device copies per actor call.

**Fix (applied):** non-persistent buffers, which move with the module and do not change the checkpoint keys.

## Found while checking environment seeding

Items 38–39 follow from the seeding fix (#11) and the environment cache (#12). Neither has a test yet.
Line numbers refer to the working tree.

### 38. `evaluate_sold.py` replays the scenes training started with

`configs/evaluate_sold.yaml:17` seeds the environment with the same `seed` (42) as `configs/train_sold.yaml:56`.
Scene randomization (gripper start, object colors and positions, goal) comes from each environment's own `np_random`, so evaluation sees exactly the scene sequence the training run began with.
The model was trained on those scenes long ago, so this rarely matters, but evaluation is not on scenes that are guaranteed to be held out.

A related trap: each distractor variant is seeded with `seed + number of distractors` (`sold/envs/from_mof.py`), so neighboring seeds share streams (run `seed=1` with 1 distractor and run `seed=2` with 0 distractors both use generator seed 2). Runs meant to be independent seeds are slightly correlated.

**Possible fix:** a separate `eval_seed` config value (or an offset added to `seed` in code), and spaced-out variant seeds such as `seed * 1000 + n`.

### 39. Evaluation during training shares the environment with training

`OnlineModule.play_episode` (`sold/utils/training.py:212-218`) runs evaluation episodes, and the three logged training episodes, on `self.env`, the same environment used to collect training data (`:162, 170`).
This was already true before the fixes.

- Evaluation episodes advance the same random generators as training, so the evaluation scenes differ from checkpoint to checkpoint. Success rates of different checkpoints are measured on different scenes, which adds noise to the learning curve.
- Whether evaluation runs (`eval_freq`, `num_eval_episodes`) changes the sequence of training scenes, so changing evaluation settings changes training even with the same seed.
- Evaluation also changes the global `random` state used to choose the number of distractors.

**Possible fix:** a separate evaluation environment with its own seed, reset to a fixed seed before each evaluation so every checkpoint is scored on the same scenes.
