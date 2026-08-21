# PrimeBot Household Pi0.5 SFT

The PrimeBot SFT path streams each LeRobot v2.1 episode from parquet and MP4.
It does not construct a 32-million-frame global index and does not merge or
copy the four source datasets.

The input contract is:

- three RGB views resized with padding to `224 x 224`;
- all 89 normalized state dimensions encoded by Pi0.5's discrete-state prompt;
- absolute actions packed as raw indices `0:22` followed by `83:86` (25 values);
- a 30-step action horizon, padded by OpenPI from 25 to its pretrained 32-D head;
- a 512-token prompt budget.

For every task, episodes are sorted globally by `episode_index`. The last 100
episodes form the held-out set even when they span multiple `chunk-*`
directories. The infinite training stream can only see the preceding episodes.

Compute the five normalization assets (four task-specific plus one uniformly
weighted mixture) from the training split without decoding video. Held-out
episodes are excluded to avoid evaluation leakage:

```bash
./.venv/bin/python toolkits/lerobot/calculate_primebot_norm_stats.py
```

Convert the existing local LeRobot/OpenPI PyTorch checkpoint to the
OpenPI_RLinf layout. This command only reads local files; it does not download
Pi0.5 weights:

```bash
./.venv/bin/python -m rlinf.utils.ckpt_convertor.openpi.convert \
  --mode openpi_pytorch_to_openpi_rlinf \
  --input-model /mnt/workspace/base_model/pi05_base \
  --input-norm-stats /mnt/workspace/base_model/pi05_base_rlinf_torch/primebot/mixed_uniform/norm_stats.json \
  --output-model /mnt/workspace/base_model/pi05_base_rlinf_torch \
  --output-norm-stats /mnt/workspace/base_model/pi05_base_rlinf_torch/norm_stats.json
```

The PaliGemma tokenizer is a separate 4 MB asset. The prepared local cache for
this setup is `/mnt/workspace/base_model/pi05_base_rlinf_torch/openpi_cache`;
using it keeps worker startup offline. RLinf builds the model shape from the
experiment YAML, so the copied LeRobot `config.json` values do not override the
30-step horizon or 512-token budget.

Train the uniformly sampled four-task mixture:

```bash
OPENPI_DATA_HOME=/mnt/workspace/base_model/pi05_base_rlinf_torch/openpi_cache \
bash examples/sft/run_vla_sft.sh primebot_sft_openpi_pi05_mixed
```

For separate models, use `primebot_sft_openpi_pi05_task01`, `task02`, `task03`,
or `task09`. Each config selects its matching per-task normalization file.

Evaluation is a separate process and is never launched by the training configs,
so it cannot introduce a validation-time GPU-memory peak during training. Stop
training or use separate GPUs, then point the eval process at an RLinf actor
checkpoint directory containing `model_state_dict/full_weights.pt`:

```bash
OPENPI_DATA_HOME=/mnt/workspace/base_model/pi05_base_rlinf_torch/openpi_cache \
./.venv/bin/python examples/sft/train_vla_sft.py \
  --config-name primebot_eval_openpi_pi05_mixed \
  actor.model.model_path=/path/to/checkpoints/global_step_N/actor
```

The standalone job streams every frame from the globally last 100 episodes of
each task, with `data.eval_num_workers: 0` for an exact finite traversal. It
reports `eval/loss`, per-task `eval/loss/<task_name>`, and evaluated frame
counts. The loss is the same action flow-matching MSE used by SFT, with image
augmentation disabled in evaluation mode. As in training, flow noise and time
are sampled, so the reported loss is a Monte Carlo estimate.
