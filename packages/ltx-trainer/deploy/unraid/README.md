# Running the LTX-2 Trainer on Unraid

This image is a **toolbox container**, not a service. It starts idle (`tail -f /dev/null`);
you open its **Console** (`>_` icon in the Docker tab) and run the preprocessing and training
scripts by hand. That suits ML training, where you want to watch logs, stop, tweak, and re-run.

Tested target: **RTX PRO 6000 Blackwell (96 GB)** — sm_120, supported by the torch 2.11 / CUDA 13
stack baked into the image. With 96 GB you can use the standard configs (no low-VRAM config needed).

---

## 1. Prerequisites

- **Nvidia Driver plugin** installed (Unraid → Apps → search "Nvidia Driver", by *ich777*), then reboot.
  Verify on the Unraid terminal: `nvidia-smi` lists the RTX PRO 6000.
- The **LTX-2 checkpoint** (`.safetensors`) and the **Gemma text-encoder** directory downloaded
  onto a share (these are large — tens of GB — and are **not** in the image).

## 2. Install the container

Either:

- **Import the template:** copy `ltx-2-trainer.xml` to
  `/boot/config/plugins/dockerMan/templates-user/` on the Unraid box, then
  Docker → *Add Container* → pick **ltx-2-trainer** from the *Template* dropdown; or
- **Fill the Add Container form** manually with the values in the table below.

### Container fields

| Field | Value |
|---|---|
| **Repository** | `ghcr.io/ethanfel/ltx-2-trainer:latest` |
| **Network Type** | `bridge` (needed for W&B / Hub; no inbound ports) |
| **Extra Parameters** | `--runtime=nvidia --ipc=host --restart=no` |
| **Post Arguments** | `tail -f /dev/null` |

`--runtime=nvidia` hands the GPU to the container (the Nvidia plugin registers it).
`--ipc=host` gives the PyTorch DataLoader enough shared memory (the 64 MB Docker default
causes "DataLoader worker killed" / bus errors). `--restart=no` stops Unraid from
re-launching a finished run.

### Path mappings (volumes)

| Container path | Host (example) | Mode | Holds |
|---|---|---|---|
| `/models` | `/mnt/user/ai/ltx2/models` | **ro** | LTX-2 `.safetensors` + Gemma encoder dir |
| `/workspace` | `/mnt/user/ai/ltx2/workspace` | **rw** | datasets, preprocessed latents, YAML configs, outputs |

### Environment variables

| Variable | Value | Notes |
|---|---|---|
| `NVIDIA_VISIBLE_DEVICES` | `all` | or a UUID from `nvidia-smi -L` to pin one GPU |
| `NVIDIA_DRIVER_CAPABILITIES` | `all` | |
| `WANDB_API_KEY` | *(blank)* | optional; or set `WANDB_MODE=offline` to disable W&B |
| `HF_TOKEN` | *(blank)* | only if pushing the LoRA to the Hub |

## 3. Folder layout on the share

```
/mnt/user/ai/ltx2/
├── models/
│   ├── ltx-2-model.safetensors      ->  /models/ltx-2-model.safetensors
│   └── gemma-text-encoder/          ->  /models/gemma-text-encoder
└── workspace/
    ├── configs/  my_t2v_lora.yaml   ->  /workspace/configs/my_t2v_lora.yaml
    ├── dataset/                     (raw videos + captions to preprocess)
    ├── preprocessed/                (process_dataset.py output = preprocessed_data_root)
    └── outputs/                     (training output_dir: checkpoints, logs, validation)
```

## 4. Point the config at the mounts

In your training YAML (copy one from `configs/` as a starting point), use **container** paths:

```yaml
model:
  model_path: "/models/ltx-2-model.safetensors"
  text_encoder_path: "/models/gemma-text-encoder"
data:
  preprocessed_data_root: "/workspace/preprocessed"
  num_dataloader_workers: 4
output_dir: "/workspace/outputs/t2v_lora"
```

## 5. Run it (container Console)

```bash
# one-time: encode your dataset into latents + caption embeddings
python scripts/process_dataset.py --help     # see options, then run for your dataset

# train
python scripts/train.py /workspace/configs/my_t2v_lora.yaml
```

The Console opens in the trainer working dir (`/app/packages/ltx-trainer`), so `scripts/...`
and the bundled `configs/` are right there. Training writes to `/workspace/outputs/...`.

---

### Notes

- **File ownership:** the container runs as root, so files written to `/workspace` are
  root-owned. If that bothers your share permissions, run Unraid's *Tools → New Permissions*
  on the share afterward, or add `--user 99:100` to Extra Parameters (Unraid's `nobody:users`).
- **Long runs:** training can run for hours. The keep-alive container stays up; closing the
  Console does **not** stop training only if you launch it with `nohup ... &` or `tmux`.
  Otherwise the run is tied to the Console session.
- **Updating the image:** Unraid's *Force Update* (or `docker pull`) fetches a new
  `:latest`; your `/models` and `/workspace` data are untouched.
