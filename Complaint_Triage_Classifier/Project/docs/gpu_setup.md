# GPU Setup for Transformer Training

The transformer training scripts use Hugging Face `Trainer`. When a CUDA-enabled PyTorch build is installed, `Trainer` automatically places the model and batches on the GPU.

## 1. Verify that Windows can see the GPU

```powershell
nvidia-smi
```

If this command fails, install or update the NVIDIA driver first.

## 2. Verify that PyTorch can see CUDA

From the activated project environment:

```powershell
python scripts/10_check_gpu.py
```

A working GPU environment should print something like:

```json
{
  "cuda_available": true,
  "selected_device": "cuda:0",
  "device_name": "NVIDIA GeForce RTX ..."
}
```

## 3. Reinstall PyTorch with CUDA support if needed

If `cuda_available` is `false`, the environment probably has a CPU-only PyTorch build.

From the activated virtual environment:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Then run:

```powershell
python scripts/10_check_gpu.py
```

## 4. Run transformer training

```powershell
python scripts/05_train_transformer.py --task triage --config configs/distilbert.yaml --train data/processed/train.csv --val data/processed/val.csv --output-dir models/distilbert/triage
```

During startup, the logs should show:

```text
PyTorch device summary: ... "selected_device": "cuda:0" ...
Hugging Face Trainer device: cuda:0
```

## 5. Force CPU only when debugging

Set this in the model config if you explicitly want CPU mode:

```yaml
training:
  force_cpu: true
```
