# Not my Dent

Team Everest's rental vehicle damage claim checker. Upload a BEFORE walkaround video, an AFTER return video, the rental company's damage photo, and their written claim. The server runs the NotMyDent engine from `notmydent.ipynb` (YOLOv8 damage detection → CLIP + SIFT frame alignment → Qwen3-VL-8B comparison) and shows the verdict, timestamps, descriptions, detector scores and evidence images.

A CUDA GPU is required (the 4-bit Qwen3-VL-8B needs roughly 8–10 GB of VRAM; an A40 / L40S is plenty).

## Layout

```text
not-my-dent/
├── app/
│   ├── main.py                 # FastAPI app; preloads models in the background at startup
│   ├── routes/claims.py        # Upload, job polling, evidence-image endpoints
│   ├── pipeline/engine.py      # The notebook engine (run_notmydent), logic unchanged
│   ├── pipeline/runner.py      # Loads models once, runs one GPU job at a time
│   ├── templates/index.html    # Single-page interface
│   └── static/                 # CSS and vanilla JavaScript
├── data/                       # Sample before/after videos and claim photo
├── deploy/                     # Nginx site + systemd unit (bare-metal option)
├── Dockerfile
├── docker-compose.yml
└── tests/test_flow.py          # HTTP flow tests with a fake engine (no GPU needed)
```

**How it runs.** On startup a background thread loads YOLO, CLIP and Qwen3-VL into GPU memory once; `/healthz` reports `models: loading | ready | failed`. Every request reuses those models. Jobs run one at a time behind a lock (up to 3 may be queued). Uploaded inputs are deleted when a job finishes; result images are kept for 1 hour and served from `/api/jobs/{id}/files/...`. Jobs live in memory, so run **one Uvicorn worker**; a restart drops in-flight jobs.

The only changes to the notebook engine: `load_clip()` / `load_vlm()` cache models (like `load_detector()` already did) instead of loading and freeing them on every call, YOLO is no longer moved to the CPU after detection, so it stays ready for the next request, and matplotlib uses the non-GUI `Agg` backend so figures can be saved from worker threads.

## API

```bash
curl -sS -X POST http://127.0.0.1:8000/api/verify-claim \
  -F video_before=@data/before.mp4 -F video_after=@data/after.mp4 -F claim_image=@data/scratch.png \
  -F 'claim_text=Scratches on the front bumper'
# → 202 {"job_id": ..., "status_url": "/api/jobs/JOB_ID"}
curl -sS http://127.0.0.1:8000/api/jobs/JOB_ID
# → {"state": "queued" | "processing"} … then {"state": "complete", "report": {...}}
```

`report` is exactly what `run_notmydent()` returns (`claim`, `overall_verdict`, `findings[]` with `company` / `before` / `after` descriptions, statuses, timestamps, detector confidences, and `comparisons`). Image paths in it are rewritten to URLs. Limits: 220 MiB per video, 15 MiB for the photo. With the sample data and that claim, the expected verdict is **POSSIBLE NEW DAMAGE**.

## Run with Docker (recommended)

Needs the NVIDIA driver plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

```bash
docker compose up -d --build
docker compose logs -f          # first start downloads ~17 GB of Qwen3-VL weights into the "models" volume
curl -sS http://127.0.0.1:8000/healthz
```

Open <http://127.0.0.1:8000>. The site is usable immediately; jobs submitted while models load wait in the queue. Weights persist in the `models` volume, so later restarts only take the load time. Set `HF_TOKEN` in the environment to avoid Hugging Face rate limits.

## Local development (UI only, no GPU)

```bash
uv venv --python 3.12
uv pip install -e '.[test]'
.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
.venv/bin/python -m pytest -q
```

Without the `ml` extra the page and API work, `/healthz` shows `models: failed`, and submitted jobs fail. Install `.[ml]` on a CUDA machine for real inference.

## Vultr Ubuntu 22.04 / 24.04 deployment

The commands below assume a fresh instance, SSH access as an account with `sudo`, and the distributable `not-my-dent-team-everest.zip` archive. Replace **SERVER_IPV4** with the instance's public IPv4 address. Open inbound TCP 22, 80, and 443 in the Vultr firewall and any Ubuntu firewall. Port 8000 is bound to localhost and should remain private.

### 1. Check the A40 / L40S and driver

```bash
lspci | grep -i nvidia
nvidia-smi
# Optional toolkit check, only if a CUDA toolkit was already installed:
command -v nvcc && nvcc --version || true
```

If `nvidia-smi` works and shows the GPU, retain the image's existing driver. Its “CUDA Version” is the highest CUDA API version supported by that driver, **not** proof that the toolkit is installed. If no working driver is installed, inspect Ubuntu's server driver choices and install the recommended GPGPU driver, then reboot:

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers list --gpgpu
sudo ubuntu-drivers install --gpgpu
sudo reboot
# After reconnecting:
nvidia-smi
```

You don't need a CUDA toolkit on the host: the Docker image ships PyTorch's CUDA 12.6 runtime, which needs driver 525 or newer. To check the GPU inside the container, run `sudo docker compose exec notmydent python -c 'import torch; print(torch.cuda.is_available())'`. On a Vultr managed/vGPU image, follow the image's vendor driver/licensing guidance before replacing a working driver.

### 2. Install and start the application

Copy the project to the VM (e.g. `scp -r not-my-dent YOUR_SSH_USER@SERVER_IPV4:/opt/notmydent`), then install Docker and the NVIDIA Container Toolkit and start it:

```bash
curl -fsSL https://get.docker.com | sudo sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit nginx snapd
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
cd /opt/notmydent
sudo docker compose up -d --build
curl -sS http://127.0.0.1:8000/healthz
```

The container restarts automatically (`restart: unless-stopped`) and listens only on `127.0.0.1:8000`. To update: copy the new code and rerun `sudo docker compose up -d --build`.

**Without Docker:** create a Python 3.12 venv in `/opt/notmydent`, install `torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126`, then `pip install -e '.[ml]'`, and use `deploy/notmydent.service` (it sets `HF_HOME=/var/lib/notmydent/huggingface` so weights are cached in a writable directory).

### 3. Configure Porkbun DNS

In **Porkbun → Domain Management → notmydent.biz → DNS**, set:

| Type | Host | Answer / Value | TTL |
|---|---|---|---|
| A | *(blank, apex)* | `SERVER_IPV4` | Default |
| CNAME | `www` | `notmydent.biz` | Default |

Remove conflicting web A/AAAA/CNAME/ALIAS records for the apex or `www` (including parked site records). Preserve unrelated mail/TXT/MX records. If you have a configured public IPv6 address, you may add an AAAA apex record; otherwise remove stale AAAA web records. Check propagation:

```bash
sudo apt install -y dnsutils
dig +short A notmydent.biz
dig +short CNAME www.notmydent.biz
```

### 4. Install Nginx

If you're not using Docker, first install the unit: `sudo cp /opt/notmydent/deploy/notmydent.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now notmydent`.

```bash
sudo cp /opt/notmydent/deploy/nginx-notmydent.conf /etc/nginx/sites-available/notmydent
sudo ln -s /etc/nginx/sites-available/notmydent /etc/nginx/sites-enabled/notmydent
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl reload nginx
```

Nginx allows 500M uploads, streams them to the app, and uses 300-second timeouts. Analysis itself runs in the background while the browser polls, so long jobs don't hit proxy timeouts. Keep at least ~1 GiB free in `/tmp` per concurrent maximum-size upload.

If UFW is active:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw status
```

### 5. Issue and renew HTTPS certificates

Wait until both DNS names resolve to the VM and HTTP port 80 is reachable. Install the official Certbot snap and let it update the Nginx site and redirect HTTP to HTTPS:

```bash
sudo snap install core
sudo snap refresh core
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/bin/certbot
sudo certbot --nginx -d notmydent.biz -d www.notmydent.biz --redirect
sudo certbot renew --dry-run
sudo nginx -t
curl -I https://notmydent.biz/
```

If `/usr/bin/certbot` exists already, skip the `ln -s` line. Certbot's snap installs automatic renewal. Check logs with `sudo docker compose logs -f` (or `sudo journalctl -u notmydent -f` without Docker); view the final site using `https://notmydent.biz`.

References for server setup: [Astral uv installation](https://docs.astral.sh/uv/getting-started/installation/), [uv Python versions](https://docs.astral.sh/uv/guides/install-python/), [Ubuntu NVIDIA server driver guide](https://ubuntu.com/server/docs/nvidia-drivers-installation/), [Porkbun DNS records](https://kb.porkbun.com/article/231-how-to-add-dns-records-on-porkbun), [Nginx proxy directives](https://nginx.org/en/docs/http/ngx_http_proxy_module.html), [Certbot Nginx instructions](https://certbot.eff.org/instructions?os=snap&ws=nginx).
