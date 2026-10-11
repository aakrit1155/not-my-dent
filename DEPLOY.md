# Deploying Not my Dent

This guide deploys the app on a Linux GPU server with Docker, behind Nginx, with HTTPS.

**Server requirements**

- Ubuntu 22.04 or 24.04
- NVIDIA GPU with at least 16 GB of VRAM (A10, L4, A40, L40S, …), driver 525 or newer
- About 40 GB of free disk (Docker image plus ~17 GB of model weights)
- Inbound TCP ports 22, 80 and 443 open in the cloud firewall

Run all commands on the server unless a step says otherwise.

## 1. Check the GPU driver

```bash
nvidia-smi
```

If it lists your GPU with driver 525 or newer, go to step 2. Otherwise install the driver and reboot:

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install --gpgpu
sudo reboot
```

The host doesn't need a CUDA toolkit. The Docker image ships PyTorch's CUDA 12.6 runtime.

## 2. Install Docker and the NVIDIA Container Toolkit

```bash
curl -fsSL https://get.docker.com | sudo sh

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit nginx git
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Check that containers can see the GPU:

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

## 3. Get the code and start the app

```bash
sudo git clone -b develop-febe https://github.com/kshitizregmi/not-my-dent.git /opt/notmydent
cd /opt/notmydent
sudo docker compose up -d --build
```

If the repository is private, clone with an SSH deploy key or a personal access token.

On first start the app downloads about 17 GB of model weights. Follow the logs:

```bash
sudo docker compose logs -f
```

Then check the model status:

```bash
curl http://127.0.0.1:8000/healthz
```

Wait until it reports `"models":"ready"`. The website is up before that; jobs submitted while the models load wait in the queue.

Optional: set a Hugging Face token to avoid download rate limits:

```bash
echo "HF_TOKEN=hf_xxx" | sudo tee /opt/notmydent/.env
sudo docker compose up -d
```

## 4. Configure Nginx

The container listens only on `127.0.0.1:8000`. Nginx makes it reachable on ports 80 and 443.

```bash
sudo cp /opt/notmydent/deploy/nginx-notmydent.conf /etc/nginx/sites-available/notmydent
sudo ln -sf /etc/nginx/sites-available/notmydent /etc/nginx/sites-enabled/notmydent
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

If UFW is active:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
```

The server name in `deploy/nginx-notmydent.conf` is `notmydent.biz`. If you use a different domain, edit that file before copying it.

## 5. Point the domain at the server

At your DNS provider, add:

| Type  | Host            | Value            |
|-------|-----------------|------------------|
| A     | *(blank, apex)* | `SERVER_IPV4`    |
| CNAME | `www`           | `notmydent.biz`  |

Remove any conflicting A, AAAA or CNAME records for the apex or `www`. Check propagation:

```bash
dig +short A notmydent.biz
```

## 6. Enable HTTPS

Run this once DNS resolves to the server:

```bash
sudo snap install --classic certbot
sudo ln -sf /snap/bin/certbot /usr/bin/certbot
sudo certbot --nginx -d notmydent.biz -d www.notmydent.biz --redirect
sudo certbot renew --dry-run
```

Certbot renews the certificate automatically.

## 7. Smoke test

Open `https://notmydent.biz` and upload the sample files from the repository:

- Before video: `data/before.mp4`
- After video: `data/after.mp4`
- Company photo: `data/scratch.png`
- Claim: **Scratches on the front bumper**

The expected verdict is **POSSIBLE NEW DAMAGE**, with evidence images for the company photo, the before frame and the after frame.

Or test the API from the server:

```bash
cd /opt/notmydent
curl -sS -X POST http://127.0.0.1:8000/api/verify-claim \
  -F video_before=@data/before.mp4 -F video_after=@data/after.mp4 \
  -F claim_image=@data/scratch.png -F 'claim_text=Scratches on the front bumper'
# Copy job_id from the response, then poll until "state": "complete":
curl -sS http://127.0.0.1:8000/api/jobs/JOB_ID
```

## Updating

```bash
cd /opt/notmydent
sudo git pull
sudo docker compose up -d --build
```

Model weights live in the `models` Docker volume, so updates and restarts only reload them; nothing is downloaded again. Restarting drops any jobs in progress.

## Operations

| Task | Command |
|------|---------|
| Live logs | `sudo docker compose logs -f` |
| Model status | `curl http://127.0.0.1:8000/healthz` |
| Restart | `sudo docker compose restart` |
| Stop | `sudo docker compose down` |
| GPU usage | `nvidia-smi` |

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `/healthz` shows `"models":"failed"` | Read the `error` field and `docker compose logs`. Usually the container can't see the GPU: rerun the GPU check in step 2. |
| `could not select device driver "nvidia"` | The NVIDIA Container Toolkit isn't configured. Rerun `sudo nvidia-ctk runtime configure --runtime=docker` and restart Docker. |
| CUDA out of memory | Another process is using the GPU (check `nvidia-smi`), or the GPU has less than 16 GB. |
| Uploads fail with HTTP 413 | The file is over the limit (220 MiB per video, 15 MiB per photo), or Nginx wasn't reloaded after copying the config. |
| Result is INCONCLUSIVE with "Could not locate the claimed damage" | The claim's damage type doesn't match what YOLO detected in the photo. For example, the sample photo is detected as a *scratch*, so a claim saying only "dent" finds nothing. |
| Jobs stay "queued" | The models are still loading (first start), or another job is running. Only one job runs at a time. |

## Without Docker (bare metal)

1. Create a Python 3.12 virtualenv in `/opt/notmydent`.
2. Install PyTorch:
   ```bash
   pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
   ```
3. Install the app:
   ```bash
   pip install -e '.[ml]'
   ```
4. Create the service account and start the systemd unit:
   ```bash
   sudo useradd --system --user-group --home-dir /opt/notmydent --shell /usr/sbin/nologin notmydent
   sudo chown -R notmydent:notmydent /opt/notmydent
   sudo cp deploy/notmydent.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now notmydent
   ```

The unit stores model weights in `/var/lib/notmydent/huggingface`. Then follow steps 4–6 above for Nginx, DNS and HTTPS.
