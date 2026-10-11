# Not my Dent

Team Everest's rental vehicle damage claim demo. The interface and API are runnable without CUDA, model weights, LangChain, YOLO, or paid vision APIs. **All verdicts, confidence numbers, and before/after frames are simulated. Uploaded videos are accepted but not inspected.** An uploaded vendor photo appears in the third panel of the generated proof image.

## Layout

```text
not-my-dent/
├── app/
│   ├── main.py                 # FastAPI application, HTML, static files
│   ├── routes/claims.py        # Upload, job polling, proof PNG endpoint
│   ├── pipeline/mock.py        # Replaceable mock processor
│   ├── templates/index.html    # Responsive single-page interface
│   └── static/                 # CSS and vanilla JavaScript
├── deploy/
│   ├── nginx-notmydent.conf
│   └── notmydent.service
├── tests/test_flow.py
└── pyproject.toml
```

The `uploads/` concept is implemented as a unique temporary directory under the OS temp directory per request, rather than a project directory. Input files are removed as soon as the job finishes. The PNG and report live in process memory for 20 minutes after completion, then expire on the next API request. The service has **one Uvicorn worker** because its job registry is in memory. A restart loses active jobs and reports. In production, use a durable job queue and object storage if jobs must survive restarts or run across workers.

## Local run

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd not-my-dent
uv python install 3.12
uv venv --python 3.12
uv pip install -e '.[test]'
.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>, click **Load demo inputs**, then **Run verification**. To check the API directly:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/verify-claim -F demo=true -F 'claim_text=Deep dent on front passenger bumper'
# Copy job_id from the response:
curl -sS http://127.0.0.1:8000/api/jobs/JOB_ID
curl -o proof.png http://127.0.0.1:8000/api/jobs/JOB_ID/proof
```

For uploaded files, omit `demo=true` and send `-F video_before=@before.mp4 -F video_after=@after.mp4 -F claim_image=@photo.jpg` with `claim_text`. Limits: 220 MiB for each video, 15 MiB for the photo, and 500 MiB total at Nginx. The FastAPI endpoint returns HTTP 202 and a `status_url`; poll until `state=complete`, then fetch `report.proof_image_url` for a PNG. The image endpoint sends `Cache-Control: no-store`. Demo mode needs no files and generates sample frames and a sample vendor panel on demand.

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

The mock demo does not need a CUDA toolkit. When integrating models, select the CUDA/PyTorch wheels supported by the installed driver and check the framework independently, e.g. `python -c 'import torch; print(torch.cuda.is_available())'` after installing PyTorch. On a Vultr managed/vGPU image, follow the image's vendor driver/licensing guidance before replacing a working driver.

### 2. Install and start the application

Transfer the project archive to the VM (run on your local machine):

```bash
scp not-my-dent-team-everest.zip YOUR_SSH_USER@SERVER_IPV4:/tmp/
```

On the VM:

```bash
sudo apt update
sudo apt install -y curl ca-certificates nginx snapd unzip
sudo useradd --system --user-group --home-dir /opt/notmydent --shell /usr/sbin/nologin notmydent
sudo mkdir -p /opt/notmydent
cd /tmp
unzip -q not-my-dent-team-everest.zip
sudo cp -a /tmp/not-my-dent/. /opt/notmydent/
sudo chown -R notmydent:notmydent /opt/notmydent
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
sudo mkdir -p /opt/notmydent-python
sudo chown notmydent:notmydent /opt/notmydent-python
UV_PYTHON_INSTALL_DIR=/opt/notmydent-python uv python install 3.12
cd /opt/notmydent
UV_PYTHON_INSTALL_DIR=/opt/notmydent-python uv venv --clear --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
sudo chown -R notmydent:notmydent /opt/notmydent /opt/notmydent-python
.venv/bin/python --version
.venv/bin/python -c 'import fastapi, matplotlib, PIL; print("Dependencies OK")'
```

The managed Python lives under `/opt/notmydent-python` so the `notmydent` service account can access its interpreter. `--clear` replaces an existing Python 3.11 virtual environment when upgrading. If the server already has an appropriate system Python 3.12, you can instead use `uv venv --clear --python /usr/bin/python3.12 .venv`.

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

### 4. Install systemd and Nginx

```bash
sudo cp /opt/notmydent/deploy/notmydent.service /etc/systemd/system/notmydent.service
sudo systemctl daemon-reload
sudo systemctl enable --now notmydent
sudo systemctl status notmydent --no-pager
curl -sS http://127.0.0.1:8000/healthz
sudo cp /opt/notmydent/deploy/nginx-notmydent.conf /etc/nginx/sites-available/notmydent
sudo ln -s /etc/nginx/sites-available/notmydent /etc/nginx/sites-enabled/notmydent
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl reload nginx
```

For an **existing Not my Dent installation**, stop the service before copying this release and rerun the Python 3.12 environment commands above, then restart it. Old in-memory jobs and proof images will be lost during the restart:

```bash
sudo systemctl stop notmydent
sudo cp -a /tmp/not-my-dent/. /opt/notmydent/
cd /opt/notmydent
UV_PYTHON_INSTALL_DIR=/opt/notmydent-python uv python install 3.12
UV_PYTHON_INSTALL_DIR=/opt/notmydent-python uv venv --clear --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
sudo chown -R notmydent:notmydent /opt/notmydent /opt/notmydent-python
.venv/bin/python --version
sudo systemctl start notmydent
curl -sS http://127.0.0.1:8000/healthz
```

The full service and server blocks are in `deploy/notmydent.service` and `deploy/nginx-notmydent.conf`. Nginx allows 500M requests, streams them upstream, gives uploads 300-second idle timeouts, and gives polling/report requests a 300-second upstream timeout. There is no long model request in the mock implementation. Make sure `/tmp` has enough free disk for FastAPI's spooled multipart data and the per-job copies; allow at least ~1 GiB free per concurrent maximum-size upload plus additional headroom. Nginx's configured 500M limit is decimal megabytes; two 220 MiB videos plus one 15 MiB image fit below it.

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

If `/usr/bin/certbot` exists already, skip the `ln -s` line. Certbot's snap installs automatic renewal. Check service logs with `sudo journalctl -u notmydent -f`; view the final site using `https://notmydent.biz`.

## Integrating the Colab pipeline later

The project targets Python 3.12, matching the pipeline's Colab interpreter. Qwen3-V-4B, YOLO, PyTorch, and related model packages are intentionally **not** included in `pyproject.toml`. When adding them, reproduce the tested model package versions and choose a PyTorch CUDA build compatible with the Vultr NVIDIA driver; Colab's preinstalled CUDA stack and GPU drivers may differ from the server. Keep the mock path available for lightweight smoke tests. GPU memory needs and inference duration will also affect the queue design once the real processor is connected.

## Replacing the mock pipeline

Replace `process_claim_pipeline(...)` in `app/pipeline/mock.py`, or point the import in `app/routes/claims.py` at your implementation. Keep the signature `(video_before: Path | None, video_after: Path | None, claim_text: str, claim_image: Path | None) -> tuple[dict, bytes]`. The video and photo paths are valid only for the duration of the call. Generate the proof PNG into `io.BytesIO`, return `output.getvalue()`, and keep the report JSON serializable. If real inference takes minutes or needs multiple workers/restart recovery, move job execution to a queue and replace the in-memory registry with durable status storage. Replace the mock labels and confidence text only after real validation.

## Test

```bash
uv pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

References for server setup: [Astral uv installation](https://docs.astral.sh/uv/getting-started/installation/), [uv Python versions](https://docs.astral.sh/uv/guides/install-python/), [Ubuntu NVIDIA server driver guide](https://ubuntu.com/server/docs/nvidia-drivers-installation/), [Porkbun DNS records](https://kb.porkbun.com/article/231-how-to-add-dns-records-on-porkbun), [Nginx proxy directives](https://nginx.org/en/docs/http/ngx_http_proxy_module.html), [Certbot Nginx instructions](https://certbot.eff.org/instructions?os=snap&ws=nginx).
