# Appendix D. Deploying on a $5/month VPS

Everything in the book has run on your laptop. Some fraction of what you build, you will want other people to be able to use. This appendix is a practical guide to putting a small LangGraph or LlamaIndex app in front of users on the cheapest VPS you can rent, with no managed services in the loop.

## What "$5/month VPS" actually gets you in 2026

The current entry-level tier at the major low-cost providers (Hetzner, DigitalOcean, Linode, Vultr, OVH) is around 1 shared vCPU, 1-2 GB of RAM, 25-50 GB SSD, and 500 GB - 1 TB of bandwidth, for $4-6 per month.

That is enough to run:

- A small LangGraph or LlamaIndex Workflow app talking to a **hosted LLM API** (Gemini, Fireworks.ai, OpenAI). The VPS is not doing inference; it is doing routing, prompt assembly, tool execution, and response handling.
- A local vector store (Chroma, sqlite-vec, LanceDB) with a few hundred thousand embeddings.
- A local embedding model on CPU (BGE-small runs fine on 1 vCPU; expect ~100-500 ms per embedding call).
- A reasonable number of concurrent users, on the order of dozens.

That is not enough to run:

- A local LLM. Even a 3B parameter quantized model wants 3-4 GB of RAM to load and is painfully slow on 1 vCPU. If you need local inference, you need either a beefier VPS (32+ GB RAM, dedicated CPU) or a GPU instance, both of which cost 10-30× more.

The right pattern for a $5 VPS in 2026 is: **local retrieval, hosted inference**. Your embeddings and vector store are on the VPS; your chat model is Gemini or Fireworks.ai over their API.

## The stack

- **Ubuntu 24.04 LTS** (default on most providers).
- **Python 3.12** via `uv`. Install with `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **Your app** as a normal `uv sync`-able project. `git clone` your repo, `uv sync`, done.
- **Caddy** as reverse proxy. One line of config, automatic HTTPS via Let's Encrypt.
- **Systemd** for supervision. Ten lines of unit file, restart-on-crash, journal for logs.

Everything above is open source. Nothing costs money except the VPS rental.

## Systemd unit for a LangGraph app

Assuming your app is a FastAPI server that listens on `127.0.0.1:8000`, the same shape as the deployment chapter's `01_serve_workflow.py`:

```ini
# /etc/systemd/system/myapp.service
[Unit]
Description=My LangChain app
After=network.target

[Service]
Type=simple
User=myapp
WorkingDirectory=/home/myapp/repo
ExecStart=/home/myapp/.local/bin/uv run app.py
Restart=on-failure
RestartSec=5
Environment=OPENAI_API_KEY=sk-...
# Add other environment variables here.

[Install]
WantedBy=multi-user.target
```

`sudo systemctl enable myapp && sudo systemctl start myapp`. `journalctl -u myapp -f` for logs.

## Caddy for HTTPS

```text
# /etc/caddy/Caddyfile
myapp.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

`sudo systemctl restart caddy`. Caddy handles the Let's Encrypt certificate request, renewal, and HTTPS termination automatically. Free.

## Cost math

A concrete example. Suppose your app makes 10,000 LLM calls per month, averaging 1,000 tokens in and 500 tokens out per call.

- **VPS:** $5/month.
- **Gemini 2.5 Flash** or **GPT-4o-mini** at those volumes: roughly $2-8/month depending on model choice and exact token mix.
- **Domain name:** ~$12/year, so $1/month.
- **Backups** (a cheap object storage bucket for daily snapshots of the vector store and any persistent data): $0.50/month.

Total: under $15/month for an app that serves 10,000 real interactions. If usage doubles, the VPS bill stays the same and only the LLM bill scales; even at 100,000 calls/month you are still under $60/month total.

For comparison: a single-seat LangSmith Plus subscription in 2026 is $39/month, plus per-trace usage. That budget alone covers your entire hosting and inference stack in this setup.

## Backups

The one thing worth spending time on. Whatever the app persists (SQLite databases from Chapter 5, Chroma directories, LlamaIndex `storage/` folders) should be backed up daily. Simplest reliable pattern:

```bash
# /etc/cron.daily/myapp-backup
#!/bin/bash
tar czf /tmp/myapp-$(date +%Y%m%d).tar.gz /home/myapp/data
rclone move /tmp/myapp-*.tar.gz remote:myapp-backups/
find /tmp -name 'myapp-*.tar.gz' -mtime +7 -delete
```

`rclone` supports every object storage provider; a 10 GB backup bucket at Backblaze B2 or Cloudflare R2 is under $0.50/month. Test the restore path at least once: an untested backup is not a backup.

## When to leave the $5 VPS

The point of the setup above is that it is enough for most personal projects and early client work. Signals it is time to move up:

- **Your vector store no longer fits in RAM.** Chroma is memory-mapped; if the index is bigger than free RAM, queries slow down dramatically.
- **You need concurrent users measurable in hundreds.** Move to a 4 GB / 2 vCPU tier (~$20/month), or move the LLM calls to a background queue so the web server does not block.
- **You need local inference.** This is the big one. Rent a Hetzner dedicated CPU server (32 GB RAM, ~$30/month) for medium local models, or a GPU instance if you need low-latency inference.

None of these move you to a managed service; they just move you to a bigger box.
