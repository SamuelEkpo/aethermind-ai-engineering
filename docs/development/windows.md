# Windows development notes

## Virtualenv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,ml]"
```

## API server

`make` is not required. Use the PowerShell helpers:

```powershell
# Terminal 1 — leave running
.\scripts\dev.ps1

# Optional reload (only watches apps/ + packages/, never .venv)
.\scripts\dev.ps1 -Reload
```

## Smoke test

```powershell
# Terminal 2
.\scripts\test_api.ps1
```

PowerShell aliases `curl` to `Invoke-WebRequest`. Prefer `Invoke-RestMethod` or the script above.

## Train / eval / generate

```powershell
.\.venv\Scripts\python.exe -m training.scripts.train_lm --config tiny --steps 500 --batch-size 8 --seq-len 64
.\.venv\Scripts\python.exe -m training.scripts.evaluate_lm
.\.venv\Scripts\python.exe -m training.scripts.generate --prompt "def " --max-new-tokens 40
```

After training, **restart the API** so it reloads the checkpoint.

## Avoid reload storms

Never run `uvicorn --reload` while installing into `.venv`. WatchFiles will restart on every site-packages write and the process will thrash.
