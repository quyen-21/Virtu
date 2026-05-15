$env:ARTIFACT_DIR = if ($env:ARTIFACT_DIR) { $env:ARTIFACT_DIR } else { "./artifacts" }
python -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload

