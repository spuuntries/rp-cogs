"""
Cog predictor for the Queeree content moderation pipeline.

On setup, boots the Queeree Sanic server as a subprocess.
On predict, POSTs to the local server and streams back the result.
"""

import json
import os
import signal
import subprocess
import sys
import time

import httpx
from cog import BasePredictor, Input, Path, Secret
from pydantic import SecretStr

QUEEREE_APP_DIR = os.path.join(os.path.dirname(__file__), "queeree", "app")
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
ANALYZE_URL = f"{SERVER_URL}/api/analyze"

# How long to wait for the server to become healthy on setup
STARTUP_TIMEOUT = 300  # seconds


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Start the Queeree Sanic server and wait until it's healthy."""
        env = os.environ.copy()

        # Make sure the API key is forwarded
        if "OPENROUTER_API_KEY" not in env:
            print(
                "WARNING: OPENROUTER_API_KEY not set — the pipeline will "
                "fail on LLM calls unless the key is provided at runtime."
            )

        self.server_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sanic",
                "main:app",
                "--host",
                SERVER_HOST,
                "--port",
                str(SERVER_PORT),
                "--single-process",
            ],
            cwd=QUEEREE_APP_DIR,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        # Wait for the server to come up
        print(f"[cog] Waiting for Queeree server on {SERVER_URL} ...")
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            try:
                r = httpx.get(SERVER_URL, timeout=2)
                if r.status_code == 200:
                    print("[cog] Queeree server is ready!")
                    return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass

            # Check if the process died
            if self.server_process.poll() is not None:
                raise RuntimeError(
                    f"Queeree server exited prematurely with code "
                    f"{self.server_process.returncode}"
                )

            time.sleep(1)

        raise RuntimeError(
            f"Queeree server did not become healthy within {STARTUP_TIMEOUT}s"
        )

    def predict(
        self,
        content: str = Input(
            description="Text content to analyze for DFK violations.",
            default="",
        ),
        image: Path = Input(
            description="Optional image file to analyze alongside the text.",
            default=None,
        ),
        media_url: str = Input(
            description=(
                "Optional URL to an image/media file. "
                "Used when no local image is provided."
            ),
            default="",
        ),
        openrouter_api_key: Secret = Input(
            description=(
                "OpenRouter API key for LLM calls. "
                "If not provided, falls back to the server's configured key."
            ),
            default=None,
        ),
        classifier_model_name: str = Input(
            description="Override the LLM model used for classification.",
            default="",
        ),
        fact_checker_model_name: str = Input(
            description="Override the LLM model used for fact-checking.",
            default="",
        ),
        classifier_n_samples: int = Input(
            description="Number of parallel classification votes.",
            default=5,
        ),
        fact_checker_n_samples: int = Input(
            description="Number of parallel fact-checker verification paths.",
            default=3,
        ),
        fact_checker_max_loops: int = Input(
            description="Max iterative search loops for fact-checking.",
            default=3,
        ),
        reasoning_effort: str = Input(
            description="Reasoning effort level for the LLM.",
            choices=["low", "medium", "high"],
            default="low",
        ),
    ) -> str:
        """Run the Queeree moderation pipeline and return the JSON result."""

        # Build config overrides
        config = {
            "classifier_n_samples": classifier_n_samples,
            "fact_checker_n_samples": fact_checker_n_samples,
            "fact_checker_max_loops": fact_checker_max_loops,
            "reasoning_effort": reasoning_effort,
        }
        if classifier_model_name:
            config["classifier_model_name"] = classifier_model_name
        if fact_checker_model_name:
            config["fact_checker_model_name"] = fact_checker_model_name

        # Build auth headers for BYOK — the Queeree server's _resolve_client
        # picks up Bearer tokens from the Authorization header.
        headers = {}
        if openrouter_api_key is not None:
            if isinstance(openrouter_api_key, SecretStr):
                secret_value = openrouter_api_key.get_secret_value()
            else:
                secret_value = str(openrouter_api_key)
            headers["Authorization"] = f"Bearer {secret_value}"

        # Decide whether to send multipart (has image file) or JSON
        if image is not None:
            # Multipart form
            files = {"image": open(str(image), "rb")}
            data = {
                "content": content or "",
                "config": json.dumps(config),
            }
            if media_url:
                data["media_url"] = media_url

            with httpx.Client(timeout=600) as client:
                response = client.post(
                    ANALYZE_URL, data=data, files=files, headers=headers
                )
        else:
            # JSON body
            body = {"content": content, "config": config}
            if media_url:
                body["media_url"] = media_url

            with httpx.Client(timeout=600) as client:
                response = client.post(ANALYZE_URL, json=body, headers=headers)

        # Parse the SSE stream for the final result
        final_result = None
        for line in response.text.splitlines():
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                    if event.get("type") == "result":
                        final_result = event.get("data")
                    elif event.get("type") == "error":
                        return json.dumps(
                            {"error": event.get("data")},
                            ensure_ascii=False,
                            indent=2,
                        )
                except json.JSONDecodeError:
                    continue

        if final_result is None:
            return json.dumps(
                {"error": "No result received from the pipeline."},
                ensure_ascii=False,
                indent=2,
            )

        return json.dumps(final_result, ensure_ascii=False, indent=2)

    def __del__(self):
        """Shut down the server subprocess on cleanup."""
        proc = getattr(self, "server_process", None)
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
