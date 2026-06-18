"""Verify whether the system is running against a real LLM (online) or offline.

Loads the .env (via LLMClient), reports connection status, and — when online —
performs one tiny completion call to confirm the API key, model, and base_url all
work end-to-end. Run this right after filling OPENAI_API_KEY in .env.

Usage:
    python scripts/check_llm_online.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent.llm_client import LLMClient, LLMError  # noqa: E402
from src.common.config import get_config  # noqa: E402
from src.common.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    setup_logging()
    config = get_config()
    client = LLMClient(config)
    env_file = _PROJECT_ROOT / ".env"

    print("=== LLM connection check ===")
    print(f"config model : {config.agent.llm.model}")
    base = os_base()  # noqa
    print(f"base_url     : {base or '(default OpenAI)'}")
    env_hint = "present" if env_file.exists() else "MISSING - create it at the project root"
    print(f".env file    : {env_hint}")
    print(f"status       : {'OFFLINE' if client.offline else 'ONLINE'}")

    if client.offline:
        print("\n-> 当前离线。请在 .env 填入 OPENAI_API_KEY=sk-... 后重新运行本脚本。")
        return 0

    # Online: do one tiny real call to confirm end-to-end.
    print("\nPerforming one test completion...")
    started = time.time()
    try:
        reply = client.invoke(
            "You are a connection-test assistant. Reply with exactly: ONLINE_OK",
            "ping",
        )
    except LLMError as exc:
        print(f"\n[FAIL] 在线但调用失败: {exc}")
        print("  常见原因:key 错误/过期、base_url 与 key 不匹配、模型名在 config 里写错、网络/超时。")
        return 1

    elapsed = time.time() - started
    print(f"[OK] 调用成功 ({elapsed:.2f}s)")
    print(f"  响应: {reply.strip()[:120]}")
    print("\n系统已在线运行。可直接运行:")
    print("  python scripts/demo_agent.py")
    print("  chainlit run app.py")
    print("  python tests/evaluation/run_eval_real.py")
    return 0


def os_base() -> str:
    import os

    return os.getenv("OPENAI_BASE_URL", "")


if __name__ == "__main__":
    sys.exit(main())
