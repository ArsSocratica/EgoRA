"""HuggingFace Spaces entry point for the EgoRA dashboard."""

import os

from egora.demo import build_demo, launch_demo

demo = build_demo(
    trust_remote_code=False,
    allow_training=os.environ.get("EGORA_ALLOW_TRAINING", "1") == "1",
)

if __name__ == "__main__":
    launch_demo(demo)
