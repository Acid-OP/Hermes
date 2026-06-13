"""Entry point: python -m hermes "your task" """

import sys

from .agent import run

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "What is (128*47)+99? Then read ./note.txt."
    print("USER:", task)
    print("\nANSWER:", run(task))
