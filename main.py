# main.py
import sys
import run_attacks

if __name__ == "__main__":
    # If no argument is provided, run all model types.
    # Otherwise, you can pass a specific model type (e.g., "sagn") or "all" to run all.
    run_attacks.run_attacks_on_model()
