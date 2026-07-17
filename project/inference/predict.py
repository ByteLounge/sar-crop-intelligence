import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_complete_pipeline import run_all_phases

if __name__ == '__main__':
    print("Executing updated inference pipeline...")
    run_all_phases()
