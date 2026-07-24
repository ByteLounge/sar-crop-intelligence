import os
import sys

workspace_dir = r"D:\PC\resources"
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from project.features.extract_sentinel import run_extraction
from project.run_complete_pipeline import run_all_phases

def main():
    print("========================================================================")
    print("STEP A: EXTRACTING SENTINEL-1 AND SENTINEL-2 AUXILIARY FEATURES...")
    print("========================================================================")
    run_extraction()
    
    print("\n========================================================================")
    print("STEP B: EXECUTING MAIN PROJECT PIPELINE WITH SENTINEL INTEGRATION...")
    print("========================================================================")
    run_all_phases()
    
if __name__ == "__main__":
    main()
