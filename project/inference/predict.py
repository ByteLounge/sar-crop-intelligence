import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rebuild_pipeline_v2 import run_rich_feature_pipeline

if __name__ == '__main__':
    print("Executing updated rich-feature inference pipeline...")
    run_rich_feature_pipeline()
