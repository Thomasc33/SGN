"""
Simple runner script for generating all visualizations
"""
import os
import sys

# Add the scripts directory to the path
scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
sys.path.insert(0, scripts_dir)

# Import and run the main generation script
from generate_all_figures import main

if __name__ == "__main__":
    print("Privacy-Preserving Skeleton Analysis Visualization Generator")
    print("=" * 60)
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nVisualization generation interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check that all required dependencies are installed and data files are available.")
    
    print("\nVisualization generation finished.")
