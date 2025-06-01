#!/usr/bin/env python3

import sys
import os
import torch
import traceback

# Add current directory to path
sys.path.append('.')

try:
    print("Testing explanation system...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    from explanation import Explanation
    print("✓ Successfully imported Explanation class")
    
    explanation = Explanation('NTU')
    print("✓ Successfully created Explanation instance")
    
    print("Test completed successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
