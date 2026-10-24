#!/usr/bin/env python3
"""
IMU Gesture Controller - Main Entry Point
University of Basrah · College of Engineering · Computer Engineering Department

Academic Project Title:
"Machine Learning-Based Hand Gesture Recognition Using a Wearable IMU Sensor for PowerPoint Control"
Supervisor: Dr. Khalid Ali Abbas
"""

import sys
import os

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.gui_app import main

if __name__ == "__main__":
    main()
