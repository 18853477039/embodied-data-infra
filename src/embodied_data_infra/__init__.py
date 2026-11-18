"""End-to-end data infrastructure for LeRobot manipulation datasets.

The package is organised by pipeline stage:

- ``collection``: launch LeRobot recording sessions from declarative configs.
- ``cleaning``: two-stage cleaning/conversion pipeline for LeRobot v3 datasets.
- ``training``: launch LeRobot training runs against curated datasets.
"""

__version__ = "0.2.0"
PROJECT_NAME = "embodied-data-infra"
