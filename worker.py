"""
A.I.R.A. Arq Worker Entry Point

Run with: arq worker.WorkerSettings
"""

from src.adapters.jobs.arq_worker import WorkerSettings

__all__ = ["WorkerSettings"]
