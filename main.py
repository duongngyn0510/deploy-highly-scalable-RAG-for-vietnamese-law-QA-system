"""FastAPI app creation, logger configuration and main API routes."""

from src.di import global_injector
from src.launcher import create_app

app = create_app(global_injector)
