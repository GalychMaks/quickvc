import logging

import typer

from .commands.infer import infer_typer
from .commands.split import split_typer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Create the main Typer app
typer_app = typer.Typer(help="Main CLI entry point")

# Register subcommands
typer_app.add_typer(split_typer, name="split", help="Split audio into chunks")
typer_app.add_typer(infer_typer, name="infer", help="Run inference using QuickVC model")


if __name__ == "__main__":
    typer_app()
