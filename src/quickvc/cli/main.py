import logging

import typer

from .commands.infer import infer
from .commands.split import split
from .commands.encode import encode_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Create the main Typer app
typer_app = typer.Typer(help="Main CLI entry point")

# Register subcommands
typer_app.command(name="split", help="Split audio into chunks")(split)
typer_app.command(name="infer", help="Run inference using QuickVC model")(infer)
typer_app.command(name="encode", help="Extract and save hubert units")(encode_dataset)

if __name__ == "__main__":
    typer_app()
