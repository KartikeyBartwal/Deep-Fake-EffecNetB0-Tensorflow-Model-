"""Console script for effnetb0_deep_learning."""
import effnetb0_deep_learning

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for effnetb0_deep_learning."""
    console.print("Replace this message by putting your code into "
               "effnetb0_deep_learning.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
