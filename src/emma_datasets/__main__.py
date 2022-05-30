from rich_click import typer

from emma_datasets.commands.download_datasets import download_datasets
from emma_datasets.commands.organise_all_datasets import organise_datasets


app = typer.Typer(add_completion=False)
app.command(name="download")(download_datasets)
app.command(name="organise")(organise_datasets)

if __name__ == "__main__":
    app()
