from rich_click import typer

from emma_datasets.commands.download_datasets import app as download_datasets_cli
from emma_datasets.commands.organise_all_datasets import organise_datasets


app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(download_datasets_cli, name="download")
app.command(name="organise")(organise_datasets)

if __name__ == "__main__":
    app()
