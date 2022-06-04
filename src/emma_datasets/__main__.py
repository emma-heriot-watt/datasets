import rich_click as click
from rich_click import typer

from emma_datasets.commands.download_datasets import app as download_datasets_cli
from emma_datasets.commands.extract_annotations import app as extract_annotations_cli
from emma_datasets.commands.organise_all_datasets import organise_datasets


click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = False

app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(download_datasets_cli, name="download")
app.add_typer(extract_annotations_cli, name="extract")
app.command(name="organise")(organise_datasets)

if __name__ == "__main__":
    app()
