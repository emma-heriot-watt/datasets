import rich_click as click
import typer

from emma_datasets.commands.create_downstream_dbs import app as create_downstream_dbs_cli
from emma_datasets.commands.download_datasets import download_datasets
from emma_datasets.commands.extract_annotations import app as extract_annotations_cli
from emma_datasets.commands.organise_all_datasets import organise_datasets


click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = False

app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(extract_annotations_cli, name="extract")
app.add_typer(create_downstream_dbs_cli, name="downstream")
app.command(name="organise")(organise_datasets)
app.command(name="download")(download_datasets)

if __name__ == "__main__":
    app()
