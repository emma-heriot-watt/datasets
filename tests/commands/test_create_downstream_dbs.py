from pathlib import Path

from typer.testing import CliRunner

from emma_datasets.commands.create_downstream_dbs import app as create_downstream_dbs_app
from emma_datasets.datamodels.datasets import TeachEdhInstance
from emma_datasets.db import DatasetDb


runner = CliRunner()


def test_can_create_downstream_dbs_for_teach_edh(
    teach_edh_instance_path: Path, tmp_path: Path
) -> None:
    app_output = runner.invoke(
        create_downstream_dbs_app,
        [
            "teach-edh",
            "--teach-edh-instances-base-dir",
            teach_edh_instance_path.as_posix(),
            "--output-dir",
            tmp_path.as_posix(),
            "--num-workers",
            "1",
        ],
    )
    # Verify the API returned without erroring
    assert app_output.exit_code == 0

    # Ensure there are 3 db files that have been created
    assert len(list(tmp_path.iterdir())) == 3

    for db_path in tmp_path.iterdir():
        # Verify the Db is named correctly
        assert db_path.is_file()
        assert db_path.suffix.endswith("db")

        read_db = DatasetDb(db_path, readonly=True)

        # Ensure there are instances within the Db
        assert len(read_db)

        # Verify each instance is correct
        for _, _, instance_str in read_db:
            new_instance = TeachEdhInstance.parse_raw(instance_str)
            assert isinstance(new_instance, TeachEdhInstance)