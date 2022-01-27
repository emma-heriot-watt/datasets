from enum import Enum


class TaskPrefix(Enum):
    """Prefixes used to specify the task the model has to solve."""

    object_attribute = "describe object attribute"
    object_relation = "describe object relation"
    execute_actions = "execute actions"
    describe_scene = "describe scene"
    answer_question = "answer question"
    describe_region = "describe region"

    def __str__(self) -> str:
        """Returns the task prefix in string form."""
        return self.value
