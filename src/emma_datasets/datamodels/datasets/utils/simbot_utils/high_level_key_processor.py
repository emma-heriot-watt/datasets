import random
import re
from typing import Any, Optional

from pydantic import BaseModel, Field, validator

from emma_datasets.constants.simbot.high_level_templates import OBJECT_META_TEMPLATE
from emma_datasets.constants.simbot.simbot import get_arena_definitions, get_object_synonym


def get_previous_key(deconstructed_highlevel_key: str) -> str:
    """Get the previous decoded key after deconstructing a high level key.

    Used to populate the DecodedKey basemodel.
    """
    if "_" in deconstructed_highlevel_key:
        return deconstructed_highlevel_key.split("_")[-1:][0]
    return deconstructed_highlevel_key


def parse_deconstructed_highlevel_key_parts(  # noqa: WPS231
    decoded_key_values: dict[str, Any], part: str, parts: list[str], part_idx: int
) -> dict[str, Any]:
    """Parse a part of the deconstructed highlevel key.

    Used to populate the DecodedKey basemodel.
    """
    # If the part does not contain any dashes then this is a key on its own
    # Initialize it to true
    if "_" not in part and part_idx != len(parts) - 1:
        decoded_key_values[part] = True

    # If the part does contain dashes then it contains a value for the previous key and the name of the current key
    elif 1 <= part_idx < len(parts) - 1:
        split_part_by_value = part.split("_")
        decoded_current_key = split_part_by_value[-1]
        decoded_previous_key_value = "_".join(split_part_by_value[:-1])

        previous_key = get_previous_key(parts[part_idx - 1])

        decoded_key_values[previous_key] = decoded_previous_key_value
        decoded_key_values[decoded_current_key] = True

    elif part_idx == len(parts) - 1:
        if "_" not in part:
            if "-" in part:
                decoded_key_values[part] = True
            else:
                previous_key = get_previous_key(parts[part_idx - 1])
                decoded_key_values[previous_key] = part
        else:
            previous_key = get_previous_key(parts[part_idx - 1])
            # If the last part in the highlevel key has also a decode key include it
            if "-" in part:
                split_part_by_value = part.split("_")
                decoded_current_key = split_part_by_value[-1]
                decoded_key_values[previous_key] = "_".join(split_part_by_value[:-1])
                decoded_key_values[decoded_current_key] = True
            # Else the last part should be the value of the previous key
            else:
                decoded_key_values[previous_key] = part
    return decoded_key_values


class DecodedKey(BaseModel):
    """Decoded key base model."""

    raw_high_level_key: str

    action: str

    interaction_object: Optional[str] = Field(default=None, alias="interaction-object")
    target_object: Optional[str] = Field(default=None, alias="target-object")
    target_object_color: Optional[str] = Field(default=None, alias="target-object-color")
    target_object_is_ambiguous: Optional[bool] = Field(
        default=None, alias="target-object-is-ambiguous"
    )

    stacked_object: Optional[str] = Field(default=None, alias="stacked-object")
    stacked_object_color: Optional[str] = Field(default=None, alias="stacked-object-color")

    from_receptacle: Optional[str] = Field(default=None, alias="from-receptacle")
    from_receptacle_color: Optional[str] = Field(default=None, alias="from-receptacle-color")
    from_receptacle_is_container: Optional[bool] = Field(
        default=None, alias="from-receptacle-is-container"
    )
    # This is populated if from_receptacle is provided and from_receptacle_is_container == True
    from_container: Optional[str] = Field(default=None, alias="from-container")

    to_receptacle: Optional[str] = Field(default=None, alias="to-receptacle")
    to_receptacle_color: Optional[str] = Field(default=None, alias="to-receptacle-color")
    to_receptacle_is_container: Optional[bool] = Field(
        default=None, alias="to-receptacle-is-container"
    )

    # This is populated if to_receptacle is provided and to_receptacle_is_container == True
    to_container: Optional[str] = Field(default=None, alias="to-container")

    converted_object: Optional[str] = Field(default=None, alias="converted-object")
    converted_object_color: Optional[str] = Field(default=None, alias="converted-object-color")

    @validator("interaction_object", "target_object", "converted_object", "stacked_object")
    @classmethod
    def validate_objects_in_key(cls, field_value: str) -> str:
        """Verify that the object fields are defined in the arena."""
        arena_definitions = get_arena_definitions()
        assets_to_labels = arena_definitions["asset_to_label"]
        if field_value not in assets_to_labels:
            raise AssertionError(
                f"Expecting objects to be within the arena definitions, but found {field_value}"
            )
        return field_value

    @classmethod
    def get_field_names(cls, alias: bool = False) -> list[str]:
        """Get the field names in a list of strings."""
        return list(
            cls.schema(by_alias=alias).get("properties").keys()  # type:ignore[union-attr]
        )

    @classmethod
    def from_raw_string(cls, highlevel_key: str) -> "DecodedKey":
        """Parse a raw highlevel key."""
        decoded_key_values: dict[str, Any] = {"raw_high_level_key": highlevel_key}

        highlevel_key = "-".join(highlevel_key.split("-")[:-1])

        if "target-object-is-ambiguous" in highlevel_key:
            decoded_key_values["target-object-is-ambiguous"] = True
            highlevel_key = highlevel_key.replace("target-object-is-ambiguous", "")
            highlevel_key = highlevel_key.replace("__", "_")

        parts = highlevel_key.split("--")
        for part_idx, part in enumerate(parts):
            decoded_key_values = parse_deconstructed_highlevel_key_parts(
                decoded_key_values=decoded_key_values, part=part, parts=parts, part_idx=part_idx
            )

        can_replace_from_reptacle_from_container = (
            "from-receptacle" in decoded_key_values
            and decoded_key_values["from-receptacle"] is not None
            and "from-receptacle-is-container" in decoded_key_values
            and decoded_key_values["from-receptacle-is-container"]
        )
        if can_replace_from_reptacle_from_container:
            decoded_key_values["from-container"] = decoded_key_values["from-receptacle"]
            decoded_key_values["from-receptacle"] = None

        can_replace_to_reptacle_to_container = (
            "to-receptacle" in decoded_key_values
            and decoded_key_values["to-receptacle"] is not None
            and "to-receptacle-is-container" in decoded_key_values
            and decoded_key_values["to-receptacle-is-container"]
        )
        if can_replace_to_reptacle_to_container:
            decoded_key_values["to-container"] = decoded_key_values["to-receptacle"]
            decoded_key_values["to-receptacle"] = None
        return cls(**decoded_key_values)

    def field_has_object_id(self, field: str) -> bool:
        """Check whether a field contains an object id that should be mapped to a synonym."""
        return field in {
            "interaction_object",
            "target_object",
            "from_receptacle",
            "to_receptacle",
            "converted_object",
            "to_container",
            "from_container",
            "stacked_object",
        }

    def get_interacted_objects(self) -> list[str]:
        """Retun the list of objects the agent interacted with during the session."""
        objects_in_key = [
            self.interaction_object,
            self.target_object,
            self.from_receptacle,
            self.to_receptacle,
            self.from_container,
            self.to_container,
            self.converted_object,
        ]
        interacted_objects = []
        for object_in_key in objects_in_key:
            if object_in_key is not None:
                interacted_objects.append(object_in_key)
        return interacted_objects


class HighLevelKey(BaseModel):
    """High level key base model."""

    decoded_key: DecodedKey
    high_level_description: str
    paraphrases: list[str]


class HighLevelKeyProcessor:
    """Generate descriptions and paraphrases for a given high level key."""

    def __init__(
        self,
        prefix_inclusion_probability: float = 0.2,
        paraphrases_per_template: int = 1,
    ):
        self.prefix_inclusion_probability = prefix_inclusion_probability
        self.paraphrases_per_template = paraphrases_per_template
        self.decoded_key_fields = DecodedKey.get_field_names(alias=False)

        self._prefixes = [
            "i would like to",
            "i need to",
            "i need you to",
            "i am telling you to",
            "you should",
            "we need to",
            "let's",
            "can you",
            "could you",
            "okay",
            "okay now",
            "now",
            "please",
        ]

    def __call__(self, highlevel_key: str) -> HighLevelKey:
        """Generate description, paraphrases and plans from a given high-levle key."""
        decoded_key = DecodedKey.from_raw_string(highlevel_key=highlevel_key)

        template_metadata = OBJECT_META_TEMPLATE[decoded_key.action]

        if decoded_key.action == "interact":
            secondary_key = decoded_key.interaction_object
            if secondary_key == "YesterdayMachine_01" and decoded_key.target_object == "Carrot_01":
                secondary_key = "YesterdayMachine_01_from_Carrot"

            template_metadata = template_metadata[secondary_key]

        for decoded_key_field in self.decoded_key_fields:
            decoded_key_value = getattr(decoded_key, decoded_key_field)
            should_get_object_synonym = (
                decoded_key.field_has_object_id(decoded_key_field)
                and decoded_key_value is not None
            )
            if should_get_object_synonym:
                template_metadata[decoded_key_field] = get_object_synonym(decoded_key_value)
            else:
                template_metadata[decoded_key_field] = decoded_key_value

        formatted_paraphrases = self.get_paraphrases(template_metadata, decoded_key=decoded_key)
        return HighLevelKey(
            decoded_key=decoded_key,
            paraphrases=formatted_paraphrases,
            high_level_description="",
        )

    def get_paraphrases(  # noqa: WPS231
        self, template_metadata: dict[str, Any], decoded_key: DecodedKey
    ) -> list[str]:
        """Get the instruction paraphrases for a highlevel key."""
        paraphrases = template_metadata["paraphrases"]

        is_ambiguous = decoded_key.target_object_is_ambiguous

        formatted_paraphrases = []
        for paraphrase in paraphrases:
            formatting_fields = re.findall(r"\{(.*?)\}", paraphrase)
            formatting_dict = {}
            for field in formatting_fields:
                formatting_value = template_metadata.get(field, None)

                if formatting_value is not None and isinstance(formatting_value, list):
                    formatting_dict[field] = random.choice(formatting_value)
                else:
                    formatting_dict[field] = formatting_value

            # If any field that needs formatting in the paraphrased template is None, skip the paraphrasing template
            if any([formatting_value is None for formatting_value in formatting_dict.values()]):
                continue

            # Disambiguate only by color
            if is_ambiguous and formatting_dict.get("target_object_color", None) is None:
                continue

            formatted_paraphrase = paraphrase.format(**formatting_dict).lower()
            formatted_paraphrases.append(self._append_prefix(formatted_paraphrase))
        return formatted_paraphrases

    def _append_prefix(self, input_instruction: str) -> str:
        if random.random() < self.prefix_inclusion_probability:
            random_prefix = random.choice(self._prefixes)
            input_instruction = f"{random_prefix} {input_instruction}"
        return input_instruction
