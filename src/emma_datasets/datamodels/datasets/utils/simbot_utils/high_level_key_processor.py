import random
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from emma_datasets.constants.simbot.high_level_templates import OBJECT_META_TEMPLATE
from emma_datasets.constants.simbot.simbot import get_object_synonym


class DecodedKey(BaseModel):
    """Decoded key base model."""

    action: str

    target_object: Optional[str] = Field(default=None, alias="target-object")
    target_object_color: Optional[str] = Field(default=None, alias="target-object-color")

    from_receptacle: Optional[str] = Field(default=None, alias="from-receptacle")
    from_receptacle_color: Optional[str] = Field(default=None, alias="from-receptacle-color")
    from_receptacle_is_container: Optional[bool] = Field(
        default=None, alias="from-receptacle-is-container"
    )

    to_receptacle: Optional[str] = Field(default=None, alias="to-receptacle")
    to_receptacle_color: Optional[str] = Field(default=None, alias="to-receptacle-color")
    to_receptacle_is_container: Optional[bool] = Field(
        default=None, alias="to-receptacle-is-container"
    )

    convert_object: Optional[str] = Field(default=None, alias="converted-object")
    convert_object_color: Optional[str] = Field(default=None, alias="converted-object-color")


class HighLevelKey(BaseModel):
    """High level key base model."""

    raw_high_level_key: str
    descriptions: list[str]
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
        self.entity_meta = [
            "target_object",
            "from_container",
            "from_receptacle",
            "to_container",
            "to_receptacle",
            "converted_object_color",
            "converted_object",
        ]

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

        self._articles = ["a", "the", ""]

    def __call__(self, highlevel_key: str) -> HighLevelKey:
        """Generate description, paraphrases and plans from a given high-levle key."""
        decoded_key = self.decode_key(highlevel_key)

        self.template_metadata = self._get_template_metadata(decoded_key)

        highlevel_data: dict[str, Any] = {}

        selected_description_templates = self._get_matching_templates(
            self.template_metadata["description_templates"]
        )
        description = self._generate_descriptions(selected_description_templates)
        highlevel_data["descriptions"] = description

        selected_instruction_templates = self._get_matching_templates(
            self.template_metadata["instruction_templates"]
        )
        paraphrases = self._generate_example_utterances(
            selected_instruction_templates,
        )
        highlevel_data["paraphrases"] = paraphrases
        highlevel_data["raw_high_level_key"] = highlevel_key
        return HighLevelKey.parse_obj(highlevel_data)

    def decode_key(self, highlevel_key: str) -> DecodedKey:
        """Decodes the high level key.

        A high level key looks like this:
        action--pickup_target-object--carrot_target-object-color--black_from-receptacle--fridge_from-receptacle-is-container-HtLkD
        """
        # Remove the random string at the end of the key
        highlevel_key = "-".join(highlevel_key.split("-")[:-1])

        key_values: dict[str, Any] = {}
        kv_splits = highlevel_key.split("_")
        for kv in kv_splits:
            kv_pair = kv.split("--")
            if len(kv_pair) == 1:
                key_values[kv_pair[0]] = True
            else:
                key_values[kv_pair[0]] = kv_pair[1]
        return DecodedKey.parse_obj(key_values)

    def _get_template_metadata(self, decoded_key: DecodedKey) -> dict[str, Any]:
        """Get the template metadata for the decoded key."""
        template_metadata = OBJECT_META_TEMPLATE[decoded_key.action]
        template_metadata["prefix"] = self._prefixes
        template_metadata["article"] = self._articles

        template_metadata["target_object"] = self._handle_target_object(decoded_key)

        from_container, from_receptacle = self._handle_from_container_and_receptacle(decoded_key)
        template_metadata["from_container"] = from_container
        template_metadata["from_receptacle"] = from_receptacle

        to_container, to_receptacle = self._handle_to_container_and_receptacle(decoded_key)
        template_metadata["to_container"] = to_container
        template_metadata["to_receptacle"] = to_receptacle

        template_metadata = self._handle_converted_object_color(decoded_key, template_metadata)

        return template_metadata

    def _get_matching_templates(self, templates: list[str]) -> list[str]:
        selected_template = []
        for template in templates:
            template_entities = re.findall(r"\{(.*?)\}", template)
            template_entities = [e for e in template_entities if e in self.entity_meta]
            meta_entities = [
                k
                for k, v in self.template_metadata.items()
                if k in self.entity_meta and len(v) > 0  # noqa: WPS507
            ]
            if len(template_entities) == len(meta_entities) and all(  # noqa: WPS337
                te in template_entities for te in meta_entities
            ):
                selected_template.append(template)
        return selected_template

    def _generate_descriptions(self, templates: list[str]) -> list[str]:  # noqa: WPS231
        descriptions = []
        matching_templates = self._get_matching_templates(templates)
        for template in matching_templates:
            value_selections: dict[str, Any] = {}
            for k, v in self.template_metadata.items():
                if not v:
                    value_selections[k] = None
                    continue

                ling_variants = list(v)
                if k.lower() in {  # noqa: WPS337
                    "prefix",
                    "article",
                    "verb",
                    "instruction_templates",
                    "description_templates",
                }:
                    continue
                else:
                    value_selections[k] = ling_variants[0]

            description = template.format(**value_selections)
            description = f"{description}"
            description = description.replace("\xa0", " ")
            description = description.replace("  ", " ")
            descriptions.append(description)

        return descriptions

    def _get_value_selections_per_template(self) -> dict[str, Any]:  # noqa: WPS231
        value_selections: dict[str, Any] = {}
        for k, v in self.template_metadata.items():
            if not v:
                value_selections[k] = None
                continue
            ling_variants = list(v)
            if k.lower() in {"instruction_templates", "description_templates"}:
                continue
            elif k.lower() == "prefix":
                if random.random() < self.prefix_inclusion_probability:
                    value_selections[k] = random.choice(ling_variants).lower()
                else:
                    value_selections[k] = ""
            else:
                value_selections[k] = random.choice(ling_variants).lower()
        return value_selections

    def _generate_example_utterances(self, templates: list[str]) -> list[str]:
        paraphrased_utterances = set()
        for template in templates:
            for _ in range(self.paraphrases_per_template):
                value_selections = self._get_value_selections_per_template()
                annotated_data = template.format(**value_selections)
                annotated_data = f"{annotated_data}"
                annotated_data = annotated_data.replace("\xa0", " ")
                annotated_data = annotated_data.replace("  ", " ")
                paraphrased_utterances.add(annotated_data.strip())
        return list(paraphrased_utterances)

    def _handle_target_object(self, decoded_key: DecodedKey) -> list[str]:
        target_synonyms: list[str] = []

        target = decoded_key.target_object
        if target is None:
            return target_synonyms

        target_synonyms = get_object_synonym(target)

        target_color = decoded_key.target_object_color
        if target_color:
            target = f"{target_color} {target}"
            if target_synonyms:
                target_synonyms = [f"{target_color} {synonym}" for synonym in target_synonyms]

        return [synonym.lower() for synonym in target_synonyms]

    def _handle_from_container_and_receptacle(
        self, decoded_key: DecodedKey
    ) -> tuple[list[str], list[str]]:
        from_container: list[str] = []
        from_receptacle: list[str] = []

        receptacle_str = decoded_key.from_receptacle
        if not receptacle_str:
            return from_container, from_receptacle

        if decoded_key.from_receptacle_color:
            receptacle_str = f"{decoded_key.from_receptacle_color} {receptacle_str}"

        if decoded_key.from_receptacle_is_container:
            from_container = [receptacle_str]
        else:
            from_receptacle = [receptacle_str]

        return from_container, from_receptacle

    def _handle_to_container_and_receptacle(
        self, decoded_key: DecodedKey
    ) -> tuple[list[str], list[str]]:
        to_container = []
        to_receptacle = []

        receptacle_container = decoded_key.to_receptacle

        if decoded_key.to_receptacle_color:
            receptacle_container = f"{decoded_key.to_receptacle_color} {decoded_key.to_receptacle}"

        if receptacle_container:
            if decoded_key.to_receptacle_is_container:
                to_container = [receptacle_container]
            else:
                to_receptacle = [receptacle_container]
        return to_container, to_receptacle

    def _handle_converted_object_color(
        self, decoded_key: DecodedKey, template_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        converted_object = decoded_key.convert_object
        converted_object_color = decoded_key.convert_object_color

        if decoded_key.convert_object_color:
            if decoded_key.convert_object:
                converted_object = f"{converted_object} {decoded_key.convert_object_color}"
            else:
                converted_object = decoded_key.convert_object_color

        if decoded_key.action in {"timemachine", "colorchanger"}:
            template_metadata["converted_object_color"] = (
                converted_object_color if converted_object_color else []
            )
            template_metadata["converted_object"] = converted_object if converted_object else []
        return template_metadata
