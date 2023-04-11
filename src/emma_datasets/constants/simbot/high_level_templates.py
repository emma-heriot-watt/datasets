# flake8: noqa WPS226
from types import MappingProxyType

from emma_datasets.constants.simbot.simbot import get_object_synonym


OBJECT_META_TEMPLATE = MappingProxyType(
    {
        "pickup": {
            "basic_verb": ["pick up"],
            "verb": [
                "pick up",
                "get",
                "grab",
                "take",
                "pick",
                "retrieve",
                "collect",
                "take off",
            ],
            "description_templates": [
                # pick up the blue can
                "{basic_verb} the {target_object}",
                # pick up the blue can on the green table
                "{basic_verb} the {target_object} on the {from_receptacle}",
                # pick up the blue can inside the red fridge
                "{basic_verb} the {target_object} from inside the {from_container}",
                # pick up the blue candy bar from the white plate on the green table
                "{basic_verb} the {target_object} from the {from_receptacle} on the {to_receptacle}",
                # pick up the blue candy bar from white plate inside the black microwave
                "{basic_verb} the {target_object} from the {from_receptacle} inside the {to_container}",
                # pick up the blue can inside the black microwave on the green table
                "{basic_verb} the {target_object} inside the {from_container} on the {to_receptacle}",
            ],
            "instruction_templates": [
                "{prefix} {verb} {article} {target_object}",
                "{prefix} {verb} {article} {target_object} from {article} {from_receptacle}",
                "{prefix} {verb} {article} {target_object} on {article} {from_receptacle}",
                "{prefix} {verb} {article} {target_object} on top of {article} {from_receptacle}",
                "{prefix} {verb} {article} {target_object} from {article} {from_container}",
                "{prefix} {verb} {article} {target_object} from inside {article} {from_container}",
                "{prefix} {verb} {article} {target_object} from {from_receptacle} on {article} {to_receptacle}",
                "{prefix} {verb} {article} {target_object} on {from_receptacle} on {article} {to_receptacle}",
                "{prefix} {verb} {article} {target_object} on top of {article} {from_receptacle} on {article} {to_receptacle}",
                "{prefix} {verb} {article} {target_object} inside {article} {from_container} on {article} {to_receptacle}",
                "{prefix} {verb} {article} {target_object} from {article} {from_container} on {article} {to_receptacle}",
            ],
        },
        "place": {
            "verb": ["place", "put", "leave", "set"],
            "basic_verb": ["place"],
            "description_templates": [
                # place the blue can
                "{basic_verb} the {target_object}",
                # place the blue can on the green table
                "{basic_verb} the {target_object} on the {to_receptacle}",
                # place the blue can inside the red fridge
                "{basic_verb} the {target_object} inside the {to_container}",
                # place the blue candy bar on the green table on the white plate
                "move the {target_object} on {from_receptacle} to {to_receptacle}",
                # place the red cake inside the red freezer on the blue shelf
                "move the {target_object} inside the {from_container} to {to_receptacle}",
                # place the red cake on the blue shelf inside the red freezer
                "move the {target_object} on the {from_receptacle} to the {to_container}",
                # place the sandwich inside the fridge in the microwave
                "move the {target_object} inside the {from_container} to the {to_container}",
            ],
            "instruction_templates": [
                "{prefix} {verb} {article} {target_object}",
                "{prefix} {verb} {article} {target_object} on {article} {to_receptacle}",
                "{prefix} {verb} {article} {target_object} on top of {article} {to_receptacle}",
                "{prefix} {verb} {article} {target_object} in {article} {to_container}",
                "{prefix} {verb} {article} {target_object} inside {article} {to_container}",
                "{prefix} {verb} {article} {target_object} on {article} {from_receptacle} on {article} {to_receptacle}",
                "{prefix} move {article} {target_object} from {article} {from_receptacle} to {article} {to_receptacle}",
                "{prefix} deliver {article} {target_object} from {article} {from_receptacle} to {article} {to_receptacle}",
                "{prefix} {verb} {article} {target_object} inside {article} {from_container} on {article} {to_receptacle}",
                "{prefix} move {article} {target_object} inside {article} {from_container} to {article} {to_receptacle}",
                "{prefix} move {article} {target_object} in {article} {from_container} to {article} {to_receptacle}",
                "{prefix} deliver {article} {target_object} inside {article} {from_container} to {to_receptacle}",
                "{prefix} {verb} {article} {target_object} on {article} {from_receptacle} inside {article} {to_container}",
                "{prefix} move {article} {target_object} from {article} {from_receptacle} to {article} {to_container}",
                "{prefix} move {article} {target_object} on {article} {from_receptacle} to {article} {to_container}",
                "{prefix} deliver {article} {target_object} from {article} {from_receptacle} to {article} {to_container}",
                "{prefix} {verb} {article} {target_object} in {article} {from_container} in {article} {to_container}",
                "{prefix} move {article} {target_object} in {article} {from_container} to {article} {to_container}",
                "{prefix} move {article} {target_object} from {article} {from_container} to {article} {to_container}",
            ],
        },
        "insert": {
            "verb": ["insert", "put"],
            "basic_verb": ["insert"],
            "description_templates": [
                # insert the floppy disk
                "{basic_verb} the {target_object}",
                # insert the floppy disk into the computer
                "{basic_verb} the {target_object} into the {to_receptacle}",
                # insert the <> into the lever? => may not occur
                "{basic_verb} the {target_object} into the {to_container}",
                # insert the floppy on the green table into the white computer
                "{basic_verb} the {target_object} on {from_receptacle} on top of the {to_receptacle}",
                # insert the floppy inside the drawer into the white computer
                "{basic_verb} the {target_object} inside the {from_container} into the {to_receptacle}",
                # insert the <> on the table into the lever? => may not occur
                "{basic_verb} the {target_object} on the {from_receptacle} into the {to_container}",
                # insert the <> inside the drawer into the lever? => may not occur
                "{basic_verb} the {target_object} inside the {from_container} into the {to_container}",
            ],
            "instruction_templates": [
                # insert the floppy disk
                "{prefix} {verb} {article} {target_object}",
                # insert the floppy disk into the computer
                "{prefix} {verb} {article} {target_object} into {article} {to_receptacle}",
                "{prefix} {verb} {article} {target_object} to {article} {to_receptacle}",
                "{prefix} set {article} {target_object} on the {to_receptacle}",
                # insert the <> into the lever? => may not occur
                "{prefix} {verb} {article} {target_object} into {article} {to_container}",
                "{prefix} {verb} {article} {target_object} inside {article} {to_container}",
                "{prefix} set {article} {target_object} on the {to_container}",
                # insert the floppy on the green table into the white computer
                "{prefix} {verb} {article} {target_object} on {article} {from_receptacle} into {article} {to_receptacle}",
                "{prefix} set {article} {target_object} from {article} {from_receptacle} on {article} {to_receptacle}",
                # insert the floppy inside the drawer into the white computer
                "{prefix} {verb} {article} {target_object} inside {article} {from_container} into {article} {to_receptacle}",
                "{prefix} set {article} {target_object} from {article} {from_receptacle} on {article} {to_receptacle}",
                # insert the <> on the table into the lever? => may not occur
                "{prefix} {verb} {article} {target_object} on {article} {from_receptacle} into the {to_container}",
                # insert the <> inside the drawer into the lever? => may not occur
                "{prefix} {verb} {article} {target_object} inside {article} {from_container} into {article} {to_container}",
            ],
        },
        "pour": {
            "verb": [
                "put",
                "pour",
            ],
            "basic_verb": ["pour"],
            "description_templates": [
                # pour the milk
                "{basic_verb} the {target_object}",
                # pour the milk on the table -> may not support
                "{basic_verb} the {target_object} on the {to_receptacle}",
                # pour the milk into the bowl -> pourable target, fillable container
                "{basic_verb} the {target_object} into the {to_container}",
                # pour the milk inside the freezer on the table
                "{basic_verb} the {target_object} inside the {from_container} on the {to_receptacle}",
                # pour the milk inside the freezer into the bowl
                "{basic_verb} the {target_object} inside the {from_container} into the {to_container}",
                # pour the milk on the table on the gravity pad
                "{basic_verb} the {target_object} on the {from_receptacle} on the {to_receptacle}",
            ],
            "instruction_templates": [
                # pour the milk
                "{prefix} {verb} {article} {target_object}",
                # pour the milk on the table -> may not support
                "{prefix} {verb} {article} {target_object} on {article} {to_receptacle}",
                "{prefix} {verb} {article} {target_object} over {article} {to_receptacle}",
                # pour the milk into the bowl -> pourable target, fillable container
                "{prefix} {verb} {article} {target_object} into {article} {to_container}",
                # pour the milk inside the freezer on the table
                "{prefix} {verb} {article} {target_object} inside {article} {from_container} on {article} {to_receptacle}",
                "{prefix} {verb} {article} {target_object} in {article} {from_container} over {article} {to_receptacle}",
                # pour the milk on the table on the gravity pad
                "{prefix} {verb} {article} {target_object} on {article} {from_receptacle} on {article} {to_receptacle}",
                "{prefix} {verb} {article} {target_object} on {article} {from_receptacle} over {article} {to_receptacle}",
                # pour the milk inside the freezer into the bowl
                "{prefix} {verb} {article} {target_object} in {article} {from_container} into {article} {to_container}",
                "{prefix} {verb} {article} {target_object} in {article} {from_container} into {article} {to_container}",
                "{prefix} {verb} {article}  {target_object} on {article} {from_receptacle} into {article} {to_container}",
            ],
        },
        "fill": {
            "verb": ["fill", "fill up"],
            "basic_verb": ["fill"],
            "description_templates": [
                # fill the bowl
                "{basic_verb} the {target_object}",
                # fill the sink with water -> this is a toggle action at low-level
                "{basic_verb} the {to_container} with water",
                # use the sink to fill the mug
                "use the {to_container} to {basic_verb} the {target_object}",
                # fill the mug on the table
                "{basic_verb} the {target_object} on the {from_receptacle}",
                # use the sink to fill the mug on the table
                "use the {to_container} to {basic_verb} the {target_object} on the {from_receptacle}",
                # use the sink to fill the mug in the fridge
                "use the {to_container} to {basic_verb} the {target_object} inside the {from_container}",
            ],
            "instruction_templates": [
                # fill the bowl
                "{prefix} {verb} {article} {target_object}",
                "{prefix} {verb} {article} {target_object} with water",
                # fill the sink -> this is a toggle action at low-level
                "{prefix} {verb} the {to_container}",
                "{prefix} {verb} {to_container} with water",
                # use the sink to fill the mug
                "{prefix} use the {to_container} to {verb} {article} {target_object}",
                "{prefix} {verb} {article} {target_object} using {article} {to_container}",
                "{prefix} put water in {article} {target_object} using {article} {to_container}",
                # fill the mug on the table
                "{prefix} {verb} {article} {target_object} on {article} {from_receptacle}",
                "{prefix} put water in the {target_object} on {article} {from_receptacle}",
                # use the sink to fill the mug on the table
                "{prefix} use the {to_container} to {verb} {article} {target_object} on {article} {from_receptacle}",
                "{prefix} use the {to_container} to put water in the {target_object} on {article} {from_receptacle}",
                "{prefix} {verb} {article} {target_object} on {article} {from_receptacle} using the {to_container}",
                "{prefix} put water in the {target_object} on {article} {from_receptacle} using the {to_container}",
                # use the sink to fill the mug in the fridge
                "{prefix} use the {to_container} to {verb} {article} {target_object} inside {article} {from_container}",
                "{prefix} use the {to_container} to put water in the {target_object} inside {article} {from_container}",
                "{prefix} {verb} {article} {target_object} in {article} {from_container} using the {to_container}",
                "{prefix} put water in the {target_object} in {article} {from_container} using the {to_container}",
            ],
        },
        "clean": {
            "verb": ["clean", "clean up", "rinse", "rinse off", "wash", "wash off"],
            "basic_verb": ["clean"],
            "dirty_synonyms": ["dirty", "unclean", ""],
            "description_templates": [
                # clean the plate
                "{basic_verb} the dirty {target_object}",
                # use the sink to clean the plate
                "use the {to_container} to {basic_verb} the dirty {target_object}",
                # clean the plate on the table
                "{basic_verb} the dirty {target_object} on the {from_receptacle}",
                # use the sink to clean the plate on the table
                "use the {to_container} to {basic_verb} the dirty {target_object} on the {from_receptacle}",
                # use the sink to clean the plate in the fridge
                "use the {to_container} to {basic_verb} the dirty {target_object} inside the {from_container}",
            ],
            "instruction_templates": [
                # clean the dirty plate
                "{prefix} {verb} {article} {target_object}",
                "{prefix} {verb} {article} {dirty_synonyms} {target_object}",
                "{prefix} {verb} {article} {dirty_synonyms} {target_object} with water",
                # use the sink to clean the dirty plate
                "{prefix} use the {to_container} to {verb} {article} {dirty_synonyms} {target_object}",
                "{prefix} {verb} {article} {dirty_synonyms} {target_object} using {article} {to_container}",
                # clean the dirty plate on the table
                "{prefix} {verb} {article} {dirty_synonyms} {target_object} on {article} {from_receptacle}",
                "{prefix} put water in the {dirty_synonyms} {target_object} on {article} {from_receptacle}",
                # use the sink to clean the dirty plate on the table
                "{prefix} use the {to_container} to {verb} {article} {dirty_synonyms} {target_object} on {article} {from_receptacle}",
                "{prefix} {verb} {article} {dirty_synonyms} {target_object} on {article} {from_receptacle} using the {to_container}",
                # use the sink to clean the dirty plate in the fridge
                "{prefix} use the {to_container} to {verb} {article} {dirty_synonyms} {target_object} inside {article} {from_container}",
                "{prefix} {verb} {article} {dirty_synonyms} {target_object} in {article} {from_container} using the {to_container}",
            ],
        },
        "timemachine": {
            "verb": ["operate", "use", "utilize", "make use of"],
            "verbing": ["operating", "using", "utilizing", "making use of"],
            "machine_synonyms": get_object_synonym("time machine"),
            "repair_verb": ["repair", "fix"],
            "convert_verb": ["convert", "reverse", "turn"],
            "basic_verb": ["use"],
            "description_templates": [
                # use the time machine
                "{basic_verb} the time machine",
                # use the time machine to repair the bowl
                "{basic_verb} the time machine to repair the {target_object}",
                # use the time machine to repair the bowl on the table
                "{basic_verb} the time machine to repair the {target_object} on the {from_receptacle}",
                # use the time machine to convert the red carrot to a green banana
                "{basic_verb} the time machine to convert the {target_object} to a {converted_object}",
                # use the time machine to convert the red carrot on the table to a green banana
                "{basic_verb} the time machine to convert the {target_object} on the {from_receptacle} to a {converted_object}",
                # use the time machine to repair the bowl in the fridge
                "{basic_verb} the time machine to repair the {target_object} inside the {from_container}",
                # use the time machine to convert the red carrot in the fridge to a green banana
                "{basic_verb} the time machine to convert the {target_object} in the {from_container} to a {converted_object}",
            ],
            "instruction_templates": [
                # use the time machine
                "{prefix} {verb} {article} time machine",
                # use the time machine to repair the bowl
                "{prefix} {verb} {article} {machine_synonyms} to {repair_verb} the {target_object}",
                "{prefix} {repair_verb} {article} {target_object} by {verbing} the {machine_synonyms}",
                # use the time machine to repair the bowl on the table
                "{prefix} {verb} {article} {machine_synonyms} to {repair_verb} the {target_object} on {article} {from_receptacle}",
                "{prefix} {repair_verb} {article} {target_object} on {article} {from_receptacle} {verbing} the {machine_synonyms}",
                # use the time machine to convert the red carrot to a green banana
                "{prefix} {verb} {article} {machine_synonyms} to {convert_verb} {article} {target_object} to a {converted_object}",
                "{prefix} {convert_verb} {article} {target_object} to a {converted_object} {verbing} the {machine_synonyms}",
                # use the time machine to convert the red carrot on the table to a green banana
                "{prefix} {verb} {article} {machine_synonyms} to {convert_verb} {article} {target_object} on the {from_receptacle} to a {converted_object}",
                "{prefix} {convert_verb} {article} {target_object} on {article} {from_receptacle} to a {converted_object} {verbing} the {machine_synonyms}",
                # use the time machine to repair the bowl in the fridge
                "{prefix} {verb} {article} {machine_synonyms} to {repair_verb} the {target_object} in the {from_container}",
                "{prefix} {repair_verb} {article} {target_object} in {article} {from_container} {verbing} the {machine_synonyms}",
                # use the time machine to convert the red carrot in the fridge to a green banana
                "{prefix} {verb} {article} {machine_synonyms} to {convert_verb} the {target_object} in the {from_container} to a {converted_object}",
                "{prefix} {convert_verb} {article} {target_object} in {article} {from_container} to a {converted_object} {verbing} the {machine_synonyms}",
            ],
        },
        "colorchanger": {
            "verb": ["operate", "use", "utilize", "make use of"],
            "verbing": ["operating", "using", "utilizing", "making use of"],
            "machine_synonyms": get_object_synonym("color changer"),
            "convert_verb": ["convert", "make", "turn", "change"],
            "basic_verb": ["use"],
            "description_templates": [
                # use the color changer to make the green bowl red
                "{prefix} {convert_verb} {article} the color changer to {convert_verb} the {target_object} {converted_object_color}",
                # use the color changer to change the green bowl on the table to red/red bowl
                "{basic_verb} the color changer to change the {target_object} on the {from_receptacle} to {converted_object_color}",
                "{prefix} {convert_verb} {article} {target_object} to a {converted_object_color} {verbing} the color changer",
                # use the color changer to change the green bowl inside the fridge to red/red bowl
                "{basic_verb} the color changer to change the {target_object} inside the {from_container} to {converted_object_color}",
            ],
            "instruction_templates": [
                # use the color changer to make the green bowl red
                "{prefix} {verb} {article} {machine_synonyms} to {convert_verb} {article} {target_object} {converted_object_color}",
                # use the color changer to change the green bowl on the table to red
                "{prefix} {verb} {article} {machine_synonyms} to {convert_verb} {article} {target_object} on the {from_receptacle} to a {converted_object_color}",
                "{prefix} {convert_verb} {article} {target_object} on {article} {from_receptacle} to a {converted_object_color} {verbing} the {machine_synonyms}",
                "{prefix} {verb} {article} {machine_synonyms} to {convert_verb} {article} {target_object} on the {from_receptacle} to a {converted_object_color}",
                "{prefix} {convert_verb} {article} {target_object} on {article} {from_receptacle} to a {converted_object_color} {verbing} the {machine_synonyms}",
                # use the color changer to change the green bowl in the fridge to red
                "{prefix} {verb} {article} the {machine_synonyms} to change the {target_object} in the {from_container} to {converted_object_color}",
                "{prefix} {convert_verb} {article} {target_object} in {article} {from_container} to a {converted_object_color} {verbing} the {machine_synonyms}",
                "{prefix} {verb} {article} the {machine_synonyms} to change the {target_object} in the {from_container} to {converted_object_color}",
                "{prefix} {convert_verb} {article} {target_object} in {article} {from_container} to a {converted_object_color} {verbing} the {machine_synonyms}",
            ],
        },
        "microwave": {
            "verb": ["operate", "use", "utilize", "make use of"],
            "verbing": ["operating", "using", "utilizing", "making use of"],
            "microwave_verb": ["heat", "destroy"],
            "basic_verb": ["use"],
            "description_templates": [
                # use the microwave
                "{basic_verb} the microwave",
                # use the microwave to heat the burger
                "{basic_verb} the microwave to heat the {target_object}",
                # use the microwave to heat the burger on the table
                "{basic_verb} the microwave to heat the {target_object} on {from_receptacle}",
                # use the microwave to heat the burger in the fridge
                "{basic_verb} the microwave to heat the {target_object} in {from_container}",
            ],
            "instruction_templates": [
                # use the microwave
                "{prefix} {verb} {article} microwave",
                # use the microwave to heat the burger
                "{prefix} {verb} {article} microwave to {microwave_verb} {article} {target_object}",
                "{prefix} microwave {article} {target_object}",
                "{prefix} {verb} {article} {target_object} {verbing} the microwave",
                # use the microwave to heat the burger on the table
                "{prefix} {verb} {article} microwave to {microwave_verb} {article} {target_object} on the {from_receptacle}",
                "{prefix} microwave {article} {target_object} on the {from_receptacle}",
                "{prefix} {verb} {article} {target_object} on the {from_receptacle} {verbing} the microwave",
                # use the microwave to heat the burger inside the fridge
                "{prefix} {verb} {article} microwave to {microwave_verb} {article} {target_object} inside the {from_container}",
                "{prefix} microwave {article} {target_object} inside the {from_contianer}",
                "{prefix} {microwave_verb} {article} {target_object} inside the {from_container} {verbing} the microwave",
            ],
        },
        "3dprinter": {
            "verb": ["operate", "use", "utilize", "make use of"],
            "verbing": ["using", "utilizing", "making use of"],
            "machine_synonyms": get_object_synonym("printer"),
            "basic_verb": ["use"],
            "generate_verb": ["make, build, generate", "3d print", "print", "three d print"],
            "description_templates": [
                # use the 3d printer
                "{basic_verb} the three d printer",
                # use the 3d printer with the mug printer cartridge
                "{basic_verb} the three d printer with {target_object}",
                # use the 3d printer to make a mug
                "{basic_verb} the three d printer to make a {converted_object}",
                # use the 3d printer with the mug printer cartridge to make a mug
                "{basic_verb} the three d printer with {target_object} to make a {converted_object}",
                # use the 3d printer with the mug printer cartridge on the table to make a mug
                "{basic_verb} the three d printer with the {target_object} on the {from_receptacle} to make a {converted_object}",
                # use the 3d printer with the mug printer cartridge in the drawer to make a mug
                "{basic_verb} the three d printer with the {target_object} in the {from_container} to make a {converted_object}",
            ],
            "instruction_templates": [
                # use the 3d printer
                "{prefix} {verb} {article} {machine_synonyms}",
                # use the 3d printer with the mug printer cartridge
                "{prefix} {verb} {article} {machine_synonyms} with {target_object}",
                "{prefix} use the {target_object} to operate the {machine_synonyms}",
                "{prefix} operate {article} {machine_synonyms} {verbing} {target_object}",
                # use the 3d printer to make a mug
                "{prefix} {verb} {article} {machine_synonyms} to 3d print a {target_object}",
                "{prefix} {verb} {article} {machine_synonyms} to {generate_verb} a {converted_object}",
                "{prefix} {generate_verb} a {converted_object} {verbing} {article} {machine_synonyms}",
                # use the 3d printer with the mug printer cartridge to make a mug
                "{prefix} {verb} the printer with the {target_object} to {generate_verb} a {converted_object}",
                "{prefix} {generate_verb} a {converted_object} {verbing} {article} {target_object}",
                # use the 3d printer with the mug printer cartridge on the table to make a mug
                "{prefix} {verb} the printer with the {target_object} on the {from_receptacle} to {generate_verb} a {converted_object}",
                "{prefix} {generate_verb} a {converted_object} {verbing} {article} {target_object} on {from_receptacle}",
                # use the 3d printer with the mug printer cartridge in the drawer to make a mug
                "{prefix} {verb} the printer with the {target_object} in the {from_receptacle} to {generate_verb} a {converted_object}",
                "{prefix} {generate_verb} a {converted_object} {verbing} {article} {target_object} inside the {from_receptacle}",
                "{prefix} {generate_verb} a {converted_object} {verbing} {article} {target_object} from the {from_receptacle}",
            ],
        },
        "coffeemaker": {
            "verb": ["operate", "use", "utilize", "make use of"],
            "verbing": ["operating", "using", "utilizing", "making use of"],
            "machine_synonyms": get_object_synonym("coffee maker"),
            "repair_verb": ["repair", "correct", "put right", "fix"],
            "convert_verb": ["convert", "reverse", "turn"],
            "basic_verb": ["operate"],
            "description_templates": [
                # use the coffee maker
                "{basic_verb} the coffee maker to make a coffee",
                # use the coffee maker with coffee beans
                "{basic_verb} the coffee maker with {target_object}",
                # use the coffee maker with coffee beans on the table
                "{basic_verb} the coffee maker with {target_object} on the {from_receptacle}",
                # use the coffee maker with coffee beans inside the drawer
                "{basic_verb} the coffee maker with {target_object} inside the {from_container}",
            ],
            "instruction_templates": [
                # use the coffee maker
                "{prefix} {verb} {article} {machine_synonyms} to make a coffee",
                # use the coffee maker with coffee beans
                "{prefix} {verb} {article} {machine_synonyms} with {target_object}",
                # use the coffee maker with coffee beans on the table
                "{prefix} {verb} {article} {machine_synonyms} with {target_object} on {article} {from_receptacle}",
                "{prefix} {verb} {article} {machine_synonyms} with {target_object} from {from_receptacle}",
                "{prefix} {verb} {article} {machine_synonyms} with {target_object} on {article} {from_receptacle}",
                "{prefix} {verb} {article} {machine_synonyms} machine with {target_object} from {from_receptacle}",
                # use the coffee maker with coffee beans inside the drawer
                "{prefix} {verb} {article} {machine_synonyms} with {target_object} inside the {from_container}",
                "{prefix} {verb} {article} {machine_synonyms} with {target_object} inside the {from_container}",
            ],
        },
        "fusebox": {
            "verb": ["use", "make use of"],
            "verbing": ["using", "making use of"],
            "machine_synonyms": get_object_synonym("fuse box"),
            "basic_verb": ["use"],
            "description_templates": [
                # use the fuse box to turn the power on
                "{basic_verb} the fuse box to turn the power on",
            ],
            "instruction_templates": [
                "{prefix} {verb} {article} {machine_synonyms} to turn the power on",
                "{prefix} {verb} {article} {machine_synonyms} to turn power on",
                "{prefix} turn the power on {verbing} the {machine_synonyms}",
            ],
        },
        "coffeeunmaker": {
            "verb": ["operate", "use", "utilize", "make use of"],
            "verbing": ["operating", "using", "utilizing", "making use of"],
            "machine_synonyms": get_object_synonym("coffee unmaker"),
            "convert_verb": ["convert", "reverse", "turn back"],
            "basic_verb": ["use"],
            "description_templates": [
                "{basic_verb} the coffee composer to un make the coffee",
                "{basic_verb} the coffee composer to un make the coffee in the {target_object}",
                "{basic_verb} the coffee composer to un make the coffee in the {target_object} on the {from_receptacle}",
            ],
            "instruction_templates": [
                "{prefix} {verb} {article} {machine_synonyms} to un make the coffee",
                "{prefix} {verb} {article} {machine_synonyms} to un make the coffee in the {target_object}",
                "{prefix} {verb} {article} {machine_synonyms} to un make the coffee in the {target_object} on the {from_receptacle}",
                "{prefix} {verb} {article} {machine_synonyms} to un make the coffee in the {target_object} inside the {from_container}",
            ],
        },
        "gravitypad": {
            "verb": ["operate", "use", "utilize", "make use of"],
            "verbing": ["operating", "using", "utilizing", "making use of"],
            "machine_synonyms": get_object_synonym("gravity pad"),
            "convert_verb": ["convert", "reverse", "turn back"],
            "basic_verb": ["use"],
            "description_templates": [
                "{basic_verb} the gravity pad to flip the gravity",
                "{basic_verb} the gravity pad to flip the gravity on {target_object}",
                "{basic_verb} the gravity pad to flip the gravity on the {target_object} on {from_receptacle}",
                "{basic_verb} the gravity pad to flip the gravity on the {target_object} inside the {from_container}",
            ],
            "instruction_templates": [
                "{prefix} {verb} {article} {machine_synonyms} to flip the gravity",
                "{prefix} {verb} {article} {machine_synonyms} to flip the gravity on {target_object}",
                "{prefix} {verb} {article} {machine_synonyms} to flip the gravity on the {target_object} on {from_receptacle}",
                "{prefix} {verb} {article} {machine_synonyms} to flip the gravity on the {target_object} inside the {from_container}",
            ],
        },
    }
)
