# flake8: noqa WPS226
import random
from types import MappingProxyType

from emma_datasets.constants.simbot.simbot import get_object_synonym


def merge_strings(string1: str, string2: str) -> str:
    """Dummy way to prevent noqas."""
    return f"{string1} {string2}"


OBJECT_META_TEMPLATE = MappingProxyType(
    {
        "pickup": {
            "paraphrases": [
                "collect the {target_object_color} {target_object} from inside the {from_container}",
                "collect the {target_object_color} {target_object} from the {from_container}",
                "collect the {target_object_color} {target_object} from the {from_receptacle}",
                "collect the {target_object} from inside the {from_container}",
                "collect the {target_object} from the {from_container}",
                "collect the {target_object} from the {from_receptacle}",
                "fetch the {target_object_color} {target_object} from inside the {from_container}",
                "fetch the {target_object_color} {target_object} from the {from_container}",
                "fetch the {target_object_color} {target_object} from the {from_receptacle}",
                "fetch the {target_object} from inside the {from_container}",
                "fetch the {target_object} from the {from_container}",
                "fetch the {target_object} from the {from_receptacle}",
                "get the {target_object_color} {target_object} from inside the {from_container}",
                "get the {target_object_color} {target_object} from the {from_container}",
                "get the {target_object_color} {target_object} from the {from_receptacle}",
                "get the {target_object} from inside the {from_container}",
                "get the {target_object} from the {from_container}",
                "get the {target_object} from the {from_receptacle}",
                "grab the {target_object_color} {target_object} from inside the {from_container}",
                "grab the {target_object_color} {target_object} from the {from_container}",
                "grab the {target_object_color} {target_object} from the {from_receptacle}",
                "grab the {target_object} from inside the {from_container}",
                "grab the {target_object} from the {from_container}",
                "grab the {target_object} from the {from_receptacle}",
                "pick the {target_object_color} {target_object} from inside the {from_container}",
                "pick the {target_object_color} {target_object} from the {from_container}",
                "pick the {target_object_color} {target_object} from the {from_receptacle}",
                "pick the {target_object} from inside the {from_container}",
                "pick the {target_object} from the {from_container}",
                "pick the {target_object} from the {from_receptacle}",
                "retrieve the {target_object_color} {target_object} from inside the {from_container}",
                "retrieve the {target_object_color} {target_object} from the {from_container}",
                "retrieve the {target_object_color} {target_object} from the {from_receptacle}",
                "retrieve the {target_object} from inside the {from_container}",
                "retrieve the {target_object} from the {from_container}",
                "retrieve the {target_object} from the {from_receptacle}",
            ],
        },
        "place": {
            "paraphrases": [
                "leave the {target_object_color} {target_object} in the {to_container}",
                "leave the {target_object_color} {target_object} in the {to_receptacle_color} {to_container}",
                "leave the {target_object_color} {target_object} inside the {to_container}",
                "leave the {target_object_color} {target_object} inside the {to_receptacle_color} {to_container}",
                "leave the {target_object_color} {target_object} on the {to_receptacle_color} {to_receptacle}",
                "leave the {target_object_color} {target_object} on the {to_receptacle}",
                "leave the {target_object} in the {to_container}",
                "leave the {target_object} in the {to_receptacle_color} {to_container}",
                "leave the {target_object} inside the {to_container}",
                "leave the {target_object} inside the {to_receptacle_color} {to_container}",
                "leave the {target_object} on the {to_receptacle_color} {to_receptacle}",
                "leave the {target_object} on the {to_receptacle}",
                "place the {target_object_color} {target_object} in the {to_container}",
                "place the {target_object_color} {target_object} in the {to_receptacle_color} {to_container}",
                "place the {target_object_color} {target_object} inside the {to_container}",
                "place the {target_object_color} {target_object} inside the {to_receptacle_color} {to_container}",
                "place the {target_object_color} {target_object} on the {to_receptacle_color} {to_receptacle}",
                "place the {target_object_color} {target_object} on the {to_receptacle}",
                "place the {target_object} in the {to_container}",
                "place the {target_object} in the {to_receptacle_color} {to_container}",
                "place the {target_object} inside the {to_container}",
                "place the {target_object} inside the {to_receptacle_color} {to_container}",
                "place the {target_object} on the {to_receptacle_color} {to_receptacle}",
                "place the {target_object} on the {to_receptacle}",
                "put the {target_object_color} {target_object} in the {to_container}",
                "put the {target_object_color} {target_object} in the {to_receptacle_color} {to_container}",
                "put the {target_object_color} {target_object} inside the {to_container}",
                "put the {target_object_color} {target_object} inside the {to_receptacle_color} {to_container}",
                "put the {target_object_color} {target_object} on the {to_receptacle_color} {to_receptacle}",
                "put the {target_object_color} {target_object} on the {to_receptacle}",
                "put the {target_object} in the {to_container}",
                "put the {target_object} in the {to_receptacle_color} {to_container}",
                "put the {target_object} inside the {to_container}",
                "put the {target_object} inside the {to_receptacle_color} {to_container}",
                "put the {target_object} on the {to_receptacle_color} {to_receptacle}",
                "put the {target_object} on the {to_receptacle}",
                "set the {target_object_color} {target_object} in the {to_container}",
                "set the {target_object_color} {target_object} in the {to_receptacle_color} {to_container}",
                "set the {target_object_color} {target_object} inside the {to_container}",
                "set the {target_object_color} {target_object} inside the {to_receptacle_color} {to_container}",
                "set the {target_object_color} {target_object} on the {to_receptacle_color} {to_receptacle}",
                "set the {target_object_color} {target_object} on the {to_receptacle}",
                "set the {target_object} in the {to_container}",
                "set the {target_object} in the {to_receptacle_color} {to_container}",
                "set the {target_object} inside the {to_container}",
                "set the {target_object} inside the {to_receptacle_color} {to_container}",
                "set the {target_object} on the {to_receptacle_color} {to_receptacle}",
            ]
        },
        "pour": {
            "paraphrases": [
                "pour the {target_object}",
                "pour the {target_object} in the {to_receptacle}",
                "pour the {target_object} into the {to_receptacle}put {target_object} on the {to_receptacle}",
                "pour the {target_object} on the {to_receptacle}",
                "pour {target_object}",
                "pour {target_object} in the {to_receptacle}",
                "pour {target_object} into the {to_receptacle}",
                "pour {target_object} on the {to_receptacle}",
                "put the {target_object}",
                "put the {target_object} in the {to_receptacle}",
                "put the {target_object} into the {to_receptacle}",
                "put {target_object}",
                "put {target_object} in the {to_receptacle}",
                "put {target_object} into the {to_receptacle}put the {target_object} on the {to_receptacle}",
            ],
        },
        "fill": {
            "paraphrases": [
                "fill the {target_object_color} {target_object}",
                "fill the {target_object_color} {target_object} in the {interaction_object}",
                "fill the {target_object_color} {target_object} with water",
                "fill the {target_object_color} {target_object} with water from the {interaction_object}",
                "fill the {target_object}",
                "fill the {target_object} in the {interaction_object}",
                "fill the {target_object} with water",
                "fill the {target_object} with water from the {interaction_object}",
                "fill up the {target_object_color} {target_object}",
                "fill up the {target_object_color} {target_object} in the {interaction_object}",
                "fill up the {target_object_color} {target_object} with water",
                "fill up the {target_object_color} {target_object} with water from the {interaction_object}",
                "fill up the {target_object}",
                "fill up the {target_object} in the {interaction_object}",
                "fill up the {target_object} with water",
                "fill up the {target_object} with water from the {interaction_object}",
                "put water from the {interaction_object} in the {target_object_color} {target_object}",
                "put water from the {interaction_object} in the {target_object}",
                "put water from the {interaction_object} into the {target_object_color} {target_object}",
                "put water from the {interaction_object} into the {target_object}",
                "use the {interaction_object} to fill the {target_object_color} {target_object}",
                "use the {interaction_object} to fill the {target_object}",
                "use the {interaction_object} to fill up the {target_object_color} {target_object}",
                "use the {interaction_object} to fill up the {target_object}",
            ]
        },
        "clean": {
            "paraphrases": [
                "clean the dirty {target_object_color} {target_object} in the {interaction_object}",
                "clean the dirty {target_object} in the {interaction_object}",
                "clean the {target_object_color} {target_object} in the {interaction_object}",
                "clean the {target_object} in the {interaction_object}",
                "make the dirty {target_object_color} {target_object} clean",
                "make the dirty {target_object} clean",
                "make the {target_object_color} {target_object} clean",
                "make the {target_object} clean",
                "rinse off the dirty {target_object_color} {target_object} in the {interaction_object}",
                "rinse off the dirty {target_object} in the {interaction_object}",
                "rinse off the {target_object_color} {target_object} in the {interaction_object}",
                "rinse off the {target_object} in the {interaction_object}",
                "rinse the dirty {target_object_color} {target_object} in the {interaction_object}",
                "rinse the dirty {target_object} in the {interaction_object}",
                "rinse the {target_object_color} {target_object} in the {interaction_object}",
                "rinse the {target_object} in the {interaction_object}",
                "use the {interaction_object} to clean up the dirty {target_object_color} {target_object}",
                "use the {interaction_object} to clean up the dirty {target_object}",
                "use the {interaction_object} to clean up the {target_object_color} {target_object}",
                "use the {interaction_object} to clean up the {target_object}",
                "wash off the dirty {target_object_color} {target_object} in the {interaction_object}",
                "wash off the dirty {target_object} in the {interaction_object}",
                "wash off the {target_object_color} {target_object} in the {interaction_object}",
                "wash off the {target_object} in the {interaction_object}",
                "wash the dirty {target_object_color} {target_object} in the {interaction_object}",
                "wash the dirty {target_object} in the {interaction_object}",
                "wash the {target_object_color} {target_object} in the {interaction_object}",
                "wash the {target_object} in the {interaction_object}",
            ],
        },
        "interact": {
            "YesterdayMachine_01": {
                "machine_synonym": get_object_synonym("YesterdayMachine_01"),
                "paraphrases": [
                    # repair target object without color
                    "fix the {target_object}",
                    "fix the {target_object} using the {machine_synonym}",
                    "make a journey through time to fix the {target_object} with the {machine_synonym}",
                    "make use of the {machine_synonym} to restore the {target_object}",
                    "repair the broken {target_object}",
                    "repair the broken {target_object} using the {machine_synonym}",
                    "repair the {target_object}",
                    "repair the {target_object} using the {machine_synonym}",
                    "restore the {target_object}",
                    "restore the {target_object} using the {machine_synonym}",
                    "take advantage of the {machine_synonym} to restore the {target_object}'s shape",
                    "take the {target_object} back in time to restore it with the {machine_synonym}",
                    "turn back the clock with the {machine_synonym} to mend the broken {target_object}",
                    "use the {machine_synonym} to repair the broken {target_object}",
                    "use the {machine_synonym} to repair the {target_object}",
                    "use the {machine_synonym} to return the {target_object} to its undamaged condition",
                    "utilize the {machine_synonym} to mend the {target_object}'s cracks",
                    "utilize the {machine_synonym}'s abilities to repair the {target_object}'s damages",
                    # repair target object with color
                    "fix the {target_object_color} {target_object}",
                    "fix the {target_object_color} {target_object} using the {machine_synonym}",
                    "make a journey through time to fix the {target_object_color} {target_object} with the {machine_synonym}",
                    "make use of the {machine_synonym} to restore the {target_object_color} {target_object}",
                    "repair the broken {target_object_color} {target_object}",
                    "repair the broken {target_object_color} {target_object} using the {machine_synonym}",
                    "repair the {target_object_color} {target_object}",
                    "repair the {target_object_color} {target_object} using the {machine_synonym}",
                    "restore the {target_object_color} {target_object}",
                    "restore the {target_object_color} {target_object} using the {machine_synonym}",
                    "take advantage of the {machine_synonym} to restore the {target_object_color} {target_object}'s shape",
                    "take the {target_object_color} {target_object} back in time to restore it with the {machine_synonym}",
                    "turn back the clock with the {machine_synonym} to mend the broken {target_object_color} {target_object}",
                    "use the {machine_synonym} to repair the broken {target_object_color} {target_object}",
                    "use the {machine_synonym} to repair the {target_object_color} {target_object}",
                    "use the {machine_synonym} to return the {target_object_color} {target_object} to its undamaged condition",
                    "utilize the {machine_synonym} to mend the {target_object_color} {target_object}'s cracks",
                    "utilize the {machine_synonym}'s abilities to repair the {target_object_color} {target_object}'s damages",
                    "activate the {machine_synonym} to turn a {target_object} to a {converted_object}",
                    # convert an object into another object
                    "activate the {machine_synonym} to turn a {target_object_color} {target_object} into a {converted_object_color} {converted_object}",
                    "activate the {machine_synonym} to turn a {target_object_color} {target_object} into a {converted_object}",
                    "activate the {machine_synonym} to turn a {target_object_color} {target_object} to a {converted_object_color} {converted_object}",
                    "activate the {machine_synonym} to turn a {target_object_color} {target_object} to a {converted_object}",
                    "activate the {machine_synonym} to turn a {target_object} into a {converted_object_color} {converted_object}",
                    "activate the {machine_synonym} to turn a {target_object} into a {converted_object}",
                    "activate the {machine_synonym} to turn a {target_object} to a {converted_object_color} {converted_object}",
                    "change the {target_object_color} {target_object} into a {converted_object_color} {converted_object}",
                    "change the {target_object_color} {target_object} into a {converted_object_color} {converted_object} using the {machine_synonym}",
                    "change the {target_object_color} {target_object} into a {converted_object}",
                    "change the {target_object_color} {target_object} into a {converted_object} using the {machine_synonym}",
                    "change the {target_object_color} {target_object} to a {converted_object_color} {converted_object}",
                    "change the {target_object_color} {target_object} to a {converted_object_color} {converted_object} using the {machine_synonym}",
                    "change the {target_object_color} {target_object} to a {converted_object}",
                    "change the {target_object_color} {target_object} to a {converted_object} using the {machine_synonym}",
                    "change the {target_object} into a {converted_object_color} {converted_object}",
                    "change the {target_object} into a {converted_object_color} {converted_object} using the {machine_synonym}",
                    "change the {target_object} into a {converted_object}",
                    "change the {target_object} into a {converted_object} using the {machine_synonym}",
                    "change the {target_object} to a {converted_object_color} {converted_object}",
                    "change the {target_object} to a {converted_object_color} {converted_object} using the {machine_synonym}",
                    "change the {target_object} to a {converted_object}",
                    "change the {target_object} to a {converted_object} using the {machine_synonym}",
                    "convert the {target_object_color} {target_object} into a {converted_object_color} {converted_object}",
                    "convert the {target_object_color} {target_object} into a {converted_object_color} {converted_object} using the {machine_synonym}",
                    "convert the {target_object_color} {target_object} into a {converted_object}",
                    "convert the {target_object_color} {target_object} into a {converted_object} using the {machine_synonym}",
                    "convert the {target_object_color} {target_object} to a {converted_object_color} {converted_object}",
                    "convert the {target_object_color} {target_object} to a {converted_object_color} {converted_object} using the {machine_synonym}",
                    "convert the {target_object_color} {target_object} to a {converted_object}",
                    "convert the {target_object_color} {target_object} to a {converted_object} using the {machine_synonym}",
                    "convert the {target_object} into a {converted_object_color} {converted_object}",
                    "convert the {target_object} into a {converted_object_color} {converted_object} using the {machine_synonym}",
                    "convert the {target_object} into a {converted_object}",
                    "convert the {target_object} into a {converted_object} using the {machine_synonym}",
                    "convert the {target_object} to a {converted_object_color} {converted_object}",
                    "convert the {target_object} to a {converted_object_color} {converted_object} using the {machine_synonym}",
                    "convert the {target_object} to a {converted_object}",
                    "convert the {target_object} to a {converted_object} using the {machine_synonym}",
                    "turn the {target_object_color} {target_object} into a {converted_object_color} {converted_object}",
                    "turn the {target_object_color} {target_object} into a {converted_object_color} {converted_object} using the {machine_synonym}",
                    "turn the {target_object_color} {target_object} into a {converted_object}",
                    "turn the {target_object_color} {target_object} into a {converted_object} using the {machine_synonym}",
                    "turn the {target_object_color} {target_object} to a {converted_object_color} {converted_object}",
                    "turn the {target_object_color} {target_object} to a {converted_object_color} {converted_object} using the {machine_synonym}",
                    "turn the {target_object_color} {target_object} to a {converted_object}",
                    "turn the {target_object_color} {target_object} to a {converted_object} using the {machine_synonym}",
                    "turn the {target_object} into a {converted_object_color} {converted_object}",
                    "turn the {target_object} into a {converted_object_color} {converted_object} using the {machine_synonym}",
                    "turn the {target_object} into a {converted_object}",
                    "turn the {target_object} into a {converted_object} using the {machine_synonym}",
                    "turn the {target_object} to a {converted_object_color} {converted_object}",
                    "turn the {target_object} to a {converted_object_color} {converted_object} using the {machine_synonym}",
                    "turn the {target_object} to a {converted_object}",
                    "turn the {target_object} to a {converted_object} using the {machine_synonym}",
                    "use the {machine_synonym} to change the {target_object_color} {target_object} into a {converted_object_color} {converted_object}",
                    "use the {machine_synonym} to change the {target_object_color} {target_object} into a {converted_object}",
                    "use the {machine_synonym} to change the {target_object_color} {target_object} to a {converted_object_color} {converted_object}",
                    "use the {machine_synonym} to change the {target_object_color} {target_object} to a {converted_object}",
                    "use the {machine_synonym} to change the {target_object} into a {converted_object_color} {converted_object}",
                    "use the {machine_synonym} to change the {target_object} into a {converted_object}",
                    "use the {machine_synonym} to change the {target_object} to a {converted_object_color} {converted_object}",
                    "use the {machine_synonym} to change the {target_object} to a {converted_object}",
                    "use the {machine_synonym} to transform a {target_object_color} {target_object} into a {converted_object_color} {converted_object}",
                    "use the {machine_synonym} to transform a {target_object_color} {target_object} into a {converted_object}",
                    "use the {machine_synonym} to transform a {target_object_color} {target_object} to a {converted_object_color} {converted_object}",
                    "use the {machine_synonym} to transform a {target_object_color} {target_object} to a {converted_object}",
                    "use the {machine_synonym} to transform a {target_object} into a {converted_object_color} {converted_object}",
                    "use the {machine_synonym} to transform a {target_object} into a {converted_object}",
                    "use the {machine_synonym} to transform a {target_object} to a {converted_object_color} {converted_object}",
                    "use the {machine_synonym} to transform a {target_object} to a {converted_object}",
                    "use the {machine_synonym} to turn the {target_object_color} {target_object} into a {converted_object_color} {converted_object}",
                    "use the {machine_synonym} to turn the {target_object_color} {target_object} into a {converted_object}",
                    "use the {machine_synonym} to turn the {target_object_color} {target_object} to a {converted_object_color} {converted_object}",
                    "use the {machine_synonym} to turn the {target_object_color} {target_object} to a {converted_object}",
                    "use the {machine_synonym} to turn the {target_object} into a {converted_object_color} {converted_object}",
                    "use the {machine_synonym} to turn the {target_object} into a {converted_object}",
                    "use the {machine_synonym} to turn the {target_object} to a {converted_object_color} {converted_object}",
                    "use the {machine_synonym} to turn the {target_object} to a {converted_object}",
                ],
            },
            "CoffeeUnMaker_01": {
                "machine_synonym": get_object_synonym("CoffeeUnMaker_01"),
                "paraphrases": [
                    merge_strings(
                        string1="activate the {machine_synonym} to turn the coffee",
                        string2=f"into {random.choice(get_object_synonym('CoffeeBeans_01'))}",
                    ),
                    merge_strings(
                        string1="activate the {machine_synonym} to turn the {target_object}",
                        string2=f"into {random.choice(get_object_synonym('CoffeeBeans_01'))}",
                    ),
                    merge_strings(
                        string1="activate the {machine_synonym} to change the coffee",
                        string2=f"into {random.choice(get_object_synonym('CoffeeBeans_01'))}",
                    ),
                    merge_strings(
                        string1="activate the {machine_synonym} to change the {target_object}",
                        string2=f"into {random.choice(get_object_synonym('CoffeeBeans_01'))}",
                    ),
                    merge_strings(
                        string1="use the {machine_synonym} to turn the coffee",
                        string2=f"into {random.choice(get_object_synonym('CoffeeBeans_01'))}",
                    ),
                    merge_strings(
                        string1="use the {machine_synonym} to turn the {target_object}",
                        string2=f"into {random.choice(get_object_synonym('CoffeeBeans_01'))}",
                    ),
                    merge_strings(
                        string1="use the {machine_synonym} to change the coffee",
                        string2=f"into {random.choice(get_object_synonym('CoffeeBeans_01'))}",
                    ),
                    merge_strings(
                        string1="use the {machine_synonym} to change the {target_object}",
                        string2=f"into {random.choice(get_object_synonym('CoffeeBeans_01'))}",
                    ),
                    merge_strings(
                        string1=f"turn the coffee to {random.choice(get_object_synonym('CoffeeBeans_01'))}",
                        string2="by using the {machine_synonym}",
                    ),
                    merge_strings(
                        string1=f"turn the coffee to a {random.choice(get_object_synonym('CoffeeBeans_01'))}",
                        string2="by using the {machine_synonym}",
                    ),
                    merge_strings(
                        string1=f"turn the coffee into {random.choice(get_object_synonym('CoffeeBeans_01'))}",
                        string2="by using the {machine_synonym}",
                    ),
                    merge_strings(
                        string1=f"turn the coffee into a {random.choice(get_object_synonym('CoffeeBeans_01'))}",
                        string2="by using the {machine_synonym}",
                    ),
                ],
            },
        },
        #     "timemachine": {
        #         "verb": ["operate", "use", "utilize", "make use of"],
        #         "verbing": ["operating", "using", "utilizing", "making use of"],
        #         "machine_synonyms": get_object_synonym("time machine"),
        #         "repair_verb": ["repair", "fix"],
        #         "convert_verb": ["convert", "reverse", "turn"],
        #         "basic_verb": ["use"],
        #         "description_templates": [
        #             # use the time machine
        #             "{basic_verb} the time machine",
        #             # use the time machine to repair the bowl
        #             "{basic_verb} the time machine to repair the {target_object}",
        #             # use the time machine to repair the bowl on the table
        #             "{basic_verb} the time machine to repair the {target_object} on the {from_receptacle}",
        #             # use the time machine to convert the red carrot to a green banana
        #             "{basic_verb} the time machine to convert the {target_object} to a {converted_object}",
        #             # use the time machine to convert the red carrot on the table to a green banana
        #             "{basic_verb} the time machine to convert the {target_object} on the {from_receptacle} to a {converted_object}",
        #             # use the time machine to repair the bowl in the fridge
        #             "{basic_verb} the time machine to repair the {target_object} inside the {from_container}",
        #             # use the time machine to convert the red carrot in the fridge to a green banana
        #             "{basic_verb} the time machine to convert the {target_object} in the {from_container} to a {converted_object}",
        #         ],
        #         "instruction_templates": [
        #             # use the time machine
        #             "{prefix} {verb} {article} time machine",
        #             # use the time machine to repair the bowl
        #             "{prefix} {verb} {article} {machine_synonyms} to {repair_verb} the {target_object}",
        #             "{prefix} {repair_verb} {article} {target_object} by {verbing} the {machine_synonyms}",
        #             # use the time machine to repair the bowl on the table
        #             "{prefix} {verb} {article} {machine_synonyms} to {repair_verb} the {target_object} on {article} {from_receptacle}",
        #             "{prefix} {repair_verb} {article} {target_object} on {article} {from_receptacle} {verbing} the {machine_synonyms}",
        #             # use the time machine to convert the red carrot to a green banana
        #             "{prefix} {verb} {article} {machine_synonyms} to {convert_verb} {article} {target_object} to a {converted_object}",
        #             "{prefix} {convert_verb} {article} {target_object} to a {converted_object} {verbing} the {machine_synonyms}",
        #             # use the time machine to convert the red carrot on the table to a green banana
        #             "{prefix} {verb} {article} {machine_synonyms} to {convert_verb} {article} {target_object} on the {from_receptacle} to a {converted_object}",
        #             "{prefix} {convert_verb} {article} {target_object} on {article} {from_receptacle} to a {converted_object} {verbing} the {machine_synonyms}",
        #             # use the time machine to repair the bowl in the fridge
        #             "{prefix} {verb} {article} {machine_synonyms} to {repair_verb} the {target_object} in the {from_container}",
        #             "{prefix} {repair_verb} {article} {target_object} in {article} {from_container} {verbing} the {machine_synonyms}",
        #             # use the time machine to convert the red carrot in the fridge to a green banana
        #             "{prefix} {verb} {article} {machine_synonyms} to {convert_verb} the {target_object} in the {from_container} to a {converted_object}",
        #             "{prefix} {convert_verb} {article} {target_object} in {article} {from_container} to a {converted_object} {verbing} the {machine_synonyms}",
        #         ],
        #     },
        #     "colorchanger": {
        #         "verb": ["operate", "use", "utilize", "make use of"],
        #         "verbing": ["operating", "using", "utilizing", "making use of"],
        #         "machine_synonyms": get_object_synonym("color changer"),
        #         "convert_verb": ["convert", "make", "turn", "change"],
        #         "basic_verb": ["use"],
        #         "description_templates": [
        #             # use the color changer to make the green bowl red
        #             "{prefix} {convert_verb} {article} the color changer to {convert_verb} the {target_object} {converted_object_color}",
        #             # use the color changer to change the green bowl on the table to red/red bowl
        #             "{basic_verb} the color changer to change the {target_object} on the {from_receptacle} to {converted_object_color}",
        #             "{prefix} {convert_verb} {article} {target_object} to a {converted_object_color} {verbing} the color changer",
        #             # use the color changer to change the green bowl inside the fridge to red/red bowl
        #             "{basic_verb} the color changer to change the {target_object} inside the {from_container} to {converted_object_color}",
        #         ],
        #         "instruction_templates": [
        #             # use the color changer to make the green bowl red
        #             "{prefix} {verb} {article} {machine_synonyms} to {convert_verb} {article} {target_object} {converted_object_color}",
        #             # use the color changer to change the green bowl on the table to red
        #             "{prefix} {verb} {article} {machine_synonyms} to {convert_verb} {article} {target_object} on the {from_receptacle} to a {converted_object_color}",
        #             "{prefix} {convert_verb} {article} {target_object} on {article} {from_receptacle} to a {converted_object_color} {verbing} the {machine_synonyms}",
        #             "{prefix} {verb} {article} {machine_synonyms} to {convert_verb} {article} {target_object} on the {from_receptacle} to a {converted_object_color}",
        #             "{prefix} {convert_verb} {article} {target_object} on {article} {from_receptacle} to a {converted_object_color} {verbing} the {machine_synonyms}",
        #             # use the color changer to change the green bowl in the fridge to red
        #             "{prefix} {verb} {article} the {machine_synonyms} to change the {target_object} in the {from_container} to {converted_object_color}",
        #             "{prefix} {convert_verb} {article} {target_object} in {article} {from_container} to a {converted_object_color} {verbing} the {machine_synonyms}",
        #             "{prefix} {verb} {article} the {machine_synonyms} to change the {target_object} in the {from_container} to {converted_object_color}",
        #             "{prefix} {convert_verb} {article} {target_object} in {article} {from_container} to a {converted_object_color} {verbing} the {machine_synonyms}",
        #         ],
        #     },
        #     "microwave": {
        #         "verb": ["operate", "use", "utilize", "make use of"],
        #         "verbing": ["operating", "using", "utilizing", "making use of"],
        #         "microwave_verb": ["heat", "destroy"],
        #         "basic_verb": ["use"],
        #         "description_templates": [
        #             # use the microwave
        #             "{basic_verb} the microwave",
        #             # use the microwave to heat the burger
        #             "{basic_verb} the microwave to heat the {target_object}",
        #             # use the microwave to heat the burger on the table
        #             "{basic_verb} the microwave to heat the {target_object} on {from_receptacle}",
        #             # use the microwave to heat the burger in the fridge
        #             "{basic_verb} the microwave to heat the {target_object} in {from_container}",
        #         ],
        #         "instruction_templates": [
        #             # use the microwave
        #             "{prefix} {verb} {article} microwave",
        #             # use the microwave to heat the burger
        #             "{prefix} {verb} {article} microwave to {microwave_verb} {article} {target_object}",
        #             "{prefix} microwave {article} {target_object}",
        #             "{prefix} {verb} {article} {target_object} {verbing} the microwave",
        #             # use the microwave to heat the burger on the table
        #             "{prefix} {verb} {article} microwave to {microwave_verb} {article} {target_object} on the {from_receptacle}",
        #             "{prefix} microwave {article} {target_object} on the {from_receptacle}",
        #             "{prefix} {verb} {article} {target_object} on the {from_receptacle} {verbing} the microwave",
        #             # use the microwave to heat the burger inside the fridge
        #             "{prefix} {verb} {article} microwave to {microwave_verb} {article} {target_object} inside the {from_container}",
        #             "{prefix} microwave {article} {target_object} inside the {from_contianer}",
        #             "{prefix} {microwave_verb} {article} {target_object} inside the {from_container} {verbing} the microwave",
        #         ],
        #     },
        #     "3dprinter": {
        #         "verb": ["operate", "use", "utilize", "make use of"],
        #         "verbing": ["using", "utilizing", "making use of"],
        #         "machine_synonyms": get_object_synonym("printer"),
        #         "basic_verb": ["use"],
        #         "generate_verb": ["make, build, generate", "3d print", "print", "three d print"],
        #         "description_templates": [
        #             # use the 3d printer
        #             "{basic_verb} the three d printer",
        #             # use the 3d printer with the mug printer cartridge
        #             "{basic_verb} the three d printer with {target_object}",
        #             # use the 3d printer to make a mug
        #             "{basic_verb} the three d printer to make a {converted_object}",
        #             # use the 3d printer with the mug printer cartridge to make a mug
        #             "{basic_verb} the three d printer with {target_object} to make a {converted_object}",
        #             # use the 3d printer with the mug printer cartridge on the table to make a mug
        #             "{basic_verb} the three d printer with the {target_object} on the {from_receptacle} to make a {converted_object}",
        #             # use the 3d printer with the mug printer cartridge in the drawer to make a mug
        #             "{basic_verb} the three d printer with the {target_object} in the {from_container} to make a {converted_object}",
        #         ],
        #         "instruction_templates": [
        #             # use the 3d printer
        #             "{prefix} {verb} {article} {machine_synonyms}",
        #             # use the 3d printer with the mug printer cartridge
        #             "{prefix} {verb} {article} {machine_synonyms} with {target_object}",
        #             "{prefix} use the {target_object} to operate the {machine_synonyms}",
        #             "{prefix} operate {article} {machine_synonyms} {verbing} {target_object}",
        #             # use the 3d printer to make a mug
        #             "{prefix} {verb} {article} {machine_synonyms} to 3d print a {target_object}",
        #             "{prefix} {verb} {article} {machine_synonyms} to {generate_verb} a {converted_object}",
        #             "{prefix} {generate_verb} a {converted_object} {verbing} {article} {machine_synonyms}",
        #             # use the 3d printer with the mug printer cartridge to make a mug
        #             "{prefix} {verb} the printer with the {target_object} to {generate_verb} a {converted_object}",
        #             "{prefix} {generate_verb} a {converted_object} {verbing} {article} {target_object}",
        #             # use the 3d printer with the mug printer cartridge on the table to make a mug
        #             "{prefix} {verb} the printer with the {target_object} on the {from_receptacle} to {generate_verb} a {converted_object}",
        #             "{prefix} {generate_verb} a {converted_object} {verbing} {article} {target_object} on {from_receptacle}",
        #             # use the 3d printer with the mug printer cartridge in the drawer to make a mug
        #             "{prefix} {verb} the printer with the {target_object} in the {from_receptacle} to {generate_verb} a {converted_object}",
        #             "{prefix} {generate_verb} a {converted_object} {verbing} {article} {target_object} inside the {from_receptacle}",
        #             "{prefix} {generate_verb} a {converted_object} {verbing} {article} {target_object} from the {from_receptacle}",
        #         ],
        #     },
        #     "coffeemaker": {
        #         "verb": ["operate", "use", "utilize", "make use of"],
        #         "verbing": ["operating", "using", "utilizing", "making use of"],
        #         "machine_synonyms": get_object_synonym("coffee maker"),
        #         "repair_verb": ["repair", "correct", "put right", "fix"],
        #         "convert_verb": ["convert", "reverse", "turn"],
        #         "basic_verb": ["operate"],
        #         "description_templates": [
        #             # use the coffee maker
        #             "{basic_verb} the coffee maker to make a coffee",
        #             # use the coffee maker with coffee beans
        #             "{basic_verb} the coffee maker with {target_object}",
        #             # use the coffee maker with coffee beans on the table
        #             "{basic_verb} the coffee maker with {target_object} on the {from_receptacle}",
        #             # use the coffee maker with coffee beans inside the drawer
        #             "{basic_verb} the coffee maker with {target_object} inside the {from_container}",
        #         ],
        #         "instruction_templates": [
        #             # use the coffee maker
        #             "{prefix} {verb} {article} {machine_synonyms} to make a coffee",
        #             # use the coffee maker with coffee beans
        #             "{prefix} {verb} {article} {machine_synonyms} with {target_object}",
        #             # use the coffee maker with coffee beans on the table
        #             "{prefix} {verb} {article} {machine_synonyms} with {target_object} on {article} {from_receptacle}",
        #             "{prefix} {verb} {article} {machine_synonyms} with {target_object} from {from_receptacle}",
        #             "{prefix} {verb} {article} {machine_synonyms} with {target_object} on {article} {from_receptacle}",
        #             "{prefix} {verb} {article} {machine_synonyms} machine with {target_object} from {from_receptacle}",
        #             # use the coffee maker with coffee beans inside the drawer
        #             "{prefix} {verb} {article} {machine_synonyms} with {target_object} inside the {from_container}",
        #             "{prefix} {verb} {article} {machine_synonyms} with {target_object} inside the {from_container}",
        #         ],
        #     },
        #     "fusebox": {
        #         "verb": ["use", "make use of"],
        #         "verbing": ["using", "making use of"],
        #         "machine_synonyms": get_object_synonym("fuse box"),
        #         "basic_verb": ["use"],
        #         "description_templates": [
        #             # use the fuse box to turn the power on
        #             "{basic_verb} the fuse box to turn the power on",
        #         ],
        #         "instruction_templates": [
        #             "{prefix} {verb} {article} {machine_synonyms} to turn the power on",
        #             "{prefix} {verb} {article} {machine_synonyms} to turn power on",
        #             "{prefix} turn the power on {verbing} the {machine_synonyms}",
        #         ],
        #     },
        #     "coffeeunmaker": {
        #         "verb": ["operate", "use", "utilize", "make use of"],
        #         "verbing": ["operating", "using", "utilizing", "making use of"],
        #         "machine_synonyms": get_object_synonym("coffee unmaker"),
        #         "convert_verb": ["convert", "reverse", "turn back"],
        #         "basic_verb": ["use"],
        #         "description_templates": [
        #             "{basic_verb} the coffee composer to un make the coffee",
        #             "{basic_verb} the coffee composer to un make the coffee in the {target_object}",
        #             "{basic_verb} the coffee composer to un make the coffee in the {target_object} on the {from_receptacle}",
        #         ],
        #         "instruction_templates": [
        #             "{prefix} {verb} {article} {machine_synonyms} to un make the coffee",
        #             "{prefix} {verb} {article} {machine_synonyms} to un make the coffee in the {target_object}",
        #             "{prefix} {verb} {article} {machine_synonyms} to un make the coffee in the {target_object} on the {from_receptacle}",
        #             "{prefix} {verb} {article} {machine_synonyms} to un make the coffee in the {target_object} inside the {from_container}",
        #         ],
        #     },
        #     "gravitypad": {
        #         "verb": ["operate", "use", "utilize", "make use of"],
        #         "verbing": ["operating", "using", "utilizing", "making use of"],
        #         "machine_synonyms": get_object_synonym("gravity pad"),
        #         "convert_verb": ["convert", "reverse", "turn back"],
        #         "basic_verb": ["use"],
        #         "description_templates": [
        #             "{basic_verb} the gravity pad to flip the gravity",
        #             "{basic_verb} the gravity pad to flip the gravity on {target_object}",
        #             "{basic_verb} the gravity pad to flip the gravity on the {target_object} on {from_receptacle}",
        #             "{basic_verb} the gravity pad to flip the gravity on the {target_object} inside the {from_container}",
        #         ],
        #         "instruction_templates": [
        #             "{prefix} {verb} {article} {machine_synonyms} to flip the gravity",
        #             "{prefix} {verb} {article} {machine_synonyms} to flip the gravity on {target_object}",
        #             "{prefix} {verb} {article} {machine_synonyms} to flip the gravity on the {target_object} on {from_receptacle}",
        #             "{prefix} {verb} {article} {machine_synonyms} to flip the gravity on the {target_object} inside the {from_container}",
        #         ],
        #     },
    }
)
