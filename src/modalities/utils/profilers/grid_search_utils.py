import copy
from dataclasses import dataclass
from itertools import product
from typing import Any


@dataclass
class GridSearchItem:
    name: str
    values: list[Any]


@dataclass
class ConfigValue:
    name: str
    value: Any


class GridSearchUtils:
    @staticmethod
    def get_configs_from_grid_search(
        config_dict: dict[str, Any], grid_search: list[GridSearchItem]
    ) -> list[dict[str, ConfigValue]]:
        def _get_cartesian_product(grid_search: list[GridSearchItem]) -> list[dict[str, ConfigValue]]:
            # Extract all names, values, and config_path flags
            names = [item.name for item in grid_search]
            value_lists = [item.values for item in grid_search]

            result = []
            for combination in product(*value_lists):
                config = {name: ConfigValue(name=name, value=value) for name, value in zip(names, combination)}
                result.append(config)
            return result

        def _add_config_updates(
            config_dict: dict[str, Any], grid_search_config: dict[str, ConfigValue]
        ) -> dict[str, Any]:
            config_dict_copy = copy.deepcopy(config_dict)
            # for each update
            for path_string, config_value in grid_search_config.items():
                path_list = config_value.name.split(".")
                current_config_dict = config_dict_copy
                # traverse to the object to update
                for key in path_list[:-1]:
                    current_config_dict = current_config_dict[key]
                # update the object
                current_config_dict[path_list[-1]] = config_value.value
            # return adapted config
            return config_dict_copy

        grid_search_configs: list[dict[str, ConfigValue]] = _get_cartesian_product(grid_search=grid_search)

        grid_search_configs_updated = [_add_config_updates(config_dict, gs_config) for gs_config in grid_search_configs]
        return grid_search_configs_updated
