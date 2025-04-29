import yaml
import os


class ProjectConfig:
    def __init__(self, config_data, file_path=None, is_root=True):
        if is_root:
            self.file_path = str(file_path) if file_path else None

        for key, value in config_data.items():
            if key == "file_path":
                continue  # don't overwrite or duplicate this manually
            if isinstance(value, dict):
                setattr(self, key, ProjectConfig(value, is_root=False))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

        # Function to check for nested attributes
    def has_nested_attribute(self, attr_chain):
        attrs = attr_chain.split('.')
        obj = self
        for attr in attrs:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                return False
        return True
    # Function to add or update an attribute
    def add_attribute(self, key, value):
        if isinstance(value, dict):
            setattr(self, key, ProjectConfig(value))
        else:
            setattr(self, key, value)

    # Function to add an item to a list
    def add_to_list(self, list_name, item):
        current_list = getattr(self, list_name, None)
        if isinstance(current_list, list):
            current_list.append(item)
        else:
            raise TypeError(f"{list_name} is not a list.")

    # Recursive function to convert the class back to a dictionary (for saving to YAML)
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ProjectConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    # Function to save the configuration back to the YAML file
    def save_config(self):
        if self.file_path is None:
            raise ValueError("No file path specified for saving the configuration.")
        with open(self.file_path, 'w') as file:
            yaml.dump(self.to_dict(), file)

    # Function to dynamically access nested attributes
    def get_dynamic_attr(self, attr_chain, dynamic_var):
        """
        Accesses a nested attribute dynamically where part of the chain is variable.

        Parameters:
        - attr_chain: The attribute chain with a placeholder for the dynamic part (e.g., "run_config.{var}.csv")
        - dynamic_var: The variable part that replaces the placeholder

        Returns:
        - The requested attribute value or raises an error if not found
        """
        # Replace the placeholder {var} with the actual dynamic variable
        attr_chain = attr_chain.replace("{var}", dynamic_var)

        # Split the chain into parts to access attributes step by step
        attrs = attr_chain.split('.')
        obj = self
        for attr in attrs:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                raise AttributeError(f"Attribute '{attr}' not found in the chain '{attr_chain}'.")

        return obj

# Function to load the YAML file and instantiate the class
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config_data = yaml.safe_load(file)
    return ProjectConfig(config_data, file_path=yaml_file)

# def has_nested_attribute(obj, attr_chain):
#     """Check if a nested attribute exists in the form of 'attr1.attr2.attr3'."""
#     attrs = attr_chain.split('.')
#     for attr in attrs:
#         if hasattr(obj, attr):
#             obj = getattr(obj, attr)
#         else:
#             return False
#     return True


def add_var(config, var_type, var_id, var_meta):
    """
    Add or update a variable entry in the config using the structure from the existing template.
    Only modifies the necessary fields in the var_id dictionary.

    Parameters:
    - config: dict loaded from YAML
    - var_type: 'col' or 'target'
    - var_id: key of the variable in config (e.g., 'col_short_var1')
    - var_meta: dict with fields to overwrite in the variable block
    """
    assert var_type in {"col", "target"}, f"Unknown var_type: {var_type}"

    # Use existing template block as base
    var_block = config.get(var_id, {})
    var_block.update({
        "data_var": var_meta.get("data_var", var_block.get("data_var")),
        "unit": var_meta.get("unit", var_block.get("unit")),
        "var": var_meta.get("var", var_block.get("var")),
        "var_label": var_meta.get("var_label", var_block.get("var_label")),
        "var_name": var_meta.get("var_name", var_block.get("var_name")),
    })
    for key in var_meta.keys():
        if key not in var_block:
            var_block[key] = var_meta[key]

    config[var_id] = var_block

    # Maintain var IDs
    var_ids_key = f"{var_type}_var_ids"
    config.setdefault(var_ids_key, [])
    if var_id not in config[var_ids_key]:
        config[var_ids_key].append(var_id)

    # Add to group entry (e.g., 'target')
    group_key = var_type
    config.setdefault(group_key, {
        "alias": var_block["var"],
        "ids": [],
        "long_label": var_block["var_label"],
        "var": var_block["var"]
    })
    group = config[group_key]
    group["alias"] = var_block["var"]
    group["var"] = var_block["var"]
    group["long_label"] = var_block["var_label"]
    if var_id not in group["ids"]:
        group["ids"].append(var_id)

    # Add to vars and surr_vars
    config.setdefault("vars", {})
    config["vars"][var_type] = var_block["var"]
    config.setdefault("surr_vars", [])
    if var_block["var"] not in config["surr_vars"]:
        config["surr_vars"].append(var_block["var"])

def new_config_from_template(
    template_path,
    output_path,
    *,
    project_name,
    data_csv_name,
    delta_t,
    time_unit,
    target_relation,
    vars_to_add  # list of tuples: (var_type, var_id, var_meta)
):
    with open(template_path, "r") as f:
        config = yaml.safe_load(f)

    config["proj_name"] = project_name
    config["raw_data"]["data_csv"] = data_csv_name
    config["raw_data"]["delta_t"] = delta_t
    config["raw_data"]["time_unit"] = time_unit
    config["target_relation"] = target_relation

    for var_type, var_id, var_meta in vars_to_add:
        add_var(config, var_type, var_id, var_meta)

    config["col_var"]=config['col']['var']
    for key in ['target_var1', 'col_var1', 'target_short_var1', 'col_short_var1']:
        if key in config.keys():
            config.pop(key)

    config['surr_vars'] = [var for var in config['surr_vars'] if var not in ['target_var_gen', 'col_var_gen']]
    config["target_var_ids"] = [var_id for var_id in config["target_var_ids"] if var_id != "target_short_var1"]
    config["target"]['ids'] = [var_id for var_id in config["target"]['ids'] if var_id != "target_short_var1"]

    config["col_var_ids"] = [var_id for var_id in config["col_var_ids"] if var_id != "col_short_var1"]
    config["col"]['ids'] = [var_id for var_id in config["col"]['ids'] if var_id != "col_short_var1"]
    config['target_var'] = config['target']['var']
    config['col_var'] = config['col']['var']

    palette = config["pal"]
    new_palette = {}
    for key in palette.keys():
        value = palette[key]
        new_key = key.replace("target_var", config["target"]["var"]).replace("col_var", config["col"]["var"])
        new_palette[new_key] = value
    new_palette[config["target_var_ids"][0]] = palette["target_short_var1"]
    new_palette[config["col_var_ids"][0]] = palette["col_short_var1"]
    new_palette.pop("target_short_var1")
    new_palette.pop("col_short_var1")


    config["pal"] = new_palette
    with open(output_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    return config



# Usage example:
if __name__ == "__main__":
    config = load_config("proj_config.yaml")
    print(config)  # Print the whole configuration as a class

    # Add a new attribute
    config.add_attribute("new_attribute", "new_value")

    # Add a new item to a list (make sure it's a valid list name from the YAML file)
    config.add_to_list("col_var_ids", "new_id")

    # Save the changes back to the file
    config.save_config()
