import numpy as np
# Example data
dipole = np.array([0.0, 0.0, 1.85])
alpha = np.array([[9.8, 0.0, 0.0],
                [0.0, 9.8, 0.0],
                [0.0, 0.0, 10.2]])

def test_yaml():
    import yaml

    data_to_save = {
        "molecule": "H2O",
        "properties": {
            "dipole": {
                "units": "Debye",
                "shape": list(dipole.shape),
                "data": dipole.tolist()
            },
            "polarizability": {
                "units": "a.u.",
                "shape": list(alpha.shape),
                "data": alpha.tolist()
            }
        }
    }

    # Custom representer to dump all lists inline
    def represent_list_inline(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    yaml.add_representer(list, represent_list_inline)

    with open("molecule_props.yaml", "w") as f:
        yaml.safe_dump(data_to_save, f, sort_keys=False)

def test_json():
    import json

    data_to_save = {
        "molecule": "H2O",
        "properties": {
            "dipole": {"units": "Debye", "data": dipole.tolist()},
            "polarizability": {"units": "a.u.", "data": alpha.tolist()},
        }
    }

    with open("molecule_props.json", "w") as f:
        json.dump(data_to_save, f, indent=2)
    
    
    with open("molecule_props.json") as f:
        data = json.load(f)

    dipole_json = np.array(data["properties"]["dipole"]["data"])
    alpha_json  = np.array(data["properties"]["polarizability"]["data"])
    assert np.allclose(dipole, dipole_json)
    assert np.allclose(alpha, alpha_json)

def test_parse_from_source():
    pass