import os


def init_machine(platform_name: str, ichor_config: dict) -> str:
    """Loops through the keys of the config file, which contain the
    machine abbreviation/name which is also found in the platform name

    :param platform_name: The platform name given by `platform.node()`
    :param ichor_config: A dictionary of the ichor config file, containing names of machines as keys
    :return: Returns key in config file which is contained in the platform name.
        If the key is not found, then returns `"_default"` which would mean that default settings are used
    """

    if ichor_config:

        override = os.environ.get("ICHOR_MACHINE")
        if override:
            machine = override.strip()
            if machine in ichor_config:
                return machine
            raise ValueError(
                "ICHOR_MACHINE="
                + repr(machine)
                + " is not a top-level key in ~/ichor_config.yaml"
            )

        for k in ichor_config.keys():

            if k in platform_name:

                return k

        if "_default" in ichor_config:
            return "_default"

    return None
