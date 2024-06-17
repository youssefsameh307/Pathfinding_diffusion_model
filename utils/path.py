### PATH MAPPINGS ###

def encoded_path_to_real_path(encoded_path, path_type, start, end, straight_path=None):
    """
    Converts an encoded path to a real path.
    """
    if path_type == "offset_path":
        # Offset path is a path where each point is the offset from the straight path
        if straight_path is None:
            raise ValueError("Path type is offset_path but no straight path provided.")
        return straight_path + encoded_path
    else:
        raise ValueError("Invalid path type: {}".format(path_type))
