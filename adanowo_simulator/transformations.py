import numpy as np


def array_to_dict(array: np.array, keys: list[str],
                  bounds_for_scaling: dict[str, dict[str, float]] | None = None) -> dict[str, float]:
    """
    This method should be used to convert an action array to a dictionary. This avoids mixups in the order of the
    values. The keys are used to assign the values to the correct keys.
    """

    if array.size != len(keys):
        raise Exception("Length of array and keys are not the same.")
    dictionary = dict()
    for index, key in enumerate(keys):
        if bounds_for_scaling is None:
            dictionary[key] = float(array[index].flatten()[0])
            continue
        upper_bound = bounds_for_scaling[key]["upper"]
        lower_bound = bounds_for_scaling[key]["lower"]
        scaled_value = tanh_scale(array[index], lower_bound, upper_bound)
        dictionary[key] = float(scaled_value.flatten()[0])
    return dictionary


def dict_to_array(dictionary: dict[str, float], keys: list[str],
                  bounds_for_scaling: dict[str, dict[str, float]] | None = None, mode="min_max") -> np.array:
    if len(dictionary) != len(keys):
        raise Exception("Length of dictionary and keys are not the same.")
    array = np.zeros(len(dictionary))
    for index, key in enumerate(keys):
        if bounds_for_scaling is None:
            array[index] = dictionary[key]
            continue
        upper_bound = bounds_for_scaling[key]["upper"]
        lower_bound = bounds_for_scaling[key]["lower"]
        if mode == "min_max":
            scaled_value = min_max_normalization(dictionary[key], lower_bound, upper_bound)
        elif mode == "inverse_tanh":
            scaled_value = inverse_tanh_scale(dictionary[key], lower_bound, upper_bound)
        else:
            raise ValueError("Unknown mode.")
        array[index] = scaled_value
    return array


def tanh_scale(array: np.array, min_val: float, max_val: float) -> np.array:
    """
        Apply a tanh transformation to an array and scale the output to a specified range.
        This ensures that all agent actions are centered around 0 and have the same scale.
        Constrinat violations are prevented by the tanh function.

        This method first applies the hyperbolic tangent (tanh) function to each element
        of the input array, which maps the values to the range (-1, 1). Then, it scales
        these values to a specified range (min_val, max_val).

        Parameters:
        array (np.array): The input array containing the values to be transformed.
        min_val (float): The minimum value of the desired output range.
        max_val (float): The maximum value of the desired output range.

        Returns:
        np.array: An array where each input value has been scaled to the specified range.

        Note:
        The input array should have values within the range that the tanh function can handle.
        Extremely large values may lead to numerical instability due to the properties of tanh.
        """
    # Apply the tanh transformation
    tanh_array = np.tanh(array)
    # Scale to the desired range
    scaled_array = min_val + (0.5 * (tanh_array + 1) * (max_val - min_val))
    return scaled_array


def inverse_tanh_scale(scaled_array: np.array, min_val: float, max_val: float) -> np.array:
    # Invert the scaling to get values in the range of (0, 1)
    normalized_array = (scaled_array - min_val) / (max_val - min_val)
    # Scale from (0, 1) to (-1, 1)
    adjusted_array = 2 * normalized_array - 1
    # Apply the inverse tanh (atanh)
    return np.arctanh(adjusted_array)


def inverse_min_max_normalization(array: np.array, min_val: float, max_val: float) -> np.array:
    # Scale to the desired range
    scaled_array = min_val + (0.5 * (array + 1) * (max_val - min_val))
    return scaled_array


def min_max_normalization(array: np.array, min_val: float, max_val: float) -> np.array:
    # Scale to the desired range
    normalized_array = -1 + 2*(array - min_val) / (max_val - min_val)
    return normalized_array
