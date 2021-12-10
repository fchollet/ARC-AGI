def pluck(input_dictionary):
    """Destructures a dictionary (vaguely like ES6 for JS)
    Results are SORTED by key alphabetically! You must sort your receiving vars
    """
    items = sorted(input_dictionary.items())
    return [item[1] for item in items]


def merge(d_base, d_in, loc=None):
    """Adds leaves of nested dict d_in to d_base, keeping d_base where overlapping"""
    loc = loc or []
    for key in d_in:
        if key in d_base:
            if isinstance(d_base[key], dict) and isinstance(d_in[key], dict):
                merge(d_base[key], d_in[key], loc + [str(key)])
            elif isinstance(d_base[key], list) and isinstance(d_in[key], list):
                for item in d_in[key]:
                    d_base[key].append(item)
            elif d_base[key] != d_in[key]:
                d_base[key] = d_in[key]
        else:
            d_base[key] = d_in[key]
    return d_base


def dict_xor(d_left, d_right):
    """Returns any key: val pair not in both, choosing the left value if mismatch"""
    output = d_left.copy()
    for key, val in d_right.items():
        left_val = d_left.get(key)
        if left_val == val:
            output.pop(key)
        elif left_val is None:
            output[key] = val
    return output


def reverse_dict(d_in):
    return {v: k for k, v in d_in.items()}


def reverse_1toM(d_in):
    """Reverses the input dict, returning {v: [k1, ...]} for {k1: v, k2: v, ...}"""
    output = {}
    for key, val in d_in.items():
        output[val] = output.get(val, []) + [key]
    return output
