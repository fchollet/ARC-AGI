from arc.util import logger

log = logger.fancy_logger("Selection", level=30)


def obj_rank(s_list, name):
    for idx, obj in enumerate(s_list):
        symm_idx = idx
        # TODO Seems we sometimes want this and sometimes not?
        # if idx >= len(s_list) / 2:
        #     symm_idx -= len(s_list)
        obj.traits[name] = symm_idx


def base_describe(group):
    # Reinitialize traits for current group to the base traits
    base_traits = ["category", "anchor", "color"]
    for obj in group:
        obj.traits = {attr: getattr(obj, attr, None) for attr in base_traits}


def describe(group):
    size = sorted(group, key=lambda x: x.size)
    obj_rank(size, "size")
    width = sorted(group, key=lambda x: x.shape[1])
    obj_rank(width, "width")


def common_traits(objs):
    traits = set(objs[0].traits.items())
    for obj in objs[1:]:
        traits &= set(obj.traits.items())
    return dict(list(traits))


def select(objs, selector):
    for key, val in selector.items():
        objs = [obj for obj in objs if obj.traits.get(key) == val]
    return objs


def unique_select(inputs, outputs):
    for inp, out in zip(inputs, outputs):
        describe(inp)
        log.debug("Scene:\n  Inputs:")
        for obj in inp:
            log.debug(f"    {obj.__repr__()}")
        log.debug("  Outputs:")
        for obj in out:
            log.debug(f"    {obj.__repr__()}")
    all_outputs = [obj for out in outputs for obj in out]
    traits = common_traits(all_outputs)
    for inp, out in zip(inputs, outputs):
        if sorted(select(inp, traits)) != sorted(out):
            log.warning("Mismatch, perhaps more traits needed")
            return {}
    return traits


def unique_map(inputs, outputs, code):
    code_trait = "color" if code == "c" else ""
    # Select a test trait, check until it works
    for trait in inputs[0][0].traits:
        trial_map = {}
        symm = code_trait == trait
        log.info(f"Trying trait {trait} symm={symm}")
        failed = False
        # Craft and check trial map
        hit = 0
        for inp, out in zip(inputs, outputs):
            for obj, val in zip(inp, out):
                log.info(obj.__repr__())
                if obj.traits[trait] not in trial_map:
                    trial_map[obj.traits[trait]] = val
                    # Add symmetric map if input/output are same trait
                    if symm:
                        trial_map[val] = obj.traits[trait]
                elif trial_map[obj.traits[trait]] != val:
                    failed = True
                    break
                else:
                    hit += 1
        # Hits ensure we didn't just make up a high variance mapping
        min_hit = len(inputs) / 2
        log.info(f"{hit} hits against a min of {min_hit}")
        if not failed and hit > min_hit:
            log.info(f"Working map for {trait}: {trial_map}")
            return (trait, trial_map)
    return (None, None)
