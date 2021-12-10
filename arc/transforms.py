from .util import logger

log = logger.fancy_logger("Transforms", level=30)


code_trait_map = {
    "c": "color",
    "d": "row",
    "r": "col",
}

traits = ["category", "anchor", "color", "size", "big", "width"]


def const_map(group, code):
    """For a transformation group, we attempt a constant"""
    val = None
    for s_idx, seg in group.items():
        for delta in seg:
            if len(delta.transform) > 1:
                log.warning("Multiple transforms per group not supported")
                return None
            tgt = list(delta.transform.values())[0]
            if val is None:
                val = tgt
            elif val != tgt:
                return None
    return (code, None, val)


def t2t_map(group, code):
    # Select a test trait, check until it works
    for trait in traits:
        trial_map = {}
        symm = code_trait_map.get(code) == trait
        log.info(f"Trying trait {trait} symm={symm}")
        failed = False
        # Craft and check trial map
        hit, total = 0, 0
        for s_idx, seg in group.items():
            log.debug(f"{s_idx}: {seg}")
            for delta in seg:
                total += 1
                if len(delta.transform) > 1:
                    log.warning("Multiple transforms per group not supported")
                    return None
                obj = delta.right
                tgt = list(delta.transform.values())[0]
                # Try a constant mapping
                if obj.traits[trait] not in trial_map:
                    trial_map[obj.traits[trait]] = tgt
                    # Add symmetric map if input/output are same trait
                    if symm:
                        trial_map[tgt] = obj.traits[trait]
                elif trial_map[obj.traits[trait]] != tgt:
                    failed = True
                    break
                else:
                    hit += 1
        # Hits ensure we didn't just make up a high variance mapping
        min_hit = total / 3
        log.info(f"{hit} hits against a min of {min_hit}")
        if not failed and hit > min_hit:
            log.info(f"Working map for {trait}: {trial_map}")
            return (code, trait, trial_map)
    return None
