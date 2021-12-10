import itertools

from .util import logger
from .util import dictutil

log = logger.fancy_logger("Selector", level=30)


# class Selector:
#     def __init__(self):


def avg_link_distance(group):
    all_segs = [seg for grp_segs in group.values() for seg in grp_segs]
    dist, links = 0, 0
    for obj1, obj2 in itertools.combinations(all_segs, 2):
        dist += obj1 - obj2
        links += 1
    return dist / links


def group_inputs(self, variant=0):
    # initially, we assign each input object to its own group
    groups = {idx: {0: [delta]} for idx, delta in enumerate(self.scenes[0].path)}

    # Now add to each group based on the smallest distance
    for s_idx, scene in enumerate(self.scenes[1:], 1):
        # Get all pairwise distances of new objects to groups
        dists = {}
        for g_idx, grp in groups.items():
            for p_idx, delta in enumerate(scene.path):
                dists[(g_idx, p_idx)] = 0
                for gs_idx, gs_segs in grp.items():
                    dists[(g_idx, p_idx)] += sum([delta - seg for seg in gs_segs])
        # Select the assignment permutation that greedily minimizes distance
        best = 1000000
        choice = None
        for perm in itertools.permutations(groups):
            dist = sum([dists[(idx, ass)] for idx, ass in enumerate(perm)])
            if dist < best:
                best = dist
                choice = perm
        for g_idx, ass in enumerate(choice):  # type: ignore
            groups[g_idx][s_idx] = [scene.path[ass]]

    # Identify the transforms present for each group
    codes = {}
    for g_idx, grp in groups.items():
        trsf = {}
        for s_idx, paths in grp.items():
            trsf.update(paths[0].transform)
        if len(trsf) > 1:
            log.warning(f"Transform mismatch: {g_idx}, {trsf}")
        elif not trsf:
            codes[g_idx] = None
        else:
            codes[g_idx] = list(trsf.keys())[0]

    # Now combine groups with the same transform, except Nulls
    t_groups = {}
    if variant == 1:
        t_map = dictutil.reverse_1toM(codes)
        none_ct = 0
        for code, g_idxs in t_map.items():
            if code is None:
                for g_idx in g_idxs:
                    none_ct += 1
                    t_groups[f"N{none_ct}"] = groups[g_idx]
            else:
                t_groups[code] = {}
                for g_idx in g_idxs:
                    dictutil.merge(t_groups[code], groups[g_idx])
        groups = t_groups
        codes = {code: None if code.startswith("N") else code for code in groups}

    link_avgs = []
    for g_idx, grp in groups.items():
        link_avgs.append(round(avg_link_distance(grp), 2))
        log.info(f"Code: {g_idx}, LinkAvg: {link_avgs[-1]:.2f}")
        for s_idx, segs in grp.items():
            log.info(f"  Scene: {s_idx}, {len(segs)} segments")
            for segment in segs:
                log.info(f"    {segment}")
    return groups, codes, link_avgs


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
    bigness = sorted(group, key=lambda x: x.size, reverse=True)
    obj_rank(bigness, "big")
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


def create_selectors(task):
    selectors = {}
    # Describe the inventories in each scene
    for s_idx, scene in enumerate(task.scenes):
        inputs = scene.input.rep.inventory()
        base_describe(inputs)
        describe(inputs)
        # for obj in inputs:
        #    print(obj)
        # for key, val in obj.traits.items():
        #    print(key, val)

    fails = 0
    for g_idx, group in task.groups.items():
        all_objs = [seg.right for grp_segs in group.values() for seg in grp_segs]
        traits = common_traits(all_objs)
        for s_idx, scene in enumerate(task.scenes):
            inputs = scene.input.rep.inventory()
            outputs = [seg.right for seg in group[s_idx]]
            if sorted(select(inputs, traits)) != sorted(outputs):
                # Give one overall warning, later a summary
                if not fails:
                    log.warning("Mismatch, perhaps more traits needed")
                fails += 1
            else:
                selectors[g_idx] = traits
    if fails:
        log.warning(f"{fails} total scene selection mismatches")
    return selectors, fails
