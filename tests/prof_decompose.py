from arc import ARC

from arc.util import logger
from arc.util import profile

log = logger.fancy_logger("ProfDecomp", level=30)


@profile.profile(threshold=0.00, dump_file="arc.prof")
def reduction(arc: ARC):
    log.info(f"Profiling execution on first scene from {arc.N} tasks")
    for idx in arc.selection:
        arc.tasks[idx][0].decompose()


if __name__ == "__main__":
    arc = ARC(N=10)
    reduction(arc)
