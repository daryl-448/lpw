import torch

from lpw.p1.plan import N_PLAN_N, PROG_BELOW, PROG_DONE, PROG_ROOM, PROG_UNKNOWN, progress_state


def test_progress_state_marks_the_last_allowed_count_as_done():
    # buckets: 0 = [0,1), 1 = [1,2), 2 = [2,3), 5 = [10,30), 9 = [1000, inf)
    pn = torch.tensor([0, 1, 1, 2, 2, 5, 5, 5, 9, 9, N_PLAN_N])
    n = torch.tensor([0, 0, 1, 1, 2, 9, 10, 29, 999, 10**6, 4])
    want = [PROG_DONE, PROG_BELOW, PROG_DONE, PROG_BELOW, PROG_DONE, PROG_BELOW, PROG_ROOM, PROG_DONE,
            PROG_BELOW, PROG_ROOM, PROG_UNKNOWN]
    assert progress_state(pn, n).tolist() == want


def test_clamp_known_progress_leaves_unknown():
    import torch
    from lpw.p1.plan import N_PROG, PROG_UNKNOWN
    from lpw.p1.recal import PROG_CLAMP, clamp_known_progress
    pd = torch.full((N_PROG, 2), 3 * PROG_CLAMP)
    pd[0] = -3 * PROG_CLAMP
    clamp_known_progress(pd)
    assert (pd[0] == -PROG_CLAMP).all() and (pd[1:PROG_UNKNOWN] == PROG_CLAMP).all()
    assert (pd[PROG_UNKNOWN] == 3 * PROG_CLAMP).all()
