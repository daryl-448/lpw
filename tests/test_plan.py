import torch

from lpw.p1.plan import N_PLAN_N, PROG_BELOW, PROG_DONE, PROG_ROOM, PROG_UNKNOWN, progress_state


def test_progress_state_marks_the_last_allowed_count_as_done():
    # buckets: 0 = [0,1), 1 = [1,2), 2 = [2,3), 5 = [10,30), 9 = [1000, inf)
    pn = torch.tensor([0, 1, 1, 2, 2, 5, 5, 5, 9, 9, N_PLAN_N])
    n = torch.tensor([0, 0, 1, 1, 2, 9, 10, 29, 999, 10**6, 4])
    want = [PROG_DONE, PROG_BELOW, PROG_DONE, PROG_BELOW, PROG_DONE, PROG_BELOW, PROG_ROOM, PROG_DONE,
            PROG_BELOW, PROG_ROOM, PROG_UNKNOWN]
    assert progress_state(pn, n).tolist() == want
