# P3 joint sealed-test opening: authorization record

Required by reports/p3_joint_addendum.md §J8.2. This commit is the only child of the freeze commit and changes
nothing else. `lpw.sealed.check()` verifies that before the test split can be loaded.

Authorized commit: 193d674451565e4b771835897c4533a423b3c24c

Date: 2026-11-08

The user's words, verbatim:

> I authorize the joint sealed-test opening at commit 193d674451565e4b771835897c4533a423b3c24c.

Scope:
- one complete run of `uv run python scripts/p3_joint/run.py --split test` at this commit;
- publication to `runs/p3_joint/2018/test/` by atomic rename;
- no selective peeking, automatic retry, tuning or new analysis (§J8.3);
- a technical failure follows §J8.4.
