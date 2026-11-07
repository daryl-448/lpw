# P3b Design Consult

Date: 2026-11-07. Repository: `/home/daryl/Projects/tx-pred-gen`, HEAD
`9768d52168b11c1aac4b7a8cdc20b5c7cdc1c621`.

## Recommendation

Run **one frozen-stream, learned shared-counterparty attachment experiment**. Reuse the unmodified P3a kct and B
worlds 0-2. Add a train-fitted, conditional group prior, then compare:

- **P3b-0:** a role-specific empirical attachment probability, conditional on a legal pool opportunity, with a
  uniform pointer;
- **P3b-1:** one state-only ridge-logistic attachment gate and one ridge conditional-logit pointer, fitted on real
  train. No backbone states, backbone refit, new streams, learned membership assignment, AGENT, deliveries, taint,
  GNN, or sweep repair.

Use the same groups, local tapes, and attachment random numbers in the two arms. Keep kct primary and B a
sensitivity control; do not choose between them. Distinguish **coupling fidelity**, **incremental structural-signal
transfer**, and **the learned kernel's advantage over the rate/uniform control**. None substitutes for another.

This is a legitimate, bounded learned *identity-attachment* model, not the full coupled policy/engine promised by
the original SPEC. Say that explicitly in the P3b protocol. Do not reopen A2: feedback through a new GROUP token is
not necessary to answer this narrower, useful question, and P3a's known behavioral defects are not P3b defects to
repair. Even an excellent P3b result cannot qualify the unchanged full P3 specification.

## Evidence And Inspection Boundary

I read the brief and all requested files: SPEC sections 6.2, 6.6-6.8, 7.3-7.5; the P3 plan; P3a preregistration
sections 1-16, summary and sweep report; the generator, sweep gate, observation operator, feature extractor, and
P3a common loader. I also inspected the P3a readout report, policy/batching/store code, group construction/splitting,
detector code, and the brief's `facts.py` source. I did **not** execute that script.

No data inspection was necessary to establish the invariance claim. I did not load any parquet/NPZ/checkpoint,
invoke the store, generate, fit, or run a readout. The empirical counts and rates below are the brief's supplied
development observations, not newly verified measurements. The train/val measurements recommended at the end
remain to be done. No tracked file was edited.

Important implementation finding: `scripts/p3a/common.py:16` calls `build_store`, which eagerly reads the entire
agent and event tables (`src/lpw/p1/store.py:134`); `group_ids` also eagerly reads all agent rows. Their downstream
train/val selection does not make these loaders acceptable under this consult's strict sealed-row rule.
P3b development needs a separate, genuinely train/val-filtered access path. Do not run the supplied `facts.py`
unchanged either: it reads the whole agent table before filtering. This is a prospective loader correction, not a
claim that any test-dependent result has been inspected here.

## 1. Is The Key Observation Correct?

**Yes, for a precisely specified restricted model.** The claim is stronger as a pathwise invariance than as an
informal statement about distributions.

### What The Current Policy Actually Uses

- `src/lpw/p1/store.py:19` defines only `REPEAT`, `VENUE`, and `NEW`.
- `src/lpw/p1/batch.py:196` constructs tokens from class, venue, tier, time, amount, balance, and conditioning
  fields. The raw global address is not present. Strictly, the transformer does not even embed the integer partner
  entry itself: local entries organize the external memory and link occurrences to that memory.
- `src/lpw/p1/model.py:239` encodes those tokens. Its repeat pointer uses local counts, recency, amounts, tier,
  and in-context occurrence state (`pointer_logp`, line 375), not a non-venue global address.
- `src/lpw/p1/generate.py:121` gives every NEW entry the same periphery tier, sender-capable flag, and no venue.
  Its amounts and future ledger updates depend on local state, not on a pool ID.
- Global **venue** embeddings are an exception to the broad phrase "no global identities"; P3b must leave them
  unchanged. The invariance claim concerns non-agent, non-venue identities only.

For agent i, replace its local non-venue partner identities by an injective map f_i into global periphery IDs.
Maps belonging to different agents may overlap. Preserve every local entry and its metadata; all repeats of an
entry follow the same map. Then the agent's local counterparty equality relation is unchanged. Every current
token, external-memory statistic, amount, fee, ledger update, and local feature is unchanged.

Consequently LOCAL, BAL, VENUE, CONF and PROF are exactly invariant, including distinct counterparties, repeat
share, top-counterparty share and sender/receiver counts. With the same detector inputs, settings and row order,
the existing local TSTR result is identical, not a second piece of evidence for P3b.

A useful factorization is:

```text
P(local tapes X, groups G, identity attachments Z | population C)
  = P_frozen_policy(X | C)
    P_group(G | roles(C), N)
    product_over_local_first_touches K(Z_e | causal_pool_state_e, local_prefix_e, G).
```

A live engine implementing exactly this factorization can sample X first and attach IDs later. It has the same
law as chronological online attachment with the same filtration, masks and kernel. With a per-event common random
number schedule, its attachment outputs can also be identical. Do **not** promise bitwise equivalence to the
existing batched generator rerun as a priority queue: scheduling changes RNG consumption and possibly floating
point computation. Reusing the saved tapes avoids that issue entirely.

### Qualifications That Matter

1. Injection must be enforced **within each agent**, including earlier attachments made in the same recorded
   second. Choosing a node already known to the agent would merge partner entries and break the claim.
2. Never map a periphery entry to an agent or venue, propagate a different tier/sender mask into the policy,
   retokenize a NEW as GROUP, or aggregate another member's balances into the agent's state.
3. The proof applies only if membership does not alter local sampling beyond already-fixed role/cohort/profile
   conditioning. A membership model using the completed stream introduces a different, future-informed joint law.
4. A gate using target amount, the agent's final event count, future active members, full-window pool degree, or
   future recency is not the proposed causal attachment kernel.
5. Match the training and application definitions of time and same-second availability. "Global time order" by
   itself is insufficient; see the frozen second-batch rule below.
6. A **new GROUP class conditioned on by the backbone**, or a GROUP-dependent amount head, would break this
   factorization. A genuinely upgraded SPEC policy can change subsequent local behavior even without deliveries.

### What Can Be Claimed

The fitted kernel learns **when a locally new counterparty is globally shared, and which shared identity to use**.
That is nontrivial learned cross-agent dependence and a restricted implementation of SPEC 6.6/6.10. It is not merely
a uniform identity-control if its conditional choices are fitted and evaluated against that control.

It cannot establish learned behavioral reaction to sharing, coordinated amounts/times, delivery feedback,
agent-to-agent transfers, fund conservation, transaction-label validity or taint propagation. It also cannot show
that a GROUP-aware transformer improves on the frozen policy. The plan's "refit" should mean refitting this new
attachment component, not quietly reopening kct. If the user instead requires the full original coupled-policy
claim, that needs a separately authorized A2 experiment; it is not required for this P3b.

## 2. Membership And Population

### Reuse The Populations, But Name The Conditional Prior

Groups-first is the SPEC's generative story; it is not a requirement to perform otherwise equivalent computations
in that wall-clock order. Given role totals, drawing group role-slots and then assigning existing agents uniformly
within role is coherent because their cohort/profile/activation draws were independent of membership conditional
on role. It does not require regenerating their streams.

However, **an unspecified "match groups to existing roles" is not automatically the same distribution as the
SPEC's groups-first prior**. In particular the old Bernoulli role-count law differs from group-correlated role
counts. Make this an explicit conditional-population design, not an exact implementation of SPEC 6.7.

I recommend this cheap, completely specified empirical-template prior:

1. From real train only, count group templates `(s, k)`, where s is group size and k is its scam count. Let the
   count be c[s,k] and the train population size N_train.
2. For a saved synthetic population with N agents and fixed role totals N_0, N_1, independently draw
   `M[s,k] ~ Poisson((N / N_train) * c[s,k])` for all templates with s >= 2.
3. Reject the **whole template draw** if these groups require more than N_0 normal or N_1 scam slots. Repeat from
   the same RNG stream, at most 1,000 proposals. Exhaustion is a technical failure, not permission to clip groups.
4. Fill the remaining slots with role-specific singleton groups. Train has singleton support for both roles under
   the supplied facts; verify this before freezing the artifact.
5. Independently permute existing agent IDs within each role and place them into these slots. Use a canonical
   ordering of templates `(s,k)` and copy index, with normal slots then scam slots. No activation sorting, cohort
   sorting, profile sorting, or stream-dependent placement.

This is a **Poissonized empirical multigroup prior with singleton completion, conditional on saved role totals**.
It preserves the train empirical joint in expectation away from feasibility conditioning when prevalence matches;
it is not an exact conditional draw from an iid-group law. Singleton completion and rejection can distort it, and
those distortions must be measured and reported. Its advantage over greedy sequential template fitting is that
late residuals cannot silently truncate a large template or systematically consume one role first.

Assign groups to the original sampled population, before inspecting activity or dropping zero-event agents.
Retain P3a's no-event retention rule and report both sampled and retained group sizes. The same population map is
used for kct/B and P3b-0/P3b-1. The prior, including its role mix, is shared between the two P3b arms, so their
contrast does not attribute prior-induced role signal to a learned gate.

Do not fit a second membership model. Time-proximity assignment is plausible, but it would introduce another
learned mechanism and make failures harder to interpret. Uniform within-role placement intentionally tests how
far a minimal sharing kernel can go without a coactivation prior.

### The Giant Validation Group

Do not delete, split, cap, rescale, transplant into synthetic populations, or tune to the 654-agent val group. It is
part of the held-out population. An empirical train template of size 668 makes that scale possible but does not
guarantee that three val-sized synthetic populations contain such a group. That is population-prior uncertainty,
not a seed-selection opportunity.

Freeze these complementary reports:

- group-weighted and agent-weighted group-size distributions, plus maximum share and
  `H = sum_g (n_g / N)^2` / effective group count `1/H`;
- latent assigned group sizes **and**, separately, communities recovered from each observed graph by the same
  label-blind algorithm; agreement of sampled prior sizes is not evidence that the pointer realizes those groups;
- a leave-largest-real-group-out utility influence analysis with fixed full-graph features/scores;
- an additional graph-removal sensitivity: remove that group, recompute STRUCT, rescore the already-trained
  models, but do not refit or replace the primary result;
- group-bootstrap uncertainty, explicitly conditional on the observed population, plus per-world spread.

A mismatch dominated by this group remains a mismatch. The sensitivities explain its leverage; they do not rescue
the candidate. The maximum retained group is selected by size alone, with numeric group ID as the tie-breaker.

## 3. What P3b-0 Should Be

Use the draft's rate/uniform idea, **but fix its denominator**.

Let E be a first local touch of a non-venue, non-agent counterparty. Let C_e be nodes already in the actor's group
pool, touched by another member in a strictly earlier recorded second, and not yet touched by the actor. If C_e is
empty, attachment is forced NEW. Otherwise P3b-0 attaches with probability:

```text
p0[r] = (# train E with role r, C_e nonempty, and target in C_e)
        / (# train E with role r and C_e nonempty).
```

On attachment, choose uniformly from C_e. Otherwise mint a fresh periphery ID in the group's namespace. Both new
and reused entries retain the original frozen periphery metadata.

The supplied 0.132/0.091 rates divide by **all** eligible first touches, including those with no pool opportunity.
Applying those probabilities only when C_e is nonempty would generally underproduce sharing. Do not use them
unadjusted, and do not "correct" the generated rate to val. Count opportunity availability and empty-pool forcing
separately from attachment propensity.

P3b-0 is more accurately called an **empirical-rate/uniform attachment control**, not literally "no learning": its
prior and rates are estimated from train. It isolates the value of context-sensitive gate/pointer fitting above
that shared empirical scaffold. It is also the structural-identity plumbing control because all local inputs
must remain invariant.

Keep the original P3a non-venue-private identity map as a zero-sharing anchor, without rerunning its local utility.
A global bijective shuffle of IDs changes no topology and is not a useful attachment control. Random ID collisions
are worse: without injection, chronology and namespace checks they merge an agent's partners or invent impossible
sharing. Use small constructed permutation/invariance tests instead of another scientific arm.

## 4. One Learned Candidate

### Choice And Eligibility

P3b-1 uses the same opportunity set C_e, first-touch population and forced-NEW rule as P3b-0. Fit the gate only on
events with C_e nonempty. The label is `target_global_cp in C_e`. A first touch of a counterparty known only to
another group is a negative for this within-group kernel; do not delete those observations.

For gate positives, fit the pointer over **all** C_e, with the observed target as the positive. Do not restrict the
training pool to nodes eventually shared, sample candidates using future degrees, or count future members as
active. A node enters a group pool on its first NEW touch, not only after becoming shared.

### Gate Features, Frozen List

Binary features: actor role, born cohort, and current OUT indicator. Continuous features:

1. `log1p(number of actor events before this event)`;
2. `log1p(number of actor's previously seen non-venue/non-agent partners)`;
3. `log1p(len(C_e))`;
4. `log1p(total distinct nodes in the group pool from earlier seconds)`;
5. `log1p(number of other group members)`;
6. `log1p(number of other members with an event in an earlier second)`;
7. `log1p(max(t - actor_origin, 0) / 86400)`;
8. `(t - W_start) / (W_end - W_start)`.

Include role interactions with the eight standardized continuous features, cohort and OUT. Include one intercept.
No other interactions. Standardize continuous coordinates using fitting-train opportunity rows; zero SD becomes 1.
Use arithmetic means and population SD (`ddof=0`) for both gate and pointer standardizers.
No backbone h, amount, balance, plans, real group ID, actual counterpart ID, role composition of the pool, or future
information. Current event direction/time are available from the frozen skeleton before attachment; target amount
is deliberately excluded.

Use the sweep gate's optimization convention: summed binary cross-entropy plus `10/2 * ||beta||^2`, unpenalized
intercept, deterministic L-BFGS initialized at zero, maximum 500 iterations. Fit on CPU in float64, with learning
rate 1, history size 50, strong-Wolfe line search, gradient tolerance 1e-6 and change tolerance 1e-12. There is no
class weighting, decision
threshold, post-hoc rollout calibration, val early stopping, or hyperparameter search. At application, sample the
Bernoulli probability, not a hard decision.

### Pointer Features, Frozen List

For each candidate v in C_e, use only history from strictly earlier recorded seconds:

1. `log1p(number of distinct group members who touched v)`;
2. `log1p(number of group events touching v)`;
3. `log1p((t - last_group_touch(v)) / 86400)`;
4. indicator that any prior group touch had the same OUT/IN direction as this event;
5. indicator that v's minting touch was OUT.

Fit a shared conditional softmax with these five features and their actor-role interactions, no intercept or ID
effects, minimizing summed choice NLL plus `10/2 * ||theta||^2`. Standardize the three continuous coordinates over
all candidate alternatives in fitting-train positive choice sets, weighting each alternative once; zero SD becomes
1. Use deterministic zero-initialized L-BFGS, maximum 500 iterations. Singleton choice sets are retained and have
zero choice loss. Use full denominators, computed in chunks if necessary; do not substitute an unregistered
negative-sampling approximation.

The synthetic periphery entries remain sender-capable exactly as in the saved streams. No contract typing or
sender-capability model is added or propagated to local tokens. This remains a periphery-identity model, not a
test of realistic non-agent node typing. Report real candidate sender-type heterogeneity, if available from
permitted metadata, as a limitation rather than adding a new typing mechanism after seeing the fit.

### Time And Same-Second Rule

Apply identity attachment at **recorded-second resolution**, using floor(raw synthetic ts). Preserve original
per-agent raw order inside each second; for real data preserve `(ts, block_number, tx_hash, OUT-first)` order.
Process a second as a batch for cross-agent pool state:

- every member's candidates and pointer statistics come only from earlier seconds;
- local entry maps are updated immediately, so own repeats and within-agent injection remain correct;
- cross-agent pool additions and statistics become visible only after completing the entire second;
- multiple same-second group touches cannot newly create a shared identity unless that identity was already
  available from an earlier second;
- ties in a real node's first minting direction use the real stable event order, then focal agent ID.

This honors the brief's strict `other_touch_ts < ts` label and avoids imposing a fabricated inter-agent order on
transactions with the same recorded second. It is a declared coarsened chronological engine, not the claim that
raw subsecond synthetic ordering was observed on-chain.

Real same-second co-first touches can reveal the *same* real node even when neither is a GROUP positive. A kernel
that only mints independent fresh IDs cannot reproduce that co-creation. Measure its frequency on train before
freezing and report it as an unmodeled sharing mechanism. Do not silently turn equal timestamps into GROUP
positives to improve fit. For real replay, identical real pool IDs are naturally coalesced at the end of the batch.

### Why No Backbone Re-Pass

A state-only candidate is sufficient to test learned opportunity-sensitive attachment. It avoids GPU work, hidden
state caching, token-window alignment errors, and dependence on an oracle full-stream plan on real data. h is not
needed for the invariance argument or for a learned group-pointer model. Adding it now spends complexity on an
untested premise that local embeddings contain useful sharing information. A negative state-only result will be
reported as such, not followed by an h candidate in this tranche.

## 5. Real Reference And Graph Identity

**Yes:** for a focal agent set S, use all its permitted E1 incident events in W, above the existing dust floor, and
compute degrees/sharing from S's own incident records only. Do not expand to neighbor agents' streams. This is the
split-conditional analogue `R_W(S)`. Pools and training examples are train-only; val features use val only. After
authorization, test features will use the test view only.

But the phrase "counterparties spanning splits are dropped" needs correction. **Do not drop their transactions or
the counterparty nodes.** A boundary counterparty remains in every view in which it is touched. Its degree in a
view counts only focal agents in that view. A node touched once by train and once by val has degree one in each,
not degree two and not a deleted edge. The brief's 1,059 is an existing audit count, not a reason to query the sealed
membership of those counterparties.

Maintain two explicit identity fields:

- `cp_local`: the old per-agent equality key, retained for invariant local features;
- `cp_global`: a collision-free typed node key for structure. Real keys come from raw `events.cp`, not Store.ent;
  synthetic venues retain one venue key across agents, while non-venues use the attachment map.

`src/lpw/p3/observe.py:41` currently discards real global non-venue identity by converting it to `V + ent`, and the
synthetic operator does the same. The same integer denotes unrelated local partners in different agents. Neither
observed `cp` column can be used as a global graph key. Retain a global-ID sidecar or extend a P3b-only schema;
do not change the frozen P3a operator or reinterpret its existing numeric column globally.

Use one actual address node for a real focal agent that is also another focal agent's counterparty. Deduplicate
raw real agent-agent transactions by `(block_number, tx_hash)` before graph degree/amount/strength calculations;
assert that endpoints, time and value agree across the duplicate incident recordings. Inconsistent duplicates are
a schema failure, not averaged. Synthetic row identity is `(world, policy, part, original agent, local event index)`;
the per-agent OUT and IN records remain intact for local features. Do not merge unrelated synthetic rows merely
because amount/endpoints/timestamp coincide. Synthetic committed rows are distinct transactions. Self-transfers
are counted in event summaries but self-loops are removed from simple-graph topology metrics.

For non-venue sharing features/pools, exclude counterparties marked `cp_is_agent` in the existing event metadata,
not just agents in S. Do not read other splits' agent rows to construct this exclusion. For full graph structure,
outside-S endpoints remain boundary nodes, with no outside-S streams or label information. Report both internal
agent-agent edge fraction (unique transactions with both endpoints in S divided by all unique transactions in the
view) and the existing event-metadata agent-counterparty fraction (focal incident rows marked `cp_is_agent` divided
by all focal incident rows). The synthetic numerator is zero for both.

The original venue/group artifacts were constructed transductively before the split (SPEC 5.4). Reuse and disclose
that historical artifact provenance; do not claim P3b groups are learned from a pristine inductive graph. No new
cross-split counts, IDs, membership lists, role aggregates or clean-subset discovery are permitted in development.
Keep the SPEC's clean-test sensitivity for the one authorized opening, using the frozen/pre-existing clean rule.

## 6. Estimands And Outcome Rules

### Separate The Questions

The scientific headline should be **incremental structural signal**, not another high local+STRUCT AUC dominated
by an unchanged 0.9-ish local detector. Structural fidelity is a companion requirement for claiming a realistic
coupling mechanism, not a substitute for transfer. Report the following three records separately.

### A. Incremental Structural-Signal Transfer (Primary Utility)

Let L be the existing P3a `local` block, i.e. LOCAL + BAL + VENUE. S is the frozen STRUCT block below. All models
use the existing fixed XGB settings, training-source data only. On the same real evaluation agents define:

```text
R       = AUC(TRTR, L+S) - AUC(TRTR, L)
T_j     = mean_w [AUC(TSTR_j,w, L+S) - AUC(TSTR_w, L)]     j in {0,1}
D       = T_1 - 0.8 * R
H       = T_1 - T_0
```

The local TSTR baseline is shared across j by construction. Average scalar AUCs over worlds; never pool their
scores. For kct the following rule is frozen:

- **NOT TESTABLE:** R < 0.020 AUC, or the evaluation set lacks both roles. Report R and T_j but no transfer ratio
  or affirmative structural-signal claim. A small, zero or negative R is not a failure of the generator and cannot
  be changed into a success by reporting that local+STRUCT retains the old excess-AUC ratio.
- Otherwise use `D = T_1 - 0.8 R`: **SUPPORTED** if D >= 0 and its one-sided 95% bootstrap lower bound is >= 0;
  **CONSISTENT** if D >= 0 but that lower bound is < 0; **NOT SUPPORTED** if D < 0.
- A SUPPORTED result additionally requires the one-sided 95% lower bound of R to be > 0. If D qualifies but real
  R is uncertain, downgrade to CONSISTENT. Report a two-sided 95% interval for R, T_1 and D.

The 0.020 real-gain floor is a proposed minimum useful incremental effect, **not** an empirical estimate and not
a threshold calibrated to val. Freeze it before any new label-based val measurement. The 0.8 target extends P3a's
retained-signal convention; it is not implied by the original SPEC for additive gains.

Use 1,000 seed-0 whole-real-evaluation-group bootstrap draws, identical across models, worlds and arms. Rescore
fixed predictions; do not refit or rebuild graph features inside this evaluation bootstrap. It describes
conditional evaluation uncertainty, not uncertainty from resampling graph populations or fitting the generator.
Cross-group shared counterparties also leave residual dependence between the nominal bootstrap groups; the
intervals are not guaranteed population-level confidence statements under arbitrary network dependence.
Require at least 950 valid two-role draws; otherwise inferential outcome is NOT TESTABLE. Keep the full-sample
point estimates. Report per-world values and SD separately; three worlds do not identify world variance.

Use the saved P3a matched pairs for a pre-specified secondary R/T/D/H analysis with the same fitted models and
full-split graph features. Fewer than 200 pairs makes that matched analysis NOT TESTABLE. Do not recompute STRUCT
on the matched subset: that would change the observation context. Report full-feature and STRUCT-only results
alongside the noconf/local-based primary; do not select the most favorable baseline after seeing the increments.

### B. Learning-Specific Advantage Over P3b-0

For H above, report **SUPPORTED** if H > 0 and the one-sided 95% lower bound is > 0, **CONSISTENT** if its point estimate is
> 0 but the lower bound is <= 0, and **NOT SUPPORTED** otherwise. The same valid-bootstrap requirement applies.
Do not impose the R floor on this secondary control contrast: it asks a different question.

If T_1 transfers but H is unsupported, say that coupling/prior signal transfers but the learned conditional kernel
has not been shown to improve utility over empirical-rate attachment. A gate fitted to data is "learned" by
provenance; the benefit of that learning is a separate empirical claim.

Train-only whole-group OOF gate BCE and pointer choice NLL versus the rate/uniform control are a second, proper
scoring check on whether the kernel learned useful predictive structure. Neither an OOF improvement nor H alone
proves realistic graph topology.

### C. Coupling Fidelity (Companion Mechanism Record)

Freeze an eight-item core family, all on comparable observed graph views:

1. per-agent non-venue shared-distinct-counterparty fraction, W1, normal;
2. the same, scam;
3. per-agent fraction of other focal agents reached by non-venue sharing, W1, normal;
4. the same, scam;
5. per-agent fraction of other agents in its non-venue sharing component, W1, normal;
6. the same, scam;
7. `log1p(distinct focal agents per eligible non-venue/non-agent counterparty)`, W1, pooled counterparties,
   including degree-one counterparties;
8. `log1p(recovered community agent count)`, agent-weighted W1, pooled agents. Recover communities label-blind on
   the eligible non-venue/non-agent bipartite incidence graph with sorted insertion and the existing Louvain seed
   `20261104`; count agents, not
   all bipartite vertices. Include isolated focal agents as size-one communities.

For each item m, let d_m be synthetic-to-real distance and f_m the real-halves noise distance. Average both over the
20 fixed matched-size draws and then over the three worlds, separately for kct and B; role-size matching can make
f_m differ by world. **CORE PASS** requires d_m <= 2 f_m for every
item, inclusive. Otherwise **CORE FAIL**. If any item is not estimable, the family is **INCOMPLETE**, not passed.
If f_m = 0, only d_m = 0 passes that item; do not insert an arbitrary epsilon. This is SPEC-style descriptive
thresholding, not a simultaneous 95% equivalence test. Report distances and per-world pass flags, not just a count.

Also report P3b-1 minus P3b-0 distances item by item, with no claim that a small pass-count improvement alone
demonstrates superiority. The group prior is shared; matching its input sizes cannot count as learned-pointer
success. Report the remaining SPEC 7.3 metrics regardless of the core outcome.

### Combined Wording

Use three adjacent verdicts, not one opaque score. "Learned structural signal transfers" requires supported A and
B; "the minimal coupling model meets the pre-registered core fidelity targets" requires C to pass. The strongest
joint wording requires all three. If real STRUCT gain is too small, report **utility NOT TESTABLE**, and still
report kernel predictive quality, control contrasts and structural fidelity. Do not add a GNN or search features
to manufacture a real gain. All development verdicts remain development findings until the joint sealed opening.

## 7. Leakage And Structural Shortcuts

Group-private pools eliminate reuse of literal non-venue addresses across independent synthetic worlds. They do
**not** eliminate all group-identity shortcuts.

1. **Hidden-label membership:** the prior deliberately conditions membership on roles. Components, shared degree
   or group size can therefore proxy role mix. This is legitimate only to the extent that it reproduces real
   observable signal; a detector must never receive the latent group ID, role mix, pool owner, or neighbor labels.
2. **Disconnected namespace topology:** real non-venue counterparties cross Louvain groups; synthetic pools do not.
   Conditional on the hidden grouping, synthetic non-venue components cannot cross it. This can be a large seam
   even with correct within-group rates. Do not split a real cross-group counterparty into per-group copies for
   evaluation. The supplied counts are already evidence that this is a real model limitation.
3. **Generated IDs:** `(group_id, local_node_id)` is fine internally, but no numeric ID, prefix, hash bucket, mint
   index or group ordering enters a feature. Release opaque independently remapped identifiers. Test invariance
   to ID bijections and row permutations.
4. **Fold leakage:** real and synthetic C2ST/OOF folds must both be group-disjoint. The existing P3a C2ST randomly
   folds synthetic agents (`src/lpw/p3/utility.py:104`); that is unsuitable after coupling. Do not change P3a's
   frozen result; implement and document P3b's group-aware evaluation separately.
5. **Transductive context:** S uses the complete unlabeled graph of each evaluation split. This is allowed for
   batch node classification, but it is not online prediction or independence of node observations. No role
   aggregation, label propagation, identity embeddings, or cross-split feature cache is allowed.
6. **Size shift:** train and hold graphs have very different focal N. Normalized STRUCT counts reduce, but do not
   eliminate, sampling-density effects. Diagnose this by train-only label-blind thinning. Do not treat
   normalization as a proof of scale invariance.
7. **Post-hoc future information:** Louvain groups themselves are whole-window structural objects, not known-at-birth
   causal organizations. This is a fitted generative prior, not evidence of causal criminal coordination.
8. **Irreducible missing edges:** synthetic agent-agent fraction is exactly zero. Moreover, the agent-to-nonagent
   simple graph is bipartite, hence has zero triangles/transitivity without agent-agent links. A high seam or
   failed full-graph statistic caused by this omission is expected and must stay reported. Do not replace the
   SPEC graph with an agent-sharing projection to hide it.
9. **Inherited local seam:** local features are unchanged, so the existing balance/profile/fee problems remain.
   Report STRUCT-only and L+STRUCT seams separately. A new combined C2ST may differ because its fit differs, but
   the old local seam has not been repaired by changing IDs.

## 8. Pre-Registration Skeleton With Frozen Choices

The following is one implementable proposal, not a report of completed measurements. The developer should turn it
into a protocol and hash its artifacts before the first P3b val role-utility or C2ST readout. Any feasibility-driven
change to these choices must be disclosed and frozen **before** that readout, not used as an implicit extra candidate.

### P0. Scope And Claims

2018 only; frozen-stream periphery-identity attachment; kct primary, B control; one candidate and one empirical-rate
control. No GROUP input token, behavioral feedback, new event generation, policy fitting, deliveries, AGENT, GNN,
taint, rejected sweep, learned membership, or role-based candidate selection. Record the narrowed deviation from
SPEC 6.7 and the plan's ambiguous word "refit".

### P1. Access And Artifact Provenance

Use a new guarded train/val loader with predicate filtering before collection/materialization and event semi-joins
to permitted focal agent IDs. Reject `split == test` in all development APIs. Do not invoke `build_store`,
`common.store`, `common.group_ids`, the unmodified brief `facts.py`, or scripts that eagerly compute all-split plans.
Retain necessary pre-existing venue/type metadata without loading sealed agent rows. Record every input's schema
and SHA256 and the filtered-row access audit. Precompute any normalizers from train only.

### P2. Populations, Worlds, Arms, Seeds

- Reuse `runs/p3a/2018/world{0,1,2}/pop_{train,hold}.npz` and unmodified `{kct,B}_{train,hold}.parquet` only.
  No worlds 10-12 and no kct_sweep. Local tapes and populations are never overwritten.
- Carry through all original cap and retention checks: cap 10,000; capped share > 0.001 of the original population
  is a safety stop; no-event agents dropped and counted. Report original counts even though the attachment cannot
  alter them.
- Group seed: `4000 + 10*w + p`, where p=0 for train and p=1 for hold. Use separate RNG substreams for template
  proposals and within-role permutations: NumPy SeedSequence entropy `[seed,0]` for templates and `[seed,1,r]`
  for the permutation of role r. The map is shared across policy/attachment arms.
- Attachment seed: `5000 + 10*w + p`, with separate gate and pointer substreams. Consume one gate uniform and one
  pointer uniform at **every** eligible local first touch, even if the pool is empty or the gate says NEW. For
  original agent i, use SeedSequence entropy `[seed,0,i]` for its gate and `[seed,1,i]` for its pointer. Thus its q-th
  eligible first touch receives the same uniforms across P3b-0/P3b-1 and kct/B. The latter is only partial common
  randomness because kct/B streams may differ. Use NumPy default_rng/PCG64 and freeze its environment version.
- Use a canonical global processing order `(recorded second, original agent ID, original local event index)`;
  same-second cross-member state is batch-frozen. Candidate nodes are ordered by deterministic creation order
  `(mint second, minter agent ID, minter local event index)`; no labels enter this ordering.
- Fitting/CV/readout seed 0. Noise-floor seeds 0-19. No seed shopping or replacement.

### P3. Membership Prior

Freeze the train template histogram, Poisson scaling, 1,000-proposal limit, singleton completion, canonical slot
ordering and within-role uniform assignment described in section 2. No val reweighting or behavior-aware mapping.
Retain groups assigned before observing no-event retention. Store proposed counts, rejection count, singleton
remainder and retained group sizes for every world/part. Those reports cannot change the sampled world.

### P4. Eligibility And Causal Replay

Freeze the recorded-second batch rule, local injection, non-agent/non-venue exclusions, empty-pool forced NEW,
namespace separation and unchanged local metadata. Pool histories include every relevant past touch, including
REPEAT events, while identity decisions occur only at first local touches. Keep original tokens unmodified and
log attachment choices as a sidecar with original `(agent, ent)` keys.

### P5. Fitting

Freeze the exact gate and pointer features, penalties, standardizers, optimizer and no-search restrictions in
section 4. Three train-group OOF folds, seed 0, used for diagnostics only; all group histories of an OOF group are
kept together. Control rates and normalizers in each diagnostic fold are fitted only on its fitting groups. Report
BCE, Brier/calibration by role, pointer NLL, uniform-pointer NLL, opportunity counts and choice-set size quantiles.
Then fit once on all train, store parameters, convergence/gradient diagnostics and hashes. Do not accept/reject on
TSTR; predictive weakness is reported without swapping in another model. Non-finite or invalid fits are technical
failures and stop application/readout until a documented bug is resolved within budget.

### P6. Observation And Local Invariance

Keep P3a's local observation/feature code unchanged and add global endpoints separately. Require exact equality of
every nonidentity event field, all local cp equality relations, observed local feature matrices, retained agent
IDs and cap counts across the original and both attachment arms. Local tape hashes must match the originals.
Use one P3b global-graph implementation for real/synthetic endpoints. Deduplicate only the real duplicate incident
recordings of one transaction, never a sender's two genuine same-second transfers. Every group attachment has an
earlier-second witness in the same group and a previously unseen actor entry.

### P7. STRUCT Definition

Let C_i be i's distinct non-venue/non-agent counterparty set; d(c) the number of focal agents touching c in this
split/world; N the number of retained focal agents. Let U_i be the union of other focal agents sharing any c in
C_i, and K_i the number of focal agents in i's connected component of the same sharing graph. Freeze six features:

1. `shared_any = 1[there exists c in C_i with d(c) >= 2]`;
2. `shared_frac = #{c in C_i: d(c) >= 2} / |C_i|`;
3. `mean_cp_other_frac = mean_{c in C_i} (d(c)-1) / max(N-1,1)`;
4. `max_cp_other_frac = max_{c in C_i} (d(c)-1) / max(N-1,1)`;
5. `other_agents_frac = |U_i| / max(N-1,1)`;
6. `sharing_component_frac = (K_i-1) / max(N-1,1)`.

All are zero if the relevant set is empty; an isolated agent has K_i=1. Include every focal agent, not just agents
with a non-venue edge. Compute components on the bipartite graph without materializing huge coagent cliques. No
IDs, group labels, community membership IDs, pool flags or roles enter these features. Candidate aggregate support
can be estimated on train, but this feature list is not selected using role AUCs. Full-split graph context is used
before selecting matched agents or evaluating fixed scores.

### P8. Structure Metrics And A Size-Correct Noise Floor

Do not compare a full-N synthetic graph to a half-N real graph and then call their distance a SPEC noise-floor
comparison. Graph structure changes when focal agents are removed, and the giant group amplifies that problem.

For the **core comparison**, define:

```text
n = min(floor(N_real_eval / 2), min_w floor(N_synthetic_hold_w / 2))
```

Use one common n across all four policy/attachment combinations and worlds. For each seed 0-19, take two disjoint
uniform label-blind n-agent subsets A/B of the real evaluation split and an independent n-agent subset of each
synthetic holdout. Keep the complete incident streams of selected focal agents; do not expand other agents' streams.
Rebuild all three graph views, degrees and recovered communities independently. Compute d_m between synthetic/A
and f_m between A/B. For role-conditioned W1 metrics, subsample the already-computed agent measurements in each
view to the minimum available role count in that triple, using the same seed and saved IDs; require at least 20
agents per role. Pool-counterparty metrics include degree-one nodes. Undefined items are incomplete, not passes.

These are repeated disjoint half-view comparisons, not 20 independent populations or conventional iid graph
bootstraps. They provide a finite-context descriptive noise scale. Also report full-view metrics without assigning
them a half-view noise-floor pass. This distinction is a disclosed operationalization of SPEC's ambiguous
"sample sizes matched" instruction, not a claim of independent full-size graph replication.

For SPEC 7.3 daily topology, use actual undirected transaction graphs with agents and boundary counterparty nodes,
not the coagent-sharing projection. Report both venue-included and venue-excluded variants, primary venue-included.
Within each A/B/synthetic triple and UTC day, let m_d be the minimum daily active focal-agent count. Uniformly select
m_d active focal agents separately on each side, retain all their incident transactions that day, then rebuild the
graphs. Reuse the selection draws across attachment arms; active-agent sets are invariant. Record all m_d and
excluded days. Empty graphs are undefined; for assortativity, constant-degree endpoint variance is undefined, not
zero. Define max-core fraction as vertices in the highest nonempty k-core divided by all vertices. Report GCC
ratio, component count, max k-core, max-core fraction, assortativity and transitivity, with 20-draw mean absolute
paired daily discrepancies and the corresponding A/B scale on days defined for all sides. No daily metric passes
if fewer than 80% of nonempty common days are defined.

For Clauset diagnostics, use full-window graph unique degree, transaction-count degree, strength (sum ETH, not
fees), and per-transaction ETH amount. Count each real transaction once in graph statistics. Use discrete fits for
the two degrees and continuous fits for strength/amount; standard Clauset xmin selection by minimum KS, with
lognormal likelihood ratio and 100 parametric-bootstrap goodness-of-fit refits. Require >=50 tail observations.
No alternate tail, cutoff, distribution or power-law family is selected after seeing results. Freeze the standard
fitting library/version and implementation before execution. Undefined fits are reported, not passed. These fits
and daily topology are required descriptive SPEC reports, not extra success opportunities for the eight-item core.
Report agent-role/cohort/train-defined-age-tercile marginals where the statistic applies to focal agents; do not
assign agent labels to arbitrary boundary nodes.

Also report assigned and recovered group sizes, group purity only as a hidden/real diagnostic, agent-agent event
fractions with stated denominators, cross-group shared-node incidence and same-second co-creation omission.
The old 2021 0.40/0.68 statistic is not an acceptance target for these 2018 distinct-counterparty definitions.

### P9. Utility Models And Inference

Use the existing LR/RF/XGB/LGBM settings and seed 0 from `lpw.p3.utility.make`, no tuning. XGB alone is primary;
others are sensitivity estimates. Sources are real train or one attached synthetic train-size world, not pooled
worlds. Feature sets: L, L+S, S-only, full, full+S, and confonly. Reuse frozen local/full/confonly predictions where
exact inputs and evaluation agents agree. Report AUC/AP throughout and F1 for primary-detector full+S, with a
training-source-only five-group-fold OOF threshold; use the P3a highest-threshold tie rule, but group both real and
synthetic training agents in the new P3b F1 calibration. Do not alter P3a's already-frozen thresholding protocol.

Apply section 6's exact R/T/D/H rules to L-based XGB results. STRUCT-only excess-AUC retention uses the old 0.60
TRTR testability floor and 0.8 contrast, as a separately labeled secondary result. Full/full+S, AP, matching and
other detectors do not change the primary verdict. The matched conditional-baseline comparison stays full+S minus
confonly, with the original saved propensity support/caliper and pair IDs.

### P10. Seam, Shortcut And Inherited Criticism Results

Per-role XGB C2ST: balanced source sizes by seed-0 subsampling, five group-disjoint folds on **both** real and
synthetic sides. Preserve every selected group within one fold. Assign each source's groups largest-first to its
least-populated fold, ties resolved by a seed-0 permutation; store assignments. If five valid two-domain folds
cannot be made, mark C2ST NOT TESTABLE rather than split groups. No exact per-fold size constraint may break a
group. Within each training fold, give each domain total sample weight equal to half that training fold's row
count, distributing it uniformly within domain; this prevents giant-group fold imbalances from changing the
training prior. Score the pooled OOF probabilities as in P3a. STRUCT-only is the new structural seam; L+S is the
combined seam; full+S is secondary. Each uses the existing
thresholds: both role AUCs <=0.70 and scam minus normal <=0.05. Report fixed single-feature structural seams to
explain failures, not to remove features.

Shortcut: rank the top 20 features of full+S by absolute signed role AUC distance from 0.5 on **real train**; ties
use feature order. Add all six STRUCT features and the original five balance sentinels regardless of rank. Compare
synthetic-train versus real-evaluation signed AUC, flag absolute difference >0.10. Use the mean delta over worlds,
per-world values retained. Group IDs/roles are never predictors. Keep P3a's failed local seam/shortcut results
visible; a change in new model fit does not retroactively repair them.

Report difficulty and profile-gain parity on full+S with the existing P3a definitions/thresholds and independent
hold populations. Do not refit the encoder: its inputs are locally invariant, so its existing coverage is unchanged
by construction. Do not claim the old tabular/encoder DCR or coverage analysis now validates graph memorization
or structural mode coverage; neither sees the new graph. No new embedding search or graph coverage gate is added.

### P11. Checkpoint 0 And Failure Policy

Before any utility/C2ST output, apply P3b-0 then P3b-1 to world 0 train/hold for kct/B and inspect only:

- hashes/schema/caps/no-event counts; local invariance; namespaces; group-slot totals and proposal counts;
- empty-pool and eligible opportunity counts, attachment counts and shares by role/direction, pointer choice-set
  sizes, within-agent collision count (must be zero), and same-seed reproducibility;
- strict earlier-second witnesses, no future pool access, no shared non-venue IDs between independent worlds or
  groups, and venue stability;
- normalized STRUCT ranges, finite values, expected zero-sharing results on hand-built uncoupled examples, and
  exact expected sharing/component values on small hand-built coupled examples.

Safety/schema/injection/causality failures stop the readout. A demonstrated bug may be fixed, logged and rerun with
the same seeds within budget; it is not permission to alter statistical choices. Weak attachment rates, poor
train OOF quality, large giant-group influence or poor graph fidelity do not trigger a candidate search. Write one
mechanistic diagnosis, keep the protocol, and complete the fixed development comparison unless there is a genuine
validity/budget stop. Never reject on mechanism and then promote on TSTR.

### P12. Development Readout, Freezing And Sealed Opening

Write all train-fit diagnostics and Checkpoint 0 decisions before exposing new val role utility or C2ST. Complete
worlds 1-2 without seed shopping. Run the frozen development readout once, with all negative outcomes retained.
Record any deviations next to the original result. Hash code, environment, input tapes/populations, group maps,
kernel parameters, feature order, O rules, raw predictions, matching artifact, noise-floor samples/folds and
bootstrap recipes. No new train fit or model selection is justified by a val utility improvement.

Then freeze the **joint** P3a/P3b test addendum and obtain the user's authorization. It must list:

- P3a kct primary/B control, worlds 0-2, unchanged original features/thresholds, no rejected sweep arm;
- both P3b attachment arms, including a development failure if one occurred; no choosing a gate after val;
- unchanged train-fitted group prior, gate, pointer, metadata conventions, encoder and population models;
- the same saved train/hold local tapes and attached worlds, with no tuning to test graph sizes, role counts or
  labels; matched-size structural subsampling follows the already-frozen deterministic formula;
- train-real -> test-real and synthetic-train -> test-real predictions; test-own-graph STRUCT; frozen matching
  and train-defined age bins; saved clean-subset sensitivity where required by SPEC;
- all utility/fidelity/seam/shortcut definitions, undefined-value rules, valid-bootstrap minimum and stop policy;
- automatic production of the complete report in one authorized evaluation job, no selective peek/retry, and no
  changes afterward other than clearly separated bug corrections with the original result preserved.

Both positive and negative development conclusions are confirmed. If any exploratory protocol change is made
before opening, distinguish it from the original frozen candidate and do not enlarge the one-candidate package
without a user decision.

## 9. Budget, Build Order And Reviews

This consult runs nothing. The following is a recommended execution order for the implementing agent:

1. Approve scope wording and freeze the scientific choices before development utility. Create an access audit and
   a minimal filtered loader first. Review it specifically for hidden eager all-split reads.
2. Take the permitted pre-freeze measurements below. Write feasibility facts, not additional candidate options.
3. Freeze the preregistration, candidate count, seeds and budgets. Implement a P3b-only identity/global-graph
   sidecar, conditional group sampler and P3b-0; add small fabricated-data tests.
4. Review chronology, local injection, global node namespaces, transaction deduplication and graph size matching
   before fitting anything. A design/code review need not run any real readout.
5. Implement the single state gate/pointer, shared feature functions and train-only diagnostic folds. Fit once
   under the frozen recipe, hash artifacts, then review label definitions and choice-set denominators.
6. Checkpoint 0 as above. Its hard invariants gate further application, not utility. Document weak mechanism once.
7. Attach worlds 1-2. Review the readout code/fold isolation and outcome math before exposing outputs. Run one
   complete val readout and write the full negative/positive ledger.
8. Freeze the joint sealed-test addendum; obtain explicit authorization; open once and report P3a/P3b together.

Proposed envelope: at most two build days, zero new generation/GPU hours, <=1 CPU-hour for sampler/attachment and
fit/CV, and <=3 CPU-hours for the complete development readout including graph diagnostics. These are **caps to
validate with measured throughput**, not timing claims. Benchmark candidate enumeration, chunked softmax passes,
daily graph construction, Louvain and tail refits on train only. If the prescribed fit or metrics will not fit the
budget, stop and ask; do not silently reduce graphs, fit denominators, refits, worlds or detectors. Any single
experiment projected over four hours requires the user's decision under SPEC 9. Technical/budget failure is
INCOMPLETE, not a reason to open test or try a smaller unregistered candidate.

No subagents are needed for this consult. The reviews above are implementation checkpoints, not instructions to
launch agents in this read-only task.

## 10. Permitted Measurements Before Freezing

These should resolve feasibility and quantify model limitations **without** a new val label-based utility readout.
The supplied val role summaries have already influenced this design; disclose that development exposure. Do not
pretend the final val evaluation is fresh confirmation.

### Train-Only, Roles Allowed

- Exact empirical `(size, n_scam)` histogram, singleton support by role, giant-template role counts, sampled-prior
  rejection probability and expected singleton remainder at N=17,454 and N=3,741. Use population-only sampler
  simulations or exact prior calculations, not event generation. Quantify distortion from role conditioning.
- First-touch counts, C_e-empty shares, unconditional GROUP rates **and** conditional opportunity rates p0[r],
  by role/direction/cohort/group-size bin. Freeze numerical p0 artifacts, not the brief's rounded rates.
- Positive target coverage, full choice-set size quantiles/max, total candidate alternatives, and whether
  equal-second co-first touches account for meaningful observed sharing. Verify REPEAT precedence on examples.
- Within-group versus cross-group sharing by counterparty, incidence, transaction and connected-component
  contribution. This bounds what an exclusive-group pool can represent; simply quoting a counterparty count
  does not measure topological importance.
- Coactivation dispersion within groups versus a uniform-within-role membership null. This identifies, without
  adding another candidate, whether random membership will destroy temporal opportunity.
- Optional three-group-fold OOF real role utility using the **already fixed** six S features and L/L+S, train only,
  to assess whether incremental signal is plausibly measurable. Use fold-own graph contexts and make no feature,
  threshold or model selection from it. This is optional and must itself be pre-specified before running; no such
  fit was performed for this consult. A low train gain is a warning that utility may be NOT TESTABLE, not a reason
  to search a feature/GNN menu. Gate/pointer OOF diagnostics occur after their recipe is frozen, not as a model sweep.

### Label-Blind Train/Val Structure Only

- Group-size CDFs weighted by group and agent, maximum group share, H/effective group count, daily active focal
  counts, and distribution of events/partners per focal agent. Use no val role join for these design diagnostics.
- Train induced-view thinning to 3,741 focal agents and to half-view sizes: changes in all six STRUCT coordinates,
  recovered group sizes and daily topology. Compare train halves at matched size to check that the noise-floor
  implementation does not accidentally reuse full-graph degrees or globally derived sharing labels.
- Frequency of undefined assortativity, empty days, zero noise floors, too-small power-law tails, and huge
  coagent components. Freeze undefined-value handling now rather than after a favorable/unfavorable result.
- Global-ID schema, uniqueness, venue keys, `cp_is_agent` provenance, and raw transaction recording identity.
  Verify deduplication on constructed examples and permitted focal rows only. Do not construct cross-split/test
  counterpart membership to check the already-published 1,059 audit count.
- Work estimates: candidates evaluated per fit pass, peak RAM for the largest group, graph/recovery time and
  tail-bootstrap costs. This study should not need a teacher-forced transformer pass or a GPU.

Do **not** compute val structural role AUCs, incremental TRTR gains, val label-informed feature selection,
candidate C2ST or val TSTR before freezing the recipe. Do not load or compute on sealed test agent rows at all.

## Bottom Line

The brief's invariance is correct under the stated masks and identity mapping. Exploit it rather than refitting a
policy whose search is intentionally stopped. P3b should test a **single causal learned attachment kernel** above
an honest rate/uniform control, on the same local tapes and conditional group prior. Its potential contribution is
observable shared-counterparty structure and transferable *incremental* structural signal. Its unavoidable limits
are no behavioral feedback, no direct-agent topology, exclusive-group pools, and substantial finite-population
uncertainty from a giant held-out group. Freeze those limits and the NOT TESTABLE outcome now, then let the one
joint authorized test opening decide the confirmatory claims.
