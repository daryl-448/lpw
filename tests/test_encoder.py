"""Encoder-screen variants (reports/p1_encoder_screen.md): ALiBi mask semantics and independent layer initialisation."""
import torch

from lpw.p1.model import CONFIGS, Model


def _m(name):
    torch.manual_seed(0)
    return Model(CONFIGS[name], 5, 4, torch.zeros(5)).eval()


def test_alibi_mask_is_causal_with_unpenalised_profile_and_linear_recency():
    m = _m("tf_alibi")
    M = m.alibi_mask(2, 6, "cpu", torch.float32)
    assert M.shape == (2 * 4, 6, 6)
    assert torch.isinf(M[:, 2, 3:]).all() and (M[:, 2, :3] > -1e9).all()  # no attention to the future
    assert (M[:, :, 0][~torch.isinf(M[:, :, 0])] == 0).all()               # PROFILE never penalised
    s = m.alibi_slopes
    assert torch.allclose(M[1, 5, 2], -s[1] * 3) and torch.equal(M[0], M[4])  # head h of every batch row shares a bias


def test_alibi_encoder_is_causal_and_the_bias_is_not_discarded():
    m = _m("tf_alibi")
    x = torch.randn(2, 7, 128)
    h = m.enc(x, mask=m.alibi_mask(2, 7, "cpu", x.dtype))
    x2 = x.clone(); x2[:, 4:] = torch.randn(2, 3, 128)
    h2 = m.enc(x2, mask=m.alibi_mask(2, 7, "cpu", x.dtype))
    assert torch.allclose(h[:, :4], h2[:, :4], atol=1e-5)
    m.alibi_slopes.zero_()
    h0 = m.enc(x, mask=m.alibi_mask(2, 7, "cpu", x.dtype))
    assert not torch.allclose(h[:, 3:], h0[:, 3:], atol=1e-4)  # the distance penalty changes the outputs


def test_screen_transformers_have_independently_initialised_layers():
    for n in ("tf_ref", "tf_alibi"):
        L = _m(n).enc.layers
        assert not torch.equal(L[0].linear1.weight, L[1].linear1.weight)
        assert not torch.equal(L[0].self_attn.in_proj_weight, L[1].self_attn.in_proj_weight)
    assert not hasattr(_m("tf_alibi"), "pos")
