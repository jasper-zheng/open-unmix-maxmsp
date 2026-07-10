"""Streaming, export-friendly reconstruction of Open-Unmix for ``neural.live~``.

The stock :class:`openunmix.model.Separator` is un-exportable (complex
``torch.stft``/``istft`` + Wiener EM) and non-causal.  This module rebuilds the
pipeline for real-time block streaming to Max/MSP via the ``neural_tilde``
``LiveModule`` exporter (``torch.export`` -> ExecuTorch ``.pte``):

* **RealSTFT** — a windowed cos/sin DFT expressed as a plain ``nn.Conv1d``
  (kernel ``n_fft``, stride ``n_hop``) fed by ``cached_conv.CachedPadding1d`` so
  cross-block framing state persists as an ExecuTorch mutable buffer.  No FFT /
  complex ops (MLX-safe).
* **RealISTFT** — the inverse DFT as a fixed matmul against a synthesis basis
  ``S`` (onesided scaling + synthesis window + COLA normalisation folded in),
  overlap-added with an :class:`OLATail` streaming cache.  No ``conv_transpose``
  (MLX/MPS-safe).
* **soft mask (niter=0)** with mixture-phase reuse and a ``mask_power`` exponent.
* the three pretrained causal ``OpenUnmix`` cores, wrapped with **persistent
  LSTM state** so frame-by-frame streaming equals the full-sequence forward.

All tensors are channel-major ``[batch, channels, time]``.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_BATCH_SIZE = 64  # mirrors cached_conv.MAX_BATCH_SIZE


# ---------------------------------------------------------------------------
# DFT / IDFT basis construction (int64-modulo phase, float64 -> caller casts)
# ---------------------------------------------------------------------------
def _hann(n_fft: int) -> torch.Tensor:
    """torch.hann_window(n_fft, periodic=True) in float64 (matches Open-Unmix)."""
    return torch.hann_window(n_fft, periodic=True, dtype=torch.float64)


def analysis_weight(n_fft: int, nb_bins: int) -> torch.Tensor:
    """Conv1d weight ``[2*nb_bins, 1, n_fft]`` reproducing onesided ``torch.stft``.

    Row ``k`` (0..nb_bins-1) is the real part ``w[n]*cos(2pi k n/N)``; row
    ``nb_bins+k`` is the imaginary part ``-w[n]*sin(2pi k n/N)``.  Cross-
    correlation (conv1d, no kernel flip) of these with a frame yields exactly
    ``Re/Im`` of ``torch.stft(..., onesided=True, center=False)``.

    Built in float64 with **int64 modulo phase reduction** ``(k*n) % n_fft``;
    a float32 ``arange`` loses ~1e-3 at the top bins.  Returned in float64.
    """
    n = torch.arange(n_fft, dtype=torch.int64)
    k = torch.arange(nb_bins, dtype=torch.int64)
    phase = (k[:, None] * n[None, :]) % n_fft            # [nb_bins, n_fft] int64, exact
    ang = (2.0 * math.pi / n_fft) * phase.to(torch.float64)
    w = _hann(n_fft)                                      # [n_fft]
    cos_w = w[None, :] * torch.cos(ang)                  # [nb_bins, n_fft]
    sin_w = -(w[None, :] * torch.sin(ang))
    weight = torch.cat([cos_w, sin_w], 0).unsqueeze(1)   # [2*nb_bins, 1, n_fft]
    return weight


def synthesis_basis(n_fft: int, nb_bins: int, n_hop: int) -> torch.Tensor:
    """Matmul basis ``S`` ``[2*nb_bins, n_fft]`` reproducing onesided ``torch.istft``.

    ``frames = spectrum @ S`` where ``spectrum`` is ``[..., F, 2*nb_bins]`` (real
    rows then imag rows) gives the windowed synthesis frames; overlap-adding
    them reproduces ``torch.istft(..., center=False)`` in steady state.  The
    onesided doubling (DC/Nyquist counted once, interior bins doubled), the
    synthesis window ``w[n]`` and the constant COLA envelope ``1/C`` are all
    folded into ``S``.  Built in float64.
    """
    n = torch.arange(n_fft, dtype=torch.int64)
    k = torch.arange(nb_bins, dtype=torch.int64)
    phase = (k[:, None] * n[None, :]) % n_fft
    ang = (2.0 * math.pi / n_fft) * phase.to(torch.float64)
    w = _hann(n_fft)

    scale = torch.full((nb_bins,), 2.0 / n_fft, dtype=torch.float64)
    scale[0] = 1.0 / n_fft
    scale[nb_bins - 1] = 1.0 / n_fft                     # k = N/2 (Nyquist)

    cos_rows = scale[:, None] * torch.cos(ang) * w[None, :]   # [nb_bins, n_fft]
    sin_rows = -scale[:, None] * torch.sin(ang) * w[None, :]
    S = torch.cat([cos_rows, sin_rows], 0)               # [2*nb_bins, n_fft]

    C = cola_constant(n_fft, n_hop)
    return S / C


def cola_constant(n_fft: int, n_hop: int) -> float:
    """Steady-state value of the squared-window overlap-add envelope Sum_m w[n-mH]^2."""
    w = _hann(n_fft)
    hops = n_fft // n_hop
    env = torch.zeros(n_fft, dtype=torch.float64)
    # overlap-add w^2 over the fully-covered central region
    for m in range(-hops, hops + 1):
        shift = m * n_hop
        for i in range(n_fft):
            j = i + shift
            if 0 <= j < n_fft:
                env[i] += w[j] * w[j]
    return float(env[n_fft // 2])


# ---------------------------------------------------------------------------
# Streaming overlap-add tail cache (mirrors CachedConvTranspose1d tail logic)
# ---------------------------------------------------------------------------
class OLATail(nn.Module):
    """Carry the ``n_fft - n_hop`` sample overlap tail across blocks.

    Input ``raw`` is ``[batch, 1, B + tail]`` (this block's frames already
    overlap-added).  Adds the previous block's tail to the front, emits the
    first ``B`` finalised samples, and stores the new tail.  The ``tail`` buffer
    is created **lazily** on first forward so the exporter's warm-up detects it,
    zeroes it, and serialises a zero initial state.
    """

    def __init__(self, tail: int):
        super().__init__()
        self.initialized = 0
        self.tail = tail

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x: torch.Tensor) -> None:
        _, c, _ = x.shape
        self.register_buffer("state", torch.zeros(MAX_BATCH_SIZE, c, self.tail).to(x))
        self.initialized += 1

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.init_cache(raw)
        b = raw.shape[0]
        raw = raw.clone()
        raw[..., : self.tail] = raw[..., : self.tail] + self.state[:b]
        self.state[:b].copy_(raw[..., -self.tail :])
        return raw[..., : -self.tail]


# ---------------------------------------------------------------------------
# Streaming RealSTFT / RealISTFT
# ---------------------------------------------------------------------------
class RealSTFT(nn.Module):
    """Streaming onesided STFT via CachedPadding1d + strided Conv1d.

    Forward: ``[b, ch, B] -> re, im`` each ``[b, ch, nb_bins, F]`` with
    ``F = B / n_hop``.  Channels are folded into the conv batch.
    """

    def __init__(self, n_fft: int = 4096, n_hop: int = 1024, dtype=torch.float32):
        super().__init__()
        import cached_conv as cc

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.nb_bins = n_fft // 2 + 1
        self.pad = cc.CachedPadding1d(n_fft - n_hop)
        self.conv = nn.Conv1d(1, 2 * self.nb_bins, n_fft, stride=n_hop, bias=False)
        self.conv.weight.data.copy_(analysis_weight(n_fft, self.nb_bins).to(dtype))
        self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, ch, T = x.shape
        xc = x.reshape(b * ch, 1, T)
        xc = self.pad(xc)
        spec = self.conv(xc)                             # [b*ch, 2*nb_bins, F]
        re = spec[:, : self.nb_bins]
        im = spec[:, self.nb_bins :]
        F_ = re.shape[-1]
        re = re.reshape(b, ch, self.nb_bins, F_)
        im = im.reshape(b, ch, self.nb_bins, F_)
        return re, im


def istft_matmul(S: torch.Tensor, re: torch.Tensor, im: torch.Tensor,
                 n_fft: int, n_hop: int, ola: "OLATail") -> torch.Tensor:
    """Matmul-based onesided ISTFT + streaming overlap-add for one signal.

    ``re, im`` each ``[b, ch, nb_bins, F]`` -> ``[b, ch, B]``.  ``S`` is the
    shared synthesis basis; ``ola`` is the per-signal :class:`OLATail` carrying
    that signal's overlap tail across blocks.
    """
    b, ch, nbins, Fr = re.shape
    spec = torch.cat([re, im], dim=2)                    # [b, ch, 2*nb_bins, F]
    spec = spec.reshape(b * ch, 2 * nbins, Fr).transpose(1, 2)  # [b*ch, F, 2*nb_bins]
    frames = spec @ S                                    # [b*ch, F, n_fft]

    L = (Fr - 1) * n_hop + n_fft                         # = B + tail
    raw = frames.new_zeros(b * ch, 1, L)
    for i in range(Fr):
        left = i * n_hop
        right = L - left - n_fft
        raw = raw + F.pad(frames[:, i, :], (left, right)).unsqueeze(1)

    out = ola(raw)                                       # [b*ch, 1, B]
    return out.reshape(b, ch, out.shape[-1])


class RealISTFT(nn.Module):
    """Streaming onesided ISTFT via matmul synthesis + OLATail overlap-add.

    Forward: ``re, im`` each ``[b, ch, nb_bins, F]`` -> ``[b, ch, B]``.
    """

    def __init__(self, n_fft: int = 4096, n_hop: int = 1024, dtype=torch.float32):
        super().__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.nb_bins = n_fft // 2 + 1
        self.tail = n_fft - n_hop
        self.register_buffer("S", synthesis_basis(n_fft, self.nb_bins, n_hop).to(dtype))
        self.ola = OLATail(self.tail)

    def forward(self, re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
        return istft_matmul(self.S, re, im, self.n_fft, self.n_hop, self.ola)


# ---------------------------------------------------------------------------
# Persistent LSTM state + streaming core forward
# ---------------------------------------------------------------------------
class LSTMState(nn.Module):
    """Persistent ``(h, c)`` hidden/cell state for one core's LSTM.

    Registered **lazily** (mirroring ``CachedPadding1d.init_cache``) so the
    exporter's zero-input warm-up detects the buffers as newly-created, zeroes
    them, and serialises a zero initial state.  Registering them eagerly in
    ``__init__`` would let the warm-up's non-zero ``hn/cn`` (LSTM biases) bake a
    startup transient into the ``.pte``.
    """

    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.initialized = 0
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, ref: torch.Tensor) -> None:
        shape = (self.num_layers, MAX_BATCH_SIZE, self.hidden_size)
        self.register_buffer("h", torch.zeros(shape).to(ref))
        self.register_buffer("c", torch.zeros(shape).to(ref))
        self.initialized += 1

    def read(self, ref: torch.Tensor, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self.init_cache(ref)
        return self.h[:, :b].contiguous(), self.c[:, :b].contiguous()

    def write(self, b: int, hn: torch.Tensor, cn: torch.Tensor) -> None:
        self.h[:, :b].copy_(hn)
        self.c[:, :b].copy_(cn)


def _bn3d(bn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a ``BatchNorm1d`` to a 2D ``[N, C]`` tensor as 3D ``[N, C, 1]``.

    Numerically identical to ``bn(x)``, but the MLX partitioner only supports 3D
    (BatchNorm1d) / 4D (BatchNorm2d) batch-norm — a 2D input forces a CPU-
    fallback subgraph split, and splitting the graph breaks ExecuTorch mutable-
    buffer (streaming-state) persistence across ``execute()`` on MLX.  Routing
    as 3D keeps batch-norm on-GPU so the whole graph stays one delegate.
    """
    return bn(x.unsqueeze(-1)).squeeze(-1)


def streaming_core_forward(core, X: torch.Tensor, state: LSTMState) -> torch.Tensor:
    """``OpenUnmix.forward`` (model.py:107-166) with the LSTM state threaded.

    Identical maths to the stock core, except ``self.lstm(x)`` (zero state each
    call) becomes ``core.lstm(x, (h0, c0))`` with ``(h0, c0)`` read from / written
    back to ``state`` in-place.  For a unidirectional LSTM this makes frame-by-
    frame streaming exactly equal to the full-sequence forward.

    Args/return: magnitude ``[nb_samples, nb_channels, nb_bins, nb_frames]``.
    """
    x = X.permute(3, 0, 1, 2)                            # (F, S, C, bins)
    nb_frames, nb_samples, nb_channels, _ = x.shape
    mix = x.clone()
    x = x[..., : core.nb_bins]
    x = x + core.input_mean
    x = x * core.input_scale
    x = core.fc1(x.reshape(-1, nb_channels * core.nb_bins))
    x = _bn3d(core.bn1, x)
    x = x.reshape(nb_frames, nb_samples, core.hidden_size)
    x = torch.tanh(x)

    h0, c0 = state.read(x, nb_samples)
    out, (hn, cn) = core.lstm(x, (h0, c0))
    state.write(nb_samples, hn, cn)

    x = torch.cat([x, out], -1)
    x = core.fc2(x.reshape(-1, x.shape[-1]))
    x = _bn3d(core.bn2, x)
    x = F.relu(x)
    x = core.fc3(x)
    x = _bn3d(core.bn3, x)
    x = x.reshape(nb_frames, nb_samples, nb_channels, core.nb_output_bins)
    x = x * core.output_scale
    x = x + core.output_mean
    x = F.relu(x) * mix
    return x.permute(1, 2, 3, 0)


# ---------------------------------------------------------------------------
# StreamingUMX — the neural.live~ LiveModule
# ---------------------------------------------------------------------------
STEMS = ("vocals", "drums", "bass", "other")


def _load_cores(model_dir: str, device: str = "cpu"):
    """Load the 3 pretrained causal OpenUnmix cores in fixed order."""
    from openunmix import utils

    cores = utils.load_target_models(
        targets=["vocals", "drums", "bass"], model_str_or_path=model_dir, device=device)
    mods = [cores["vocals"], cores["drums"], cores["bass"]]
    for m in mods:
        m.eval()
        assert not m.lstm.bidirectional, "core must be unidirectional (causal)"
    return mods


try:  # LiveModule only needed when building/exporting
    from neural_tilde import LiveModule
    _LIVE_BASE = LiveModule
except Exception:  # pragma: no cover
    _LIVE_BASE = nn.Module


class StreamingUMX(_LIVE_BASE):
    """Causal streaming Open-Unmix source separator for ``neural.live~``.

    ``forward(signal, mask_power, vocals_gain, drums_gain, bass_gain, other_gain)``
    with ``signal`` ``[b, 2, B]`` -> ``[b, 8, B]`` (vocals/drums/bass/other, each
    stereo L/R).  All spectral state (STFT framing, LSTM ``h/c``, ISTFT overlap
    tails) persists across blocks via ExecuTorch mutable buffers.
    """

    def __init__(self, cores, n_fft: int = 4096, n_hop: int = 1024,
                 eps: float = 1e-8, dtype=torch.float32):
        super().__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.eps = eps
        self.cores = nn.ModuleList(cores)                # vocals, drums, bass
        self.states = nn.ModuleList(
            [LSTMState(c.lstm.num_layers, c.lstm.hidden_size) for c in cores])
        self.stft = RealSTFT(n_fft, n_hop, dtype=dtype)
        self.register_buffer(
            "S", synthesis_basis(n_fft, n_fft // 2 + 1, n_hop).to(dtype))
        self.olas = nn.ModuleList([OLATail(n_fft - n_hop) for _ in range(4)])

    def forward(self, x, mask_power):
        re, im = self.stft(x)                            # each [b, 2, 2049, F]
        mix_mag = torch.sqrt(re * re + im * im + self.eps)
        p = mask_power.reshape(1, 1, 1, 1)

        est_re, est_im = [], []
        for core, state in zip(self.cores, self.states):
            est_mag = streaming_core_forward(core, mix_mag, state)   # [b, 2, 2049, F]
            gain = (est_mag / (mix_mag + self.eps)).pow(p)
            est_re.append(re * gain)
            est_im.append(im * gain)

        # residual "other" = mixture spectrum - sum of the 3 estimated stems
        est_re.append(re - (est_re[0] + est_re[1] + est_re[2]))
        est_im.append(im - (est_im[0] + est_im[1] + est_im[2]))

        outs = []
        for i in range(4):
            outs.append(istft_matmul(self.S, est_re[i], est_im[i], self.n_fft, self.n_hop,
                                     self.olas[i]))       # [b, 2, B]
        return torch.cat(outs, dim=1)                    # [b, 8, B]

    # ---- registration helpers ----
    def register_all(self):
        """Register the 5 attributes + the ``forward`` method (test_method=False).

        ``test_method=False`` is mandatory: a test forward would create the lazy
        LSTM/OLA/pad caches *before* ``export_to_pte``'s warm-up, defeating the
        re-zero and baking a startup transient into the ``.pte``.
        """
        self.register_attribute(
            "mask_power", 1.0, minimum=0.0, maximum=4.0,
            description="Soft-mask sharpness exponent gain**power (1=Wiener niter0).")
        labels = [f"(signal) {s} {lr}" for s in STEMS for lr in ("L", "R")]
        self.register_method(
            "forward", in_channels=2, in_ratio=1, out_channels=8, out_ratio=1,
            input_labels=["(signal) mix L", "(signal) mix R"],
            output_labels=labels,
            test_method=False,
            inputs=["mask_power"],
        )


def build_streaming_umx(model_dir: str, **kw) -> StreamingUMX:
    """Load cores from ``model_dir`` and assemble a :class:`StreamingUMX`."""
    return StreamingUMX(_load_cores(model_dir), **kw)


# ---------------------------------------------------------------------------
# Self-test: verify basis against torch.stft / torch.istft
# ---------------------------------------------------------------------------
def _test_reconstruction() -> None:
    torch.manual_seed(0)
    n_fft, n_hop = 4096, 1024
    nb_bins = n_fft // 2 + 1
    window = torch.hann_window(n_fft, dtype=torch.float64)

    # ---- (a) analysis basis vs torch.stft(center=False), float64 ----
    B = 8 * n_hop
    sig = torch.randn(1, 2, n_fft - n_hop + B, dtype=torch.float64)  # left-context + block
    aw = analysis_weight(n_fft, nb_bins).to(torch.float64)
    conv_out = F.conv1d(sig.reshape(2, 1, -1), aw, stride=n_hop)     # [2, 2*nb_bins, F]
    my_re = conv_out[:, :nb_bins]
    my_im = conv_out[:, nb_bins:]
    ref = torch.stft(sig.reshape(2, -1), n_fft=n_fft, hop_length=n_hop,
                     window=window, center=False, onesided=True, return_complex=True)
    a_re = (my_re - ref.real).abs().max().item()
    a_im = (my_im - ref.imag).abs().max().item()
    print(f"[STFT] max|Δre|={a_re:.2e} max|Δim|={a_im:.2e}  (f64)")
    assert a_re < 1e-9 and a_im < 1e-9, "analysis basis mismatch"

    # ---- (b) synthesis basis vs windowed irfft, float64 ----
    # torch.istft(center=False) NOLA-fails on Hann's zero endpoints, so validate
    # the inverse-DFT+window basis directly against irfft (C is pinned by the
    # round-trip identity in (d)).
    F_ = ref.shape[-1]
    S = synthesis_basis(n_fft, nb_bins, n_hop).to(torch.float64)
    C = cola_constant(n_fft, n_hop)
    S_raw = S * C                                                    # undo the /C fold
    spec = torch.cat([ref.real, ref.imag], dim=1).transpose(1, 2)    # [2, F, 2*nb_bins]
    frames = spec @ S_raw                                            # [2, F, n_fft]
    irf = torch.fft.irfft(ref, n=n_fft, dim=1).transpose(1, 2)       # [2, F, n_fft]
    ref_frames = irf * window[None, None, :]
    b_err = (frames - ref_frames).abs().max().item()
    print(f"[ISTFT] synthesis basis vs windowed-irfft max|Δ|={b_err:.2e}  (f64)")
    assert b_err < 1e-9, "synthesis basis mismatch"

    # ---- (c) streaming modules: block-by-block == one-big-block, f32 ----
    stft = RealSTFT(n_fft, n_hop)
    istft = RealISTFT(n_fft, n_hop)
    stft.eval(); istft.eval()
    audio = torch.randn(1, 2, 16 * n_hop)
    with torch.no_grad():
        re, im = stft(audio)
        big = istft(re, im)
    # fresh modules, fed in blocks of 4*n_hop
    stft2 = RealSTFT(n_fft, n_hop); istft2 = RealISTFT(n_fft, n_hop)
    stft2.eval(); istft2.eval()
    outs = []
    blk = 4 * n_hop
    with torch.no_grad():
        for s in range(0, audio.shape[-1], blk):
            r, im2 = stft2(audio[..., s : s + blk])
            outs.append(istft2(r, im2))
    stream = torch.cat(outs, dim=-1)
    c_err = (big - stream).abs().max().item()
    print(f"[STREAM] block-by-block vs one-shot max|Δ|={c_err:.2e}  (f32)")
    assert c_err < 1e-4, "streaming discontinuity"

    # ---- (d) round-trip identity (interior), streaming, f32 ----
    tail = n_fft - n_hop
    # reconstruction is the input delayed by `tail` samples
    delayed = audio[..., : stream.shape[-1] - tail]
    rt = stream[..., tail:]
    n = min(delayed.shape[-1], rt.shape[-1])
    sl2 = slice(n_fft, n - n_fft)
    rt_err = (delayed[..., sl2] - rt[..., sl2]).abs().max().item()
    print(f"[RT] round-trip interior max|Δ|={rt_err:.2e}  (f32)")
    assert rt_err < 1e-4, "round-trip not identity"

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    _test_reconstruction()
