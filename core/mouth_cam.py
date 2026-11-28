"""
Mouth close-up for the HUD — the same head, framed on the part that actually
does the lip-sync.

Why a lens and not a zoom
-------------------------
`HoloAvatar` sizes the head to a HUD band roughly 300 px tall, which puts the
mouth about 20 px wide. That is enough to show *that* something is being said
and nowhere near enough to see *what*: at that size a viseme is two pixels of
jaw travel, and /i/ and /u/ land on the same spot. This module points a much
tighter camera at the same mesh — about a 20× magnification — so a spread vowel,
a rounded vowel and a bilabial closure become three separate, legible events.

It is deliberately not just a scaled copy of the wide shot:

* **The opening is real geometry.** Rotating the jaw alone drops the lower lip
  about 0.09 head-half-heights, which at this magnification is a fifth of the
  frame — but the inner lip ring is a *closed silhouette*, so a jaw drop by
  itself reads as the whole mouth sliding down the face rather than opening.
  The two lips are therefore separated along their own measured weights
  (`mesh["lips"]` split either side of the lip seam), which is what turns one
  oval into an aperture.

* **The face is drawn as a dot lattice, not a surface.** A lit skin surface at
  20× is a wall of triangles with a mouth somewhere on it. Sampling the mesh
  into points instead — and clearing a ring around the lips — leaves the mouth
  as the only *dense* thing in the frame, so the eye goes there by itself. It
  also costs a fraction of the fills, so this renderer is cheaper than the wide
  shot despite showing far more.

* **The energy is the voice, not decoration.** Filaments spring from the lip
  seam and particles are shed by it; both are driven by the same level, opening
  and lip width the wide shot uses, and both fall silent the moment the voice
  does. Nothing in this frame moves that the audio did not cause.

The camera is a separate object; the **head is not**. This wraps the caller's
`HoloAvatar` and reads its pose and state each frame, so the close-up and the
wide shot can never disagree about what the mouth is doing — switching between
them is a cut between two cameras on one set, not two different faces.
"""

from __future__ import annotations

import math
import random

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (QBrush, QColor, QPainter, QPainterPath, QPen,
                         QPixmap, QPolygonF, QRadialGradient)

# `_c` and `_blend` are the avatar's colour helpers. Sharing them is the point:
# a colour that is "35% of primary over the background" has to mean the same
# thing on both cameras or the two views stop looking like one machine.
from core.avatar import HoloAvatar, _blend, _c
from core.avatar_mesh import get_head_mesh

# Camera distance, in head-half-heights. Further out than the wide shot's 4.6
# on purpose: a long lens is what stops the nose and the cheeks ballooning when
# you fill the frame with a mouth, and a close-up on a wide lens reads as a
# fisheye caricature.
_CAM_D = 7.6

# Half of the visible world height, in head-half-heights. The mouth is 0.44 wide
# and 0.18 tall, so framing 0.88 of world height across the aperture puts the
# resting mouth across ~53% of the diameter — big enough to read the articulation,
# with room left around it for the filaments that stream off the corners.
_VIEW_H = 0.44

# How far the lips separate at full aperture, in head-half-heights. Measured
# against what reads at this size: the jaw rig contributes ~0.09 by itself, the
# lower lip adds this much and the upper lip a quarter of it, which puts a fully
# open vowel at ~23% of the frame height — a real jaw drop, not a pout.
_AP_LOW = 0.046
_AP_UP = 0.013
_AP_BACK = 0.010     # the lower lip retreats into the mouth as it drops

# Dots. Sizes are fractions of the aperture radius so the view is identical at
# 120 px and at 400 px. The lattice radius is deliberately small and its glow
# multiplier modest: `drawPoints` with a wide round-capped pen is the most
# expensive thing in this file, and a 50 px pen over 600 points costs more than
# every other layer put together (measured: 11.5 ms of a 14 ms lattice pass).
_DOT_R = 0.0095      # skin lattice
_LIP_DOT = 0.016     # lip ring dots, the brightest thing in the frame

# Every other lattice point is kept. The field still reads as a field at half
# density — the dots are 2–3 px apart either way — and it halves the two
# drawPoints calls that dominate the frame.
_DOT_STRIDE = 2

_ALPHA_BUCKETS = 4   # batched drawPoints() calls for the skin lattice

_N_FIL = 20          # energy filaments around the lip seam
_MAX_PARTICLES = 170

# Sampling budget. `_N_OUT` / `_N_IN` interpolated dots hold the lip outline
# together between the 20 hand-placed landmarks — at 20×, twenty dots is a
# constellation, sixty is a line.
_N_OUT = 60          # dots around the outer lip ring
_N_IN = 48           # dots around the inner lip ring


def _ring_samples(ring: np.ndarray, pts: np.ndarray, count: int):
    """Evenly spaced samples around a closed landmark ring.

    Spacing is by *arc length in rest space*, not one sample per landmark
    segment. MediaPipe's lip rings are not evenly spaced — the segments near the
    corners are several times longer than the ones across the front — so
    sampling per segment leaves the corner dots strung out and the front ones
    piled up, which at 20× reads as a patchy, broken outline rather than a
    dotted one.

    Solved once here rather than per frame: the deformation is smooth, so a ring
    that is evenly spaced on the resting face stays near-even when posed.
    """
    closed = np.concatenate([ring, ring[:1]])
    seg = np.hypot(np.diff(pts[closed, 0]), np.diff(pts[closed, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])

    ai, bi, tt = [], [], []
    for k in range(count):
        s = total * k / count
        j = int(np.searchsorted(cum, s, side="right") - 1)
        j = max(0, min(len(ring) - 1, j))
        span = seg[j] if seg[j] > 1e-9 else 1.0
        ai.append(int(closed[j]))
        bi.append(int(closed[j + 1]))
        tt.append((s - cum[j]) / span)
    return (np.array(ai, dtype=np.int32),
            np.array(bi, dtype=np.int32),
            np.array(tt, dtype=np.float32))


class MouthCam:
    """Close-up renderer for the HUD's lip-sync view.

    Lifecycle mirrors `HoloAvatar`, because it *drives* one:

        cam = MouthCam(hud.avatar)
        cam.step(dt, amp, speaking=..., muted=..., ...)   # once per tick
        cam.paint(painter, cx, cy, r, primary, accent, bg)

    `step()` takes exactly the arguments `HoloAvatar.step()` takes and forwards
    them, so the HUD can swap one call for the other and change nothing else.
    """

    def __init__(self, avatar: "HoloAvatar | None" = None) -> None:
        self._av = avatar if avatar is not None else HoloAvatar()
        mesh = get_head_mesh()
        self._v0 = mesh["verts"]
        self._n0 = mesh["normals"]
        self._fade = mesh["fade"]
        self._lips_w = mesh["lips"]
        self._lip_c0 = mesh["lip_centre"].astype(np.float64)

        lm = mesh["landmarks"]
        self._lip_out = lm["lips_out"]
        self._lip_in = lm["lips_in"]
        # Same split the wide shot uses: the inner ring runs lower-lip
        # left→right, then upper-lip back, so the upper arc can anchor teeth.
        # The teeth arc is then trimmed by two landmarks at each end: right at
        # the corners the ring converges to a point, and a strip that reaches
        # there collapses into a bright blob in the corner of the mouth.
        self._lip_up = np.concatenate([lm["lips_in"][10:], lm["lips_in"][:1]])[2:-2]

        self._sn_out = _ring_samples(self._lip_out, self._v0, _N_OUT)
        self._sn_in = _ring_samples(self._lip_in, self._v0, _N_IN)

        # ── aperture weights ────────────────────────────────────────────────
        # `mesh["lips"]` is a gaussian around the lip centre that already fades
        # the limits of the rigged region (temples, the back of the jaw) and is
        # zero on the neck. Splitting it either side of the seam turns it into
        # an *opening*: the upper lip rises, the lower one drops.
        seam = float(self._v0[self._lip_in].mean(0)[1])
        y = self._v0[:, 1]
        below = np.clip((seam - y) / 0.055, 0.0, 1.0)
        above = np.clip((y - seam) / 0.045, 0.0, 1.0)
        self._w_lo = np.ascontiguousarray(self._lips_w * below, dtype=np.float32)
        self._w_up = np.ascontiguousarray(self._lips_w * above, dtype=np.float32)
        self._seam = seam

        # ── skin dot lattice ────────────────────────────────────────────────
        # Vertices plus edge midpoints: the mids smooth the lattice out to an
        # even point field, where vertices alone leave the head reading as a
        # scatter following the model's own uneven tessellation.
        edges = mesh["edges"]
        self._e0 = edges[:, 0]
        self._e1 = edges[:, 1]
        pts = np.vstack([self._v0, 0.5 * (self._v0[edges[:, 0]]
                                          + self._v0[edges[:, 1]])])
        nrm = np.vstack([self._n0, 0.5 * (self._n0[edges[:, 0]]
                                          + self._n0[edges[:, 1]])])
        fade = np.concatenate([self._fade, 0.5 * (self._fade[edges[:, 0]]
                                                  + self._fade[edges[:, 1]])])

        # Front of the head only — a dot lattice on the far side would show
        # through the face as noise, since dots have no depth to hide behind.
        keep = nrm[:, 2] > -0.12
        pts, nrm, fade = pts[keep], nrm[keep], fade[keep]
        self._dot_i = np.flatnonzero(keep)

        d = np.linalg.norm(pts - self._lip_c0[None], axis=1)
        rim = np.abs(1.0 - np.abs(nrm[:, 2])) ** 1.2
        # Dim everything, dim it further still close to the mouth: the halo of
        # blank skin around the lips is what makes the lip dots the subject.
        base = (0.13 + 0.46 * rim) * fade
        base *= 1.0 - 0.62 * np.exp(-(d / 0.19) ** 2)
        base *= np.clip(0.30 + d / 0.34, 0.0, 1.0)
        self._dot_a = np.ascontiguousarray(base, dtype=np.float32)

        # `_dot_i` holds indices into the vertex+midpoint arrays; `_dot_a` is
        # parallel to it, one alpha per surviving lattice point. The stride is
        # applied to both so they stay aligned.
        self._dot_i = self._dot_i[::_DOT_STRIDE]
        base_s = self._dot_a[::_DOT_STRIDE]
        bucket = np.clip((base_s * _ALPHA_BUCKETS).astype(np.int32),
                         0, _ALPHA_BUCKETS - 1)
        self._dot_buckets = [np.flatnonzero(bucket == b)
                             for b in range(_ALPHA_BUCKETS)]

        # Point objects, allocated once and moved every frame.
        # `drawPoints` needs a list of QPointF, and building ~560 of them per
        # frame was a third of the lattice pass on its own — setX/setY on a pool
        # is the same drawing for a fraction of the allocation.
        self._qpool = [QPointF(0.0, 0.0) for _ in range(self._dot_i.size)]
        self._qbucket = [[self._qpool[i] for i in idx]
                         for idx in self._dot_buckets]

        # ── filaments ───────────────────────────────────────────────────────
        self._fil_a = np.array([2.0 * math.pi * k / _N_FIL + 0.13
                                for k in range(_N_FIL)], dtype=np.float64)
        self._fil_ph = np.array([random.random() * 6.283 for _ in range(_N_FIL)],
                                dtype=np.float64)
        self._fil_len = np.array([random.uniform(0.55, 1.0) for _ in range(_N_FIL)],
                                 dtype=np.float64)

        # ── state ───────────────────────────────────────────────────────────
        self._t = 0.0
        self._dt = 1.0 / 30.0   # last step's interval, for the camera's own damping
        self._open = 0.0        # 0..1 aperture, smoothed separately from the jaw
        self._amp = 0.0         # smoothed voice level
        self._scan = -1.2       # normalised scan sweep, -1 .. 1
        self._aim = None        # damped world-space lip centre the lens is aimed at
        self._parts: list[list[float]] = []
        self._emit = 0.0        # fractional particle budget carried between ticks
        self._seed = random.Random(0x5EED)
        self._void_key = None   # (radius, background) of the cached lens void
        self._void = None

    # ── animation ───────────────────────────────────────────────────────────

    def step(self, dt: float, amp: float, speaking: bool = False,
             muted: bool = False, state: str = "",
             v_open: float | None = None, v_wide: float = 0.0,
             v_level: float | None = None,
             v_seq: list | None = None, v_hop: float = 0.02) -> None:
        """Advance the head, then this camera's own motion.

        The head is advanced by the wrapped `HoloAvatar` with the caller's own
        arguments, verbatim — this method adds no articulation of its own beyond
        the aperture, which is a geometric consequence of the jaw, not a second
        opinion about it.
        """
        self._av.step(dt, amp, speaking=speaking, muted=muted, state=state,
                      v_open=v_open, v_wide=v_wide, v_level=v_level,
                      v_seq=v_seq, v_hop=v_hop)

        dt = max(0.001, min(0.10, float(dt)))
        self._dt = dt
        self._t += dt
        amp = max(0.0, min(1.0, float(amp)))
        live = speaking and not muted

        # The aperture follows the jaw, but slightly behind it: letting the lips
        # separate at exactly the jaw's rate makes the two read as one rigid
        # hinge, and a mouth whose *shape* lags its opening is what a real one
        # does. It also keeps a fast consonant from snapping the whole frame.
        tgt = self._av.mouth if live else 0.0
        tau = 0.020 if tgt > self._open else (0.014 if live else 0.070)
        self._open += (tgt - self._open) * (1.0 - math.exp(-dt / tau))

        env = amp if live else 0.0
        self._amp += (env - self._amp) * (1.0 - math.exp(-dt / 0.045))

        self._scan += dt * (0.30 + 0.9 * self._amp)
        if self._scan > 1.25:
            self._scan = -1.25

        self._particles_step(dt, live)

    def _particles_step(self, dt: float, live: bool) -> None:
        """Shed embers off the lip seam, in aperture-normalised units.

        Positions are stored relative to the mouth and scaled by the aperture
        radius at draw time, so they survive a resize and a switch to fullscreen
        without piling up in the wrong place.
        """
        # Emission is driven by *change* as much as by level: a consonant
        # closure is a fast move with almost no sound, and it is exactly the
        # moment a real speaker pushes a little air out.
        budget = (2.0 + 30.0 * self._amp) * (0.25 + 0.75 * self._open)
        if not live:
            budget *= 0.0
        self._emit += budget * dt

        while self._emit >= 1.0 and len(self._parts) < _MAX_PARTICLES:
            self._emit -= 1.0
            r = self._seed
            # Along the seam, plus a little depth scatter in y.
            u = r.uniform(-0.52, 0.52)
            v = r.uniform(-1.0, 1.0) * (0.03 + 0.09 * self._open)
            ang = math.atan2(v, u * 1.9) + r.uniform(-0.45, 0.45)
            spd = r.uniform(0.16, 0.62) * (0.45 + 0.9 * self._amp)
            life = r.uniform(0.45, 1.15)
            self._parts.append([
                u, v,
                math.cos(ang) * spd,
                math.sin(ang) * spd - r.uniform(0.0, 0.16),   # buoyant
                life, life,
                r.uniform(0.008, 0.020),
                r.random(),
            ])
        if self._emit > 3.0:
            self._emit = 3.0

        alive = []
        for q in self._parts:
            q[4] -= dt
            if q[4] <= 0.0:
                continue
            q[0] += q[2] * dt
            q[1] += q[3] * dt
            q[2] *= 1.0 - 1.6 * dt          # drag: embers coast, they don't fly
            q[3] -= 0.30 * dt               # and they rise
            alive.append(q)
        self._parts = alive

    # ── geometry ────────────────────────────────────────────────────────────

    def _aperture(self, verts: np.ndarray) -> np.ndarray:
        """Separate the lips along their own rig weights, in place.

        The jaw rotation has already run inside `HoloAvatar.pose()`; this only
        adds the separation the wide shot does not need and could not survive —
        at 20 px it is invisible, at 20× it is the difference between an opening
        and a sliding silhouette.
        """
        o = self._open
        if o <= 0.002:
            return verts
        verts[:, 1] -= self._w_lo * (o * _AP_LOW)
        verts[:, 1] += self._w_up * (o * _AP_UP)
        verts[:, 2] -= self._w_lo * (o * _AP_BACK)
        return verts

    def _project(self, verts: np.ndarray, cx: float, cy: float, r: float):
        """Screen-space coordinates plus the framed lip centre.

        A real perspective divide per vertex — the same projection the wide shot
        uses, only with a longer lens and a much larger scale — which is what
        keeps the cheekbones curving away from the mouth instead of laying the
        face out flat.
        """
        w = _CAM_D - verts[:, 2]
        np.maximum(w, 0.40, out=w)
        pers = _CAM_D / w

        # Anchor on the *posed* lip centre, not the rest one: pitch moves the
        # mouth about 0.04 down the face, which at this magnification would
        # walk it out of the frame on its own.
        #
        # The aim is then *damped* rather than snapped. A lens welded to the
        # mouth looks like the face is fixed and the camera is attached to it;
        # trailing it by a fifth of a second makes the head look like it is
        # moving and the camera like it is following, which is the difference
        # between a diagram and a shot.
        anchor = verts[self._lip_out].mean(axis=0)
        if self._aim is None:
            self._aim = anchor.copy()
        else:
            self._aim += (anchor - self._aim) * (
                1.0 - math.exp(-self._dt / 0.18))
        anchor = self._aim

        aw = max(0.40, _CAM_D - float(anchor[2]))
        ap = _CAM_D / aw
        ax = float(anchor[0]) * ap
        ay = float(anchor[1]) * ap

        s = r / _VIEW_H
        xs = cx + (verts[:, 0] * pers - ax) * s
        ys = cy - (verts[:, 1] * pers - ay) * s

        # Where the mouth actually landed. It is not exactly (cx, cy) — the aim
        # lags the head — so the subject drifts a few pixels around the
        # crosshair, which is what a camera on a talking head looks like.
        mcx = cx + (float(anchor[0]) * ap - ax) * s
        mcy = cy - (float(anchor[1]) * ap - ay) * s
        return xs, ys, mcx, mcy

    # ── rendering ───────────────────────────────────────────────────────────

    def paint(self, p: QPainter, cx: float, cy: float, r: float,
              primary: QColor, accent: QColor, bg: QColor | None = None) -> None:
        """Draw the close-up with its aperture centred at (cx, cy).

        `r` is the aperture radius in pixels; the caller sizes it, so the HUD
        can fit the lens to whatever room the status line leaves it.
        """
        if bg is None:
            bg = QColor(0, 0, 0)
        r = max(24.0, float(r))

        verts, _ = self._av.pose()
        verts = self._aperture(verts)
        xs, ys, mcx, mcy = self._project(verts, cx, cy, r)
        amp = self._amp

        # ── lens ────────────────────────────────────────────────────────────
        # Everything below is clipped to the aperture, so nothing the head does
        # can spill into the status text or the waveform.
        circle = QPainterPath()
        circle.addEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))
        p.save()
        p.setClipPath(circle)

        self._paint_bloom(p, mcx, mcy, r, primary, accent, amp, bg)
        self._paint_lattice(p, xs, ys, r, primary, bg, amp)
        self._paint_mouth(p, xs, ys, r, mcx, mcy, primary, accent, bg, amp)
        self._paint_filaments(p, mcx, mcy, r, primary, accent, amp)
        self._paint_particles(p, mcx, mcy, r, primary, accent, amp)

        p.restore()

    # -- background ---------------------------------------------------------

    def _paint_bloom(self, p: QPainter, mcx: float, mcy: float, r: float,
                     primary: QColor, accent: QColor, amp: float,
                     bg: QColor) -> None:
        """A pool of light behind the mouth, and the dark of the lens around it."""
        # Void: the frame edges fall away so the head has no outline to read as
        # a boundary. Without this the aperture crops a face, and a cropped face
        # reads as a mistake; with it, it reads as a lens.
        #
        # Cached as a pixmap keyed on (radius, background) because it changes
        # only when the window is resized or the theme is changed, while a large
        # radial gradient costs ~1.5 ms to build every frame.
        key = (round(r, 1), bg.rgb())
        if self._void_key != key or self._void is None:
            ext = max(8, int(math.ceil(r * 2.5)))
            pm = QPixmap(ext, ext)
            pm.fill(QColor(bg.red(), bg.green(), bg.blue(), 0))
            q = QPainter(pm)
            q.setRenderHint(QPainter.RenderHint.Antialiasing)
            g = QRadialGradient(ext / 2.0, ext / 2.0, ext / 2.0)
            g.setColorAt(0.00, _c(bg, 0))
            g.setColorAt(0.44, _c(bg, 150))
            g.setColorAt(0.80, _c(bg, 236))
            g.setColorAt(1.00, _c(bg, 252))
            q.setPen(Qt.PenStyle.NoPen)
            q.setBrush(QBrush(g))
            q.drawEllipse(QRectF(0, 0, ext, ext))
            q.end()
            self._void = pm
            self._void_key = key
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPixmap(QPointF(mcx - r * 1.25, mcy - r * 1.25), self._void)

        # Key light. Cold when idle, warm while there is a voice, so the frame
        # gains colour exactly when it gains energy.
        hot = QRadialGradient(mcx, mcy, r * 1.15)
        col = _c(accent, 16 + 74 * amp) if amp > 0.02 else _c(primary, 20)
        hot.setColorAt(0.00, col)
        hot.setColorAt(0.34, _c(col, col.alpha() * 0.45))
        hot.setColorAt(1.00, _c(col, 0))
        p.setBrush(QBrush(hot))
        p.drawEllipse(QRectF(mcx - r * 1.15, mcy - r * 1.15, r * 2.3, r * 2.3))

    # -- skin ---------------------------------------------------------------

    def _paint_lattice(self, p: QPainter, xs, ys, r: float,
                       primary: QColor, bg: QColor, amp: float) -> None:
        """The face, as a field of points rather than a lit surface."""
        # The lattice is the mesh's vertices *plus* its edge midpoints, so the
        # projected midpoint field is built here rather than indexed out of the
        # vertex arrays — one extra concatenate buys an evenly spaced field.
        x = np.concatenate([xs, 0.5 * (xs[self._e0] + xs[self._e1])])[self._dot_i]
        y = np.concatenate([ys, 0.5 * (ys[self._e0] + ys[self._e1])])[self._dot_i]

        for q, px, py in zip(self._qpool, x.tolist(), y.tolist()):
            q.setX(px)
            q.setY(py)

        w = max(1.0, r * _DOT_R * 2.0)
        p.setBrush(Qt.BrushStyle.NoBrush)

        # Two passes over the same points: a wide dim one that fattens each point
        # into a glow, and the crisp one on top. Cheaper than a blur, and it is
        # what makes a dot field read as light rather than as grain.
        #
        # The glow pass runs aliased and the crisp pass does not. The glow is a
        # soft, barely-there disc where an uneven edge cannot be seen, and it
        # carries most of the cost — a ~14 px round cap over 560 points — so
        # this is where the antialiasing budget goes furthest. The crisp pass is
        # what the eye actually lands on, so it keeps its smoothing.
        #
        # Crucially the hint is flipped *twice* per frame, not twice per bucket:
        # changing the render hint flushes the painter's path, and doing it in
        # the inner loop cost 1.7 ms of the 4.2 ms this section used to take.
        alpha = []
        for b, idx in enumerate(self._dot_buckets):
            a = 0.0 if idx.size == 0 else 255.0 * min(1.0, (b + 0.5) / _ALPHA_BUCKETS)
            a *= 0.62 + 0.55 * amp
            alpha.append(a if a >= 2.0 else 0.0)

        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for b, a in enumerate(alpha):
            if a <= 0.0:
                continue
            p.setPen(QPen(_blend(bg, primary, a * 0.24), w * 2.0,
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPoints(self._qbucket[b])

        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for b, a in enumerate(alpha):
            if a <= 0.0:
                continue
            p.setPen(QPen(_blend(bg, primary, a), w,
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPoints(self._qbucket[b])

    # -- mouth --------------------------------------------------------------

    def _paint_mouth(self, p: QPainter, xs, ys, r: float, mcx: float, mcy: float,
                     primary: QColor, accent: QColor, bg: QColor,
                     amp: float) -> None:
        """Cavity, teeth, tongue and the two lip rings."""
        o = self._open

        inner = QPolygonF([QPointF(float(a), float(b))
                           for a, b in zip(xs[self._lip_in], ys[self._lip_in])])
        br = inner.boundingRect()
        open_h = br.height()

        # ── cavity ──────────────────────────────────────────────────────────
        # Never pure black: a black hole in a glowing face reads as a missing
        # polygon, not a mouth. Tinted with the theme so it stays part of the
        # hologram, and darkened as it opens.
        if o > 0.01:
            p.setPen(Qt.PenStyle.NoPen)
            cav = _blend(bg, primary, 10 + 30 * o)
            p.setBrush(QBrush(cav))
            p.drawPolygon(inner)

            # Tongue: a dim warm shape sitting in the floor of the mouth. At
            # this magnification its absence is the thing you notice — an open
            # mouth with nothing behind the lower teeth looks hollowed out.
            if open_h > r * 0.10:
                tw = br.width() * 0.52
                th = max(1.0, open_h * 0.42)
                ty = br.top() + open_h * 0.78
                p.setBrush(QBrush(_c(accent, (30 + 26 * o) * 0.9)))
                p.drawEllipse(QPointF(br.center().x(), ty), tw * 0.5, th * 0.5)

            # The warm pool at the back of the throat, brightest when wide
            # open. Under the teeth, not over them: the wide shot gets away with
            # washing the whole cavity because its mouth is 20 px across, and at
            # this size that wash tints the teeth orange and turns them into a
            # glowing bar.
            p.setBrush(QBrush(_c(accent, (14 + 26 * o) * 0.55)))
            p.drawPolygon(inner)

            self._paint_teeth(p, xs, ys, r, inner, open_h, primary, bg, o)

        # ── lip rings ───────────────────────────────────────────────────────
        # Rim line first, thin: it holds the lip shape together between the
        # dots when they spread apart on an open vowel.
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(_c(primary, (110 + 90 * o)), max(1.0, r * 0.004)))
        p.drawPolygon(inner)

        i0, i1, tt = self._sn_out
        ox = xs[i0] * (1.0 - tt) + xs[i1] * tt
        oy = ys[i0] * (1.0 - tt) + ys[i1] * tt

        i0, i1, tt = self._sn_in
        ix = xs[i0] * (1.0 - tt) + xs[i1] * tt
        iy = ys[i0] * (1.0 - tt) + ys[i1] * tt

        bright = 0.42 + 0.58 * amp
        w_out = max(1.2, r * _LIP_DOT * 2.0)
        w_in = max(1.0, r * _LIP_DOT * 1.45)

        # Three passes per ring, widest and dimmest first: glow, body, spark.
        # Drawn in Plus so overlapping dots accumulate into light instead of
        # flattening into a solid line — which is the whole trick of a dot
        # field that still reads as an outline.
        p.save()
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

        qo = [QPointF(float(a), float(b)) for a, b in zip(ox, oy)]
        qi = [QPointF(float(a), float(b)) for a, b in zip(ix, iy)]

        p.setPen(QPen(_c(primary, 30 * bright), w_out * 2.0,
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPoints(qo)
        p.setPen(QPen(_c(accent if amp > 0.06 else primary, 190 * bright),
                      w_out, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPoints(qo)

        p.setPen(QPen(_c(primary, 18 * bright), w_in * 1.9,
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPoints(qi)
        p.setPen(QPen(_c(primary, 150 * bright), w_in,
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPoints(qi)
        p.restore()

    # -- teeth ---------------------------------------------------------------

    def _paint_teeth(self, p: QPainter, xs, ys, r: float, inner: QPolygonF,
                     open_h: float, primary: QColor, bg: QColor,
                     o: float) -> None:
        """Two arcs of teeth, separated into individual teeth.

        The wide shot draws the upper row as one bright strip, which is right at
        20 px: one strip and eight teeth are the same three pixels. At 20× they
        are not — a solid band is a *bar*, and a bar is the single thing that
        makes a stylised open mouth look like a decal instead of a face. So each
        row is cut into teeth by draw depth, not by drawing outlines.
        """
        p.setPen(Qt.PenStyle.NoPen)

        # ── lower row ───────────────────────────────────────────────────────
        # Dim, short and drawn as one unbroken arc: lower teeth sit in the jaw's
        # shadow and are mostly covered by the lip and the tongue. Cutting gaps
        # into a strip this thin does not read as teeth — it reads as a row of
        # dots — so the shadow does the work and the separations are left out.
        lo = self._lip_in[:10]
        lx, ly = xs[lo], ys[lo]
        lh = max(1.5, open_h * 0.17)
        pts = [QPointF(float(a), float(b)) for a, b in zip(lx, ly)]
        pts += [QPointF(float(a), float(b) - lh)
                for a, b in zip(lx[::-1], ly[::-1])]
        p.setBrush(QBrush(_blend(bg, primary, 66 + 46 * o)))
        p.drawPolygon(QPolygonF(pts))

        # ── upper row ───────────────────────────────────────────────────────
        ux, uy = xs[self._lip_up], ys[self._lip_up]
        th = max(1.5, open_h * 0.32)

        pts = [QPointF(float(a), float(b)) for a, b in zip(ux, uy)]
        pts += [QPointF(float(a), float(b) + th)
                for a, b in zip(ux[::-1], uy[::-1])]
        p.setBrush(QBrush(_blend(bg, primary, 152 + 68 * o)))
        p.drawPolygon(QPolygonF(pts))

        # A soft shade along the biting edge, so the row has thickness instead
        # of ending in a hard line.
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(_blend(bg, primary, 76), max(1.0, r * 0.006)))
        p.drawPolyline(QPolygonF([QPointF(float(a), float(b) + th)
                                  for a, b in zip(ux, uy)]))

        self._tooth_gaps(p, ux, uy, th, r, bg, 1.0)

    def _tooth_gaps(self, p: QPainter, ax, ay, drop: float, r: float,
                    bg: QColor, strength: float) -> None:
        """Vertical cut marks between the teeth, spaced along the arc.

        Even *arc-length* spacing, not even index spacing: the landmark ring is
        denser at the corners than across the front, so indexing by landmark
        would bunch the incisors together and stretch the molars wide.
        """
        n = len(ax)
        if n < 6:
            return

        # Cumulative chord length along the arc, used as the parameter.
        seg = np.hypot(np.diff(ax), np.diff(ay))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        width = abs(float(ax[-1] - ax[0]))
        if total <= 1.0 or width <= 1.0:
            return

        # Teeth are ~1/8 of a mouth wide, a bit wider at the front. Clamped so a
        # narrow frame does not grow a comb and a wide one does not lose them.
        count = int(np.clip(round(width / (r * 0.072)), 5, 11))
        xs_arc = np.linspace(0.0, total, n)
        pen = QPen(_c(bg, 244 * strength), max(1.0, r * 0.0065))
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for k in range(1, count):
            f = k / count
            # Squeeze towards the centre slightly: the incisors are the widest.
            s = (f - 0.5) * (1.0 - 0.18 * (0.5 - abs(f - 0.5)) * 2.0) + 0.5
            t = s * total
            x = float(np.interp(t, cum, ax))
            y = float(np.interp(t, cum, ay))
            # Gaps start just under the gum line and stop short of the edge, so
            # the biting edge stays an unbroken line of enamel.
            p.drawLine(QPointF(x, y + drop * 0.10),
                       QPointF(x, y + drop * 0.88))

    # -- energy -------------------------------------------------------------

    def _paint_filaments(self, p: QPainter, mcx: float, mcy: float, r: float,
                         primary: QColor, accent: QColor, amp: float) -> None:
        """Lines of force leaving the lip corners.

        They are the mouth's *area of effect* made visible: length and opacity
        both follow the level, so a silent mouth holds a tight ring and a loud
        one throws lines most of the way across the lens.
        """
        # Vertical extent from the aperture itself, so the filaments always
        # start on the lips rather than near them.
        rx = r * (0.45 + 0.30 * self._open)
        ry = r * (0.10 + 0.22 * self._open)

        energy = 0.10 + 0.90 * amp
        ph = self._t * 1.9 + self._fil_ph
        shimmer = 0.58 + 0.42 * np.sin(ph)

        p.save()
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        path = QPainterPath()
        long_path = QPainterPath()

        for k in range(_N_FIL):
            a = self._fil_a[k]
            ca, sa = math.cos(a), math.sin(a)
            sx = mcx + ca * rx
            sy = mcy + sa * ry
            ln = r * (0.10 + 0.34 * self._fil_len[k]) * (0.30 + 1.10 * energy)
            ln *= shimmer[k]
            ex = sx + ca * ln
            ey = sy + sa * ln * 0.72
            # Bow the line sideways: a straight ray reads as a hair, a curved
            # one reads as a field line.
            bow = 0.30 * ln * math.sin(a * 3.0 + ph[k] * 0.7)
            mx = (sx + ex) * 0.5 - sa * bow
            my = (sy + ey) * 0.5 + ca * bow
            tgt = long_path if ln > r * 0.16 else path
            tgt.moveTo(sx, sy)
            tgt.quadTo(mx, my, ex, ey)

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(_c(primary, 30 * energy), max(1.0, r * 0.010),
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(long_path)
        p.setPen(QPen(_c(accent if amp > 0.05 else primary, 150 * energy),
                      max(1.0, r * 0.0035),
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(long_path)
        p.setPen(QPen(_c(accent if amp > 0.05 else primary, 120 * energy),
                      max(1.0, r * 0.0035),
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(path)
        p.restore()

    def _paint_particles(self, p: QPainter, mcx: float, mcy: float, r: float,
                         primary: QColor, accent: QColor, amp: float) -> None:
        """Embers shed by the lip seam — the mouth's exhaust."""
        if not self._parts:
            return
        base = accent if amp > 0.05 else primary

        p.save()
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for q in self._parts:
            f = q[4] / q[5]                       # 1 at birth, 0 at death
            a = (f ** 1.4) * (0.30 + 0.70 * q[7]) * (0.35 + 0.75 * amp)
            if a < 0.02:
                continue
            x = mcx + q[0] * r
            y = mcy + q[1] * r
            w = max(1.0, q[6] * r * 2.0)
            p.setPen(QPen(_c(base, 34 * a), w * 2.6,
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPoint(QPointF(x, y))
            p.setPen(QPen(_c(base, 210 * a), w,
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPoint(QPointF(x, y))
        p.restore()
