"""
Voice orb for the HUD — the assistant's voice as a ball of light, made of dots.

Design notes
------------
* **Nothing is drawn around it.** No reticle, no ring, no corner brackets, no
  readout, no tick marks. The other two centrepieces are instruments showing you
  a subject; this one *is* the subject, and a frame around it would be a second,
  louder thing competing with the only thing on screen. So there is no frame:
  the orb fades into its own edge and the background resumes.

* **Dots, not surfaces.** The ball is ~1300 points distributed evenly through a
  volume. A volume projected onto a screen is naturally densest at the centre of
  the disc, so uniform sampling produces a bright core and a sparse halo without
  any shading model doing the work. Depth sorts the rest: points behind the
  equator dim and shrink, points in front brighten and grow, and the eye builds
  a sphere out of two numbers per dot.

* **Every motion is the voice.** The radius breathes with the level, a low
  ripple rolls across the surface on loud syllables, a band of brightness sweeps
  through it while JARVIS is talking, and embers come off it. Silent, it turns
  slowly and shimmers faintly — and nothing else happens, because nothing else
  would mean anything.

* **Software, like the rest of the HUD.** QPainter and numpy only: no GPU, no
  OpenGL, no new package, the same picture on a VM and on a gaming rig.

Same interface as `HoloAvatar` and `MouthCam`, so the HUD can swap one call for
another and change nothing else:

    orb = VoiceOrb()
    orb.step(dt, amp, speaking=..., muted=..., state=...)
    orb.paint(painter, cx, cy, r, primary, accent, bg)
"""

from __future__ import annotations

import math
import random

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QRadialGradient

# Colour helpers shared with the other two centrepieces — "35% of primary over
# the background" has to mean the same thing everywhere in the HUD.
from core.avatar import _blend, _c

# Points in the ball. Enough that the surface reads continuous where it is dense
# and as a cloud at the edge; few enough to draw twice per frame in a HUD that
# is meant to sit running all day.
_N = 1100

# Alpha buckets. Points are sorted by brightness once per frame, then drawn in a
# handful of batched drawPoints() calls instead of one call per dot.
_BUCKETS = 4

# Ball radius, as a fraction of the radius the caller gives us — the remainder is
# room for the ripple and the embers, which push outside the shell at full voice.
_SHELL = 0.90

# Perspective camera distance, in orb radii. Short enough that the ball has
# depth, long enough that the silhouette stays round instead of bulbous.
_CAM = 3.6

_MAX_EMBERS = 150

# Dot size, as a fraction of the orb radius. This is a dot's *diameter*: at
# r = 220 px the nearest points are ~4 px and the furthest ~1.6 px, which is the
# smallest that still reads as a deliberate mark rather than an artefact.
_DOT = 0.015


class VoiceOrb:
    """A dot-built sphere that moves with the level. One per HUD canvas."""

    def __init__(self) -> None:
        rng = np.random.default_rng(20250728)

        # Two populations, and the ratio between them is what makes this a
        # *sphere* rather than a cloud.
        #
        # A volume of points projects to a disc that is brightest at its centre
        # and fades to nothing at the edge — mathematically correct and visually
        # fatal, because a sphere is recognised by its **limb**, and a uniform
        # ball has no limb at all, just a gradient. A shell projects to the
        # opposite: its density peaks exactly at the silhouette, which is the
        # edge the eye is looking for.
        #
        # So most of the points sit in a thin shell, and a third of them fill the
        # inside so the shape has volume rather than being a hollow bubble. The
        # interior points are also what gives the middle something to be dense
        # with when the voice peaks.
        dirs = rng.normal(size=(_N, 3))
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
        u = rng.random(_N)
        shell = rng.random(_N) < 0.68
        rad = np.where(shell, 0.93 + 0.07 * u, 0.92 * u ** (1.0 / 3.0))
        self._rad0 = rad.astype(np.float32)
        self._dirs = dirs.astype(np.float32)

        # Three slow standing waves across the ball. Summed in step(), they give
        # the surface something to do while it breathes that never repeats
        # visibly, and they are the whole reason a ball of dots looks alive
        # rather than like a spinning shell.
        self._wave_axis = rng.normal(size=(3, 3)).astype(np.float32)
        self._wave_axis /= np.linalg.norm(self._wave_axis, axis=1, keepdims=True)
        self._wave_freq = np.array([2.1, 3.7, 5.3], dtype=np.float32)
        self._wave_rate = np.array([0.9, 1.4, 2.1], dtype=np.float32)
        self._wave_gain = np.array([0.030, 0.020, 0.012], dtype=np.float32)

        # Per-dot shimmer phase, so the ball twinkles instead of pulsing as one.
        self._phase = rng.random(_N).astype(np.float32) * 6.2831853

        # Rotation and projection scratch, reused every frame.
        self._q = np.empty((_N, 3), dtype=np.float32)
        self._rad = np.empty(_N, dtype=np.float32)
        self._alpha = np.empty(_N, dtype=np.float32)
        self._size = np.empty(_N, dtype=np.float32)

        # QPointF pool, written in sorted order each frame so each bucket is a
        # contiguous slice of one list. Allocating ~1300 point objects per frame
        # cost more than the drawing did.
        self._pool = [QPointF(0.0, 0.0) for _ in range(_N)]

        self._t = 0.0
        self._dt = 1.0 / 30.0
        self._amp = 0.0        # smoothed voice level
        self._surge = 0.0      # smoothed *change* in level — the strike, not the tone
        self._spin = 0.0       # integrated rotation, never derived from absolute time
        self._tilt = 0.0
        self._scale = 1.0      # smoothed overall size, so it never snaps
        self._dim = 1.0        # 0 asleep/muted .. 1 awake
        self._emit = 0.0       # fractional ember budget carried between ticks
        self._embers: list[list[float]] = []
        self._seed = random.Random(0x5EED0B)

    # ── animation ───────────────────────────────────────────────────────────

    def step(self, dt: float, amp: float, speaking: bool = False,
             muted: bool = False, state: str = "",
             v_open: float | None = None, v_wide: float = 0.0,
             v_level: float | None = None,
             v_seq: list | None = None, v_hop: float = 0.02) -> None:
        """Advance the orb.

        Takes the same arguments as `HoloAvatar.step()` so the HUD can call it
        with the identical signature; the viseme fields are accepted and unused,
        because a ball of light has no lips to shape.
        """
        dt = max(0.001, min(0.10, float(dt)))
        self._t += dt
        self._dt = dt
        amp = max(0.0, min(1.0, float(amp)))
        live = speaking and not muted

        st = (state or "").upper()
        asleep = st in ("SLEEPING", "STANDBY", "OFFLINE")
        thinking = st in ("THINKING", "PROCESSING")

        env = amp if live else 0.0
        self._amp += (env - self._amp) * (1.0 - math.exp(-dt / 0.075))

        # The strike, used for things that should *hit* rather than swell.
        # Reacting to the level alone gives a slow tide; reacting to its rise is
        # what makes a consonant land.
        rise = max(0.0, self._amp - self._surge)
        self._surge += (self._amp - self._surge) * (1.0 - math.exp(-dt / 0.30))

        # Rotation is integrated, not taken from absolute time. Deriving it as
        # `t * speed` means the ball teleports the instant the speed changes —
        # exactly when JARVIS starts or stops talking, which is the worst moment
        # to jump.
        speed = 0.30
        if live:
            speed += 0.85 * self._amp
        if thinking:
            speed *= 0.55          # a thinking orb slows down and concentrates
        if muted or asleep:
            speed *= 0.35
        self._spin += dt * speed
        self._tilt = 0.46 * math.sin(self._t * 0.17) + 0.10 * math.sin(self._t * 0.53)

        # Size follows the level, but lazily and with a floor: a ball that
        # collapses to nothing in silence reads as a fault, not as quiet.
        size_t = (0.94 if asleep else 1.0) + 0.16 * self._amp
        if muted:
            size_t = 0.88
        self._scale += (size_t - self._scale) * (1.0 - math.exp(-dt / 0.10))

        dim_t = 1.0
        if asleep:
            dim_t = 0.32
        if muted:
            dim_t = 0.45
        elif thinking and not live:
            dim_t = 0.82
        self._dim += (dim_t - self._dim) * (1.0 - math.exp(-dt / 0.25))

        self._embers_step(dt, live, rise)

    def _embers_step(self, dt: float, live: bool, rise: float) -> None:
        """Lift embers off the surface, in orb space, in units of the radius.

        Emitted from wherever the ball is loudest rather than uniformly: an orb
        shedding evenly looks like a particle system, and an orb shedding from
        one flank looks like it is being pushed.
        """
        rate = (0.0 if not live else
                5.0 + 70.0 * self._amp + 300.0 * rise)
        self._emit += rate * dt

        r = self._seed
        while self._emit >= 1.0 and len(self._embers) < _MAX_EMBERS:
            self._emit -= 1.0
            # A random point on the front hemisphere, biased to the outside.
            while True:
                d = np.array([r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1)])
                ln = float(np.linalg.norm(d))
                if 0.35 < ln <= 1.0:
                    break
            d /= ln
            if d[2] < 0.0:
                d[2] = -d[2]
            rr = 0.72 + 0.30 * float(r.random()) ** 0.5
            spd = r.uniform(0.16, 0.52) * (0.4 + 1.2 * self._amp)
            life = r.uniform(0.55, 1.35)
            self._embers.append([
                float(d[0] * rr), float(d[1] * rr), float(d[2] * rr),
                float(d[0] * spd), float(d[1] * spd), float(d[2] * spd),
                life, life, r.uniform(0.006, 0.016), r.random(),
            ])
        if self._emit > 3.0:
            self._emit = 3.0

        alive = []
        for e in self._embers:
            e[6] -= dt
            if e[6] <= 0.0:
                continue
            e[0] += e[3] * dt
            e[1] += e[4] * dt
            e[2] += e[5] * dt
            drag = 1.0 - 1.3 * dt
            e[3] *= drag
            e[4] *= drag
            e[5] *= drag
            alive.append(e)
        self._embers = alive

    # ── geometry ────────────────────────────────────────────────────────────

    def _rotate(self) -> tuple[np.ndarray, np.ndarray]:
        """Posed point cloud plus its depth, in orb-local units.

        Returns (px, py, pz) — screen-plane x/y already through the perspective
        divide, and raw z for the depth cues.
        """
        cy, sy = math.cos(self._spin), math.sin(self._spin)
        cp, sp = math.cos(self._tilt), math.sin(self._tilt)

        d = self._dirs
        q = self._q
        np.matmul(d, np.array([[cy, 0.0, sy],
                               [sp * sy, cp, -sp * cy],
                               [-cp * sy, sp, cp * cy]],
                              dtype=np.float32).T, out=q)

        # Standing waves, summed on the *direction* so they travel across the
        # surface rather than pulsing the sphere as a whole.
        lobe = np.zeros(_N, dtype=np.float32)
        proj = np.empty((_N, 3), dtype=np.float32)
        np.matmul(d, self._wave_axis.T, out=proj)
        for k in range(3):
            lobe += self._wave_gain[k] * np.sin(
                proj[:, k] * self._wave_freq[k] + self._t * self._wave_rate[k])

        # Voice displacement. The `sin` term is a *travelling* ridge: at 5-6
        # lobes across the ball it moves visibly in 200 ms, which is syllable
        # speed — anything slower reads as a wobble rather than a word.
        voice = (0.14 * self._amp) * np.sin(
            proj[:, 0] * 5.0 - self._t * 7.5) * np.cos(proj[:, 1] * 3.2)

        breath = 0.030 * math.sin(self._t * 1.35)
        rad = self._rad0 * (1.0 + breath + lobe + voice)
        np.maximum(rad, 0.05, out=rad)

        # The limb is a hard cap, and the cap is the point.
        #
        # Without it, every ripple pushes its own points past the nominal radius
        # and the silhouette dribbles off into strays — the difference between a
        # sphere and a swarm. Clamping instead makes the displaced points pile
        # up *on* the limb, so the edge gets both a definite circle and a
        # brighter rim, for free. The cap breathes with the level, so the sphere
        # swells rather than the rim wandering.
        cap = _SHELL + 0.035 * self._amp
        np.minimum(rad, cap / _SHELL, out=rad)
        self._rad = rad

        q *= rad[:, None]
        w = _CAM - q[:, 2]
        np.maximum(w, 0.35, out=w)
        k = _CAM / w
        q[:, 0] *= k
        q[:, 1] *= k
        return q[:, 0], q[:, 1], q[:, 2]

    # ── rendering ───────────────────────────────────────────────────────────
    #
    # The whole picture is: a soft glow, the ball of dots, the embers. Three
    # layers, no line, no shape, no text — see the module docstring.

    def paint(self, p: QPainter, cx: float, cy: float, r: float,
              primary: QColor, accent: QColor, bg: QColor | None = None) -> None:
        """Draw the orb with its centre at (cx, cy) and radius `r` in pixels."""
        if bg is None:
            bg = QColor(0, 0, 0)
        r = max(20.0, float(r))

        px, py, pz = self._rotate()
        sx = cx + px * r * _SHELL * self._scale
        sy = cy - py * r * _SHELL * self._scale

        self._paint_dots(p, cx, cy, sx, sy, pz, r, primary, accent, bg)
        self._paint_glow(p, cx, cy, r, primary, accent, bg)
        self._paint_core(p, cx, cy, r, primary, accent)
        self._paint_embers(p, cx, cy, r, primary, accent)

    # -- glow ---------------------------------------------------------------

    def _paint_glow(self, p: QPainter, cx: float, cy: float, r: float,
                    primary: QColor, accent: QColor, bg: QColor) -> None:
        """Bloom, added on top of the dots with a plus-composite.

        Drawn *after* the ball rather than behind it, which matters more than it
        sounds: the dots are opaque (that is what keeps them on the painter's
        fast path), so a gradient behind them would be erased point by point and
        the ball would show dark specks in its own core. Adding light on top is
        also what a real bloom does — the bright centre of the cloud bleeds into
        the air around it.
        """
        amp = self._amp
        hot = accent if amp > 0.05 else primary
        g = QRadialGradient(cx, cy, r * (1.30 + 0.10 * amp))
        g.setColorAt(0.00, _c(hot, (14 + 46 * amp) * self._dim))
        g.setColorAt(0.30, _c(primary, (7 + 18 * amp) * self._dim))
        g.setColorAt(0.62, _c(primary, 4 * self._dim))
        g.setColorAt(1.00, _c(bg, 0))
        p.save()
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(g))
        rad = r * 1.40
        p.drawEllipse(QRectF(cx - rad, cy - rad, rad * 2, rad * 2))
        p.restore()

    def _paint_core(self, p: QPainter, cx: float, cy: float, r: float,
                    primary: QColor, accent: QColor) -> None:
        """A tight hot spot where the voice comes from.

        The ball of dots has no single brightest thing in it — it is a cloud,
        and a cloud has no spark — so the frame gets one here. Small on purpose:
        a wide core reads as fog behind the dots, a tight one reads as something
        burning in the middle of them.
        """
        amp = self._amp
        k = r * (0.10 + 0.05 * amp)
        hot = accent if amp > 0.05 else primary
        # Three stops, not two: an inner white hot spot that no theme colour can
        # reach on its own. Additive light goes to white as it saturates, so the
        # centre of something genuinely bright *has* to be white — a core that
        # stays the accent colour reads as tinted fog, which is exactly what the
        # first version of this looked like.
        #
        # The alphas are near-opaque at full voice on purpose: a plus-composite
        # adds source * alpha to the background, so 135 of white over near-black
        # is a grey 135, not a hot spot. It has to clip to be white.
        g = QRadialGradient(cx, cy, k)
        wht = QColor(255, 255, 255)
        g.setColorAt(0.00, _c(wht, (70 + 185 * amp) * self._dim))
        g.setColorAt(0.26, _c(hot, (40 + 150 * amp) * self._dim))
        g.setColorAt(0.60, _c(hot, (10 + 40 * amp) * self._dim))
        g.setColorAt(1.00, _c(hot, 0))
        p.save()
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(g))
        p.drawEllipse(QRectF(cx - k, cy - k, k * 2, k * 2))
        p.restore()

    # -- the ball -----------------------------------------------------------

    def _paint_dots(self, p: QPainter, cx: float, cy: float, sx, sy, pz,
                    r: float, primary: QColor, accent: QColor,
                    bg: QColor) -> None:
        """Every point, sorted by brightness and drawn in batched calls."""
        amp = self._amp

        # Depth: the only shading this renderer has, and it is enough. Points on
        # the near side run bright and large, points behind the equator dim to a
        # quarter — which is the whole illusion of a sphere.
        depth = np.clip(pz * 0.5 + 0.5, 0.0, 1.0)      # 0 far .. 1 near

        # A band of brightness travelling through the ball while it speaks. It
        # is light moving *over* the dots, not a shape drawn on them, so it
        # reads as energy passing through rather than as a ring around anything.
        sweep = np.zeros(_N, dtype=np.float32)
        if amp > 0.02:
            band = np.sin(pz * 4.2 - self._t * 6.0)
            sweep = amp * 0.50 * np.maximum(0.0, band)

        twinkle = 0.86 + 0.14 * np.sin(self._t * 2.4 + self._phase)

        # The limb again, this time as light: a sphere of any material is
        # brightest where you are looking *along* its surface rather than into
        # it, and on a shell of dots that is `|z| -> 0`. Depth alone lights the
        # near face and leaves the silhouette dim, which is what flattened the
        # first version of this into a disc.
        rim = np.maximum(0.0, 1.0 - np.abs(pz)) ** 1.6

        a = (0.14 + 0.72 * depth ** 1.30 + 0.62 * rim) * twinkle
        a *= 0.45 + 0.75 * self._dim
        a += sweep * depth
        np.clip(a, 0.0, 1.0, out=a)

        # Size, and the warm tint, are both depth ramps: nothing here distributes
        # colour the way a surface shader would — one number per dot is all this
        # renderer carries, so both cues hang off the same one.
        size = (0.60 + 0.80 * depth) * _DOT * r
        np.maximum(size, 1.0, out=size)
        mix = 0.10 + 0.65 * amp * depth

        # One sort, then each bucket is a contiguous slice of one list — no
        # per-bucket masking, and no per-point Python objects built from scratch
        # every frame.
        bucket = np.clip((a * _BUCKETS).astype(np.int32), 0, _BUCKETS - 1)
        order = np.argsort(bucket, kind="stable")
        bounds = np.searchsorted(bucket[order], np.arange(_BUCKETS + 1))

        xs = sx[order].tolist()
        ys = sy[order].tolist()
        for q, x, y in zip(self._pool, xs, ys):
            q.setX(x)
            q.setY(y)

        widths = np.maximum(1.0, np.array(
            [size[order][int(bounds[b]):int(bounds[b + 1])].mean()
             if int(bounds[b + 1]) > int(bounds[b]) else 1.0
             for b in range(_BUCKETS)], dtype=np.float32))
        mixes = np.array([
            mix[order][int(bounds[b]):int(bounds[b + 1])].mean()
            if int(bounds[b + 1]) > int(bounds[b]) else 0.0
            for b in range(_BUCKETS)], dtype=np.float32)

        p.setBrush(Qt.BrushStyle.NoBrush)

        # Glow pass, aliased: a wide dim round cap under a dot is what makes the
        # cloud read as light rather than as a scatter plot, and at these widths
        # the antialiasing is invisible while costing more than the drawing does.
        #
        # Only the two brightest buckets get one. A halo under a dot that is
        # already dim is invisible — it is the *faint* points that cost the most
        # here and earn the least, and skipping them takes the pass from 1.0 to
        # 0.55 of the drawing time. The crisp pass below keeps its smoothing.
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for b in range(_BUCKETS // 2, _BUCKETS):
            lo, hi = int(bounds[b]), int(bounds[b + 1])
            if hi <= lo:
                continue
            al = 255.0 * min(1.0, (b + 0.6) / _BUCKETS)
            p.setPen(QPen(_blend(bg, primary, al * 0.20), widths[b] * 1.8,
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPoints(self._pool[lo:hi])

        # The hottest dots turn warm as the voice rises; the rest keep the theme
        # colour, so the orb takes on colour with energy instead of being
        # recoloured wholesale.
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for b in range(_BUCKETS):
            lo, hi = int(bounds[b]), int(bounds[b + 1])
            if hi <= lo:
                continue
            al = 255.0 * min(1.0, (b + 0.6) / _BUCKETS)
            col = _blend(primary, accent, float(mixes[b]) * 255.0)
            p.setPen(QPen(_blend(bg, col, al), widths[b],
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPoints(self._pool[lo:hi])

    # -- embers -------------------------------------------------------------

    def _paint_embers(self, p: QPainter, cx: float, cy: float, r: float,
                      primary: QColor, accent: QColor) -> None:
        """Points that have left the ball. Same material, no longer attached."""
        if not self._embers:
            return
        s = r * _SHELL * self._scale
        base = accent if self._amp > 0.05 else primary

        # Three brightness bands, drawn as three batched calls. Setting a pen
        # per ember was 190 painter state changes a frame for what is, at most,
        # a hundred 2 px dots — more expensive than everything the embers do.
        bands = ([], [], [])
        for e in self._embers:
            f = e[6] / e[7]
            a = (f ** 1.5) * (0.35 + 0.65 * e[9]) * (0.30 + 0.80 * self._dim)
            if a < 0.02:
                continue
            k = _CAM / max(0.35, _CAM - e[2])
            bands[min(2, int(a * 3.0))].append(
                QPointF(cx + e[0] * k * s, cy - e[1] * k * s))

        p.save()
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i, pts in enumerate(bands):
            if not pts:
                continue
            a = (i + 0.6) / 3.0
            p.setPen(QPen(_c(base, 44 * a), max(3.0, r * 0.030),
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPoints(pts)
            p.setPen(QPen(_c(base, 215 * a), max(1.0, r * 0.009),
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPoints(pts)
        p.restore()
