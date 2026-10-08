"""Metric water-plane camera model: image pixels <-> metres on the lake, in the boat's frame.

Geometry
--------
Boat frame (moves with the boat): origin on the water directly below the camera (on the pylon/tower),
  Z = distance BEHIND the boat along its axis (towards the skier), X = sideways, Y = up.
  The course centre line (the boat path) is therefore simply X = 0 in every frame, however the
  operator pans/tilts the camera.
Camera: pinhole, focal length f [px], principal point = image centre, height h [m] above the water,
  orientation = yaw psi (pan), pitch theta (tilt down > 0), roll rho.
Every buoy and the skier's feet lie on the water plane Y = 0, so one pixel -> one ray -> one point (X, Z).

What is measured / estimated
----------------------------
* Frame-to-frame camera ROTATION comes from sparse optical flow on the far background (shore/trees),
  like camera-motion compensation in BoT-SORT: (dx, dy, droll) -> dpsi = -atan(dx/f), dtheta = -atan(dy/f).
* The unknown constants (f, h, pitch theta0, a slowly varying yaw correction (knots every 3 s), boat speed v)
  are CALIBRATED per shot with slalom facts, by robust least squares:
    - the skier is at the end of a taut rope:      |(X_skier, Z_skier)| ~ rope length (+ ~0.8 m to the feet)
    - buoys are anchored: in the boat frame they keep X and recede at boat speed:  Z_b(t) = Z_b0 + v t
    - the skier swings symmetrically about the boat path:  mean X_skier ~ 0 over each few seconds
    - priors: camera 1-3.5 m above water, boat 52-60 km/h.
  The fitted values are reported so they can be sanity-checked (boat speed, camera height, buoy offsets).
"""
from __future__ import annotations

import cv2
import numpy as np


# --------------------------------------------------------------------------------------------- motion
class CameraMotion:
    """Per-frame camera rotation increments (dx, dy [px at full resolution], droll [rad]) from LK flow
    on the background (upper image band, minus overlays and the skier)."""

    def __init__(self, band=(0.04, 0.36), scale=0.5, max_pts=400):
        self.band, self.scale, self.max_pts = band, scale, max_pts
        self.prev = None

    def reset(self):
        self.prev = None

    def update(self, frame, exclude=()):
        g = cv2.cvtColor(cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        inc = (0.0, 0.0, 0.0)
        if self.prev is not None and self.prev.shape == g.shape:
            h, w = g.shape
            m = np.zeros((h, w), np.uint8)
            m[int(h * self.band[0]):int(h * self.band[1]), :] = 255
            m[:int(h * 0.16), :int(w * 0.22)] = 0
            m[:int(h * 0.16), int(w * 0.78):] = 0
            for b in exclude:
                x0, y0, x1, y1 = (int(v * self.scale) for v in b[:4])
                pw, ph = (x1 - x0) // 2 + 4, (y1 - y0) // 2 + 4
                m[max(0, y0 - ph):y1 + ph, max(0, x0 - pw):x1 + pw] = 0
            pts = cv2.goodFeaturesToTrack(self.prev, self.max_pts, 0.01, 6, mask=m)
            if pts is not None and len(pts) >= 8:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, pts, None, winSize=(21, 21), maxLevel=3)
                ok = st.ravel() == 1
                if ok.sum() >= 8:
                    M, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC,
                                                         ransacReprojThreshold=2.0)
                    if M is not None and inl is not None and inl.sum() >= 6:
                        a = np.arctan2(M[1, 0], M[0, 0])
                        # displacement near the horizon row: forward motion of the boat makes the shore
                        # recede towards the vanishing point; that parallax is smallest near the horizon
                        cxs, cys = w / 2, 0.33 * h
                        dx = M[0, 0] * cxs + M[0, 1] * cys + M[0, 2] - cxs
                        dy = M[1, 0] * cxs + M[1, 1] * cys + M[1, 2] - cys
                        inc = (dx / self.scale, dy / self.scale, float(a))
        self.prev = g
        return inc


def _smooth(x, w):
    """Centred moving average (edge-padded): the operator pans smoothly, frame-to-frame flow noise is removed."""
    if w < 2:
        return x
    k = np.ones(2 * w + 1) / (2 * w + 1)
    return np.convolve(np.pad(x, w, mode="edge"), k, mode="valid")


# --------------------------------------------------------------------------------------------- model
def _rot(theta, psi, rho):
    """Camera(x right, y down, z forward) -> world(X, Y up, Z back) rotation, vectorised over frames."""
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(psi), np.sin(psi)
    cr, sr = np.cos(rho), np.sin(rho)
    return ct, st, cp, sp, cr, sr


def pixel_to_water(u, v, f, cx, cy, h, theta, psi, rho=0.0):
    """Pixel(s) -> (X, Z) metres on the water in the boat frame. Points above the horizon -> NaN."""
    xc, yc = (np.asarray(u, float) - cx) / f, (np.asarray(v, float) - cy) / f
    ct, st, cp, sp, cr, sr = _rot(theta, psi, rho)
    # undo roll (rotate image coordinates)
    xr, yr = cr * xc - sr * yc, sr * xc + cr * yc
    # camera -> y-up, then pitch (about X) then yaw (about Y)
    x, y, z = xr, -yr, np.ones_like(xr)
    y2, z2 = ct * y - st * z, st * y + ct * z
    X, Z = cp * x + sp * z2, -sp * x + cp * z2
    t = np.where(y2 < -1e-6, -h / np.where(y2 < -1e-6, y2, -1), np.nan)
    return X * t, Z * t


def water_to_pixel(X, Z, f, cx, cy, h, theta, psi, rho=0.0):
    """(X, Z) metres on the water -> pixel (u, v). Points behind the camera -> NaN."""
    X, Z = np.asarray(X, float), np.asarray(Z, float)
    ct, st, cp, sp, cr, sr = _rot(theta, psi, rho)
    Y = -h * np.ones_like(X)
    # inverse yaw, inverse pitch
    x, z2 = cp * X - sp * Z, sp * X + cp * Z
    y, z = ct * Y + st * z2, -st * Y + ct * z2
    ok = z > 1e-3
    xc, yc = x / np.where(ok, z, 1), -y / np.where(ok, z, 1)
    xr, yr = cr * xc + sr * yc, -sr * xc + cr * yc
    return np.where(ok, cx + f * xr, np.nan), np.where(ok, cy + f * yr, np.nan)


# --------------------------------------------------------------------------------------------- rope length
OFF_TO_ROPE_M = {15: 14.25, 22: 13.0, 28: 12.0, 32: 11.25, 35: 10.75, 38: 10.25, 39.5: 9.75, 41: 9.5,
                 41.5: 9.5, 43: 9.25, 44: 9.0}


def rope_length_from_name(name: str, default=11.25):
    """'6 at 32 off' -> 11.25 m; 'Rd3-6@12m' / '6at12' -> 12 m (Greece files give metres)."""
    import re
    n = name.replace("_", " ")
    m = re.search(r"(?:at|@)\s*(\d+(?:\.\d+)?)\s*(m\b)?", n)
    if not m:
        return default
    val = float(m.group(1))
    if m.group(2) or 9 <= val <= 18.25:  # explicit metres, or a value that can only be metres
        return val if 9 <= val <= 18.25 else default
    return OFF_TO_ROPE_M.get(val, default)


# --------------------------------------------------------------------------------------------- calibration
class ShotCalibration:
    """Calibrated camera for one continuous shot (frames s0..s1-1)."""

    def __init__(self, s0, s1, W, H, fps, inc):
        self.s0, self.s1, self.W, self.H, self.fps = s0, s1, W, H, fps
        self.cx, self.cy = W / 2, H / 2
        self.inc = np.asarray(inc[s0:s1], float)  # (n, 3) dx, dy, droll
        self.p = None
        self.ok = False
        # wide boat cameras: ~30-90 deg horizontal field of view. Zoomed (operated) cameras change f over
        # time, which this model does not handle; such fits end at a bound and are flagged.
        self.f_range = (0.5, 2.0)
        self.report = {}

    # angles of every frame of the shot for given f and offsets
    def knot_times(self, every_s=3.0):
        T = len(self.inc) / self.fps
        return np.append(np.arange(0.0, T, every_s), T)

    def angles(self, f, theta0, knots):
        """knots: yaw offset (rad) at self.knot_times(); piecewise-linear correction of the integrated pan,
        which drifts by several degrees over a pass."""
        # pan is integrated from flow; tilt and roll are taken as constant per shot: integrated vertical
        # flow drifts (boat motion -> receding shore looks like tilt) while the operator mostly pans
        dpsi = -np.arctan(self.inc[:, 0] / f)
        t = np.arange(len(self.inc)) / self.fps
        psi = _smooth(np.cumsum(dpsi), int(0.3 * self.fps)) + np.interp(t, self.knot_times(), knots)
        theta = np.full(len(self.inc), theta0)
        rho = np.zeros(len(self.inc))
        return theta, psi, rho

    def fit(self, skier_obs, buoy_tracks, rope_m, v_prior=15.6, anchor_s=2.0, fixed=None, quick=False):
        """skier_obs: list of (frame, u, v); buoy_tracks: list of [(frame, u, v), ...] (observed points).
        fixed=(f_rel, h): lens (focal length / image width) and camera height known, e.g. from a joint
        calibration of several videos filmed with the same boat camera (tools/calibrate_rig.py)."""
        from scipy.optimize import least_squares
        sk = np.array([o for o in skier_obs if self.s0 <= o[0] < self.s1], float)
        if len(sk) < 30:
            self.report = dict(status="too little skier data")
            return False
        sk = sk[:: max(1, len(sk) // 400)]
        tracks = [np.array([o for o in tr if self.s0 <= o[0] < self.s1], float) for tr in buoy_tracks]
        tracks = [tr for tr in tracks if len(tr) >= 4]
        sk_i = (sk[:, 0] - self.s0).astype(int)
        L = rope_m + 0.8  # handle -> feet
        win = (sk_i // int(4 * self.fps))  # 4 s windows for the symmetry term
        wins_big = [w for w in np.unique(win) if (win == w).sum() >= 15]
        # centre-line anchor: at the start of a run the skier is pulled straight behind the boat (X ~ 0)
        start = (sk_i < anchor_s * self.fps) if self.s0 == 0 else np.zeros(len(sk_i), bool)

        def resid(p):
            lf, h, th0, v = p[:4]
            kn = p[4:]
            f = np.exp(lf)
            theta, psi, rho = self.angles(f, th0, kn)
            X, Z = pixel_to_water(sk[:, 1], sk[:, 2], f, self.cx, self.cy, h, theta[sk_i], psi[sk_i], rho[sk_i])
            bad = ~np.isfinite(X)
            X, Z = np.nan_to_num(X, nan=0.0), np.nan_to_num(Z, nan=100.0)
            r = [np.where(bad, 10.0, (np.hypot(X, Z) - L) / 1.5)]
            # symmetric swing per window
            sym = [X[win == w].mean() / 1.5 * np.sqrt((win == w).sum() / 10) for w in np.unique(win)]
            r.append(np.array(sym))
            for tr in tracks:  # fixed residual count per track (least_squares needs a constant length)
                ti = (tr[:, 0] - self.s0).astype(int)
                bx, bz = pixel_to_water(tr[:, 1], tr[:, 2], f, self.cx, self.cy, h, theta[ti], psi[ti], rho[ti])
                okb = np.isfinite(bx) & (np.nan_to_num(bz, nan=99) < L + 12)
                tt = ti / self.fps
                if okb.sum() < 3:
                    r.append(np.zeros(2 * len(tr) + 1))
                    continue
                mx, mz, mt = bx[okb].mean(), bz[okb].mean(), tt[okb].mean()
                r.append(np.where(okb, (np.nan_to_num(bx) - mx) / 0.8, 0.0))
                r.append(np.where(okb, (np.nan_to_num(bz) - (mz + v * (tt - mt))) / 1.5, 0.0))
                # course layout: a buoy is either a turn buoy (11.5 m out) or a gate / boat-guide buoy (~1.2 m)
                ax = abs(mx)
                r.append(np.array([min(abs(ax - 11.5) / 1.2, abs(ax - 1.2) / 0.6)]))
            # the skier reaches the buoy line (~11.5 m out) at every turn, whatever the rope length
            for w in wins_big:
                xs = np.abs(X[win == w])
                r.append(np.array([(np.percentile(xs, 92) - 10.5) / 2.5]))
            r.append(np.array([(v - v_prior) / 0.25]))  # boat speed is set by the rules
            r.append(np.diff(kn) / 0.06)  # yaw correction changes slowly (~3.5 deg per 3 s at most)
            if fixed is None:
                r.append(np.array([(h - 2.2) / 0.6, (lf - np.log(0.95 * self.W)) / 0.6]))  # weak priors
            return np.concatenate(r)

        best = None
        nk = len(self.knot_times())
        lo = [np.log(self.f_range[0] * self.W), 1.5, -0.1, 12.0] + [-1.6] * nk
        hi = [np.log(self.f_range[1] * self.W), 3.2, 0.7, 18.5] + [1.6] * nk
        f_starts = (0.7, 1.0, 1.4, 1.9)
        if fixed is not None:
            lf_fix = np.log(fixed[0] * self.W)
            lo[0], hi[0], lo[1], hi[1] = lf_fix - 1e-6, lf_fix + 1e-6, fixed[1] - 1e-6, fixed[1] + 1e-6
            f_starts = (fixed[0],)
        th_starts, ps_starts = ((0.06, 0.15), (-0.4, 0.0, 0.4)) if quick else ((0.08, 0.16, 0.25), (-0.5, 0.0, 0.5))
        for f0 in f_starts:
            for th0 in th_starts:
                for ps0 in ps_starts:
                    p0 = [np.log(f0 * self.W), fixed[1] if fixed is not None else 2.0, th0, v_prior] + [ps0] * nk
                    try:
                        res = least_squares(resid, p0, loss="soft_l1", f_scale=1.0, max_nfev=200, bounds=(lo, hi))
                    except Exception as e:  # noqa: BLE001
                        self.last_error = repr(e)
                        continue
                    if best is None or res.cost < best.cost:
                        best = res
        if best is None:
            self.report = dict(status="fit failed", error=getattr(self, "last_error", ""))
            return False
        self.p = best.x
        lf, h, th0, v = best.x[:4]
        theta, psi, rho = self.angles(np.exp(lf), th0, best.x[4:])
        self.theta, self.psi, self.rho = theta, psi, rho
        X, Z = self.to_water(sk[:, 0].astype(int), sk[:, 1], sk[:, 2])
        dist = np.hypot(X, Z)
        at_bound = fixed is None and (abs(lf - np.log(self.f_range[0] * self.W)) < 0.02
                                      or abs(lf - np.log(self.f_range[1] * self.W)) < 0.02
                                      or abs(h - 1.5) < 0.02 or abs(h - 3.2) < 0.02)
        self.ok = bool(np.nanmedian(np.abs(dist - L)) < 2.5 and not at_bound)
        self.report = dict(status="ok" if self.ok else ("at parameter bound (zoom camera?)" if at_bound else "poor fit"), f_px=round(float(np.exp(lf)), 1),
                           hfov_deg=round(float(np.degrees(2 * np.arctan(self.W / 2 / np.exp(lf)))), 1),
                           cam_height_m=round(float(h), 2), pitch0_deg=round(float(np.degrees(th0)), 1),
                           boat_speed_kmh=round(float(v * 3.6), 1), rope_m=rope_m,
                           skier_dist_median_m=round(float(np.nanmedian(dist)), 2),
                           skier_dist_mad_m=round(float(np.nanmedian(np.abs(dist - np.nanmedian(dist)))), 2),
                           n_buoy_tracks=len(tracks), cost=round(float(best.cost), 1),
                           n_residuals=int(len(best.fun)), calibration="rig (shared)" if fixed is not None else "per shot")
        return self.ok

    @property
    def f(self):
        return float(np.exp(self.p[0]))

    @property
    def h(self):
        return float(self.p[1])

    @property
    def v(self):
        return float(self.p[3])

    def to_water(self, frame, u, v):
        i = np.asarray(frame, int) - self.s0
        return pixel_to_water(u, v, self.f, self.cx, self.cy, self.h, self.theta[i], self.psi[i], self.rho[i])

    def to_pixel(self, frame, X, Z):
        i = int(frame) - self.s0
        return water_to_pixel(X, Z, self.f, self.cx, self.cy, self.h, self.theta[i], self.psi[i], self.rho[i])
