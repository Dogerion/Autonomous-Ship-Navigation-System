"""
Live visualization of a single course-keeping episode.

The environment is heading-only (state = [psi, r, delta, integral_psi]), so there is no
true x/y position. For the top-down view we reconstruct a path by driving the ship forward
at a constant speed along its actual heading, with the target course along the
x-axis. Because the controller only regulates heading (no cross-track feedback), the boat
straightens to run parallel to the course line rather than rejoining it -- this is expected.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_trajectory(traj, *, dt, rudder_bound, agent_name,
                       forward_speed=3.0, save_path=None, show=True, sysid=False, fps=20):
    """
    Animate one episode as a 2D top-down boat path plus synced psi/delta/r time-series.

    Args:
        traj: list of per-step dicts with keys {psi, r, delta, integral_psi, reward, K, T}
              (and K_hat/T_hat when produced by the SysID+MPC agent).
        dt: environment time step (seconds), used for the time axis and path integration.
        rudder_bound: rudder angle limit (radians), drawn as guide lines on the delta plot.
        agent_name: label shown in the title (e.g. "ppo", "sysid_mpc").
        forward_speed: synthetic constant surge speed used only to reconstruct the 2D path.
        save_path: if given, write the animation to this path as an animated GIF.
        show: if True, also open a live interactive window.
        sysid: if True, add two extra panels plotting the GRU's predicted K/T against the true K&T.
        fps: playback frames per second (decoupled from the sim dt); higher = faster animation.
    """
    if not traj:
        raise ValueError("Empty trajectory -- nothing to animate.")

    psi = np.array([s["psi"] for s in traj], dtype=float)
    r = np.array([s["r"] for s in traj], dtype=float)
    delta = np.array([s["delta"] for s in traj], dtype=float)
    n = len(traj)
    t = np.arange(n) * dt
    K = traj[0].get("K")
    T = traj[0].get("T")

    # Only show the SysID diagnostic panels if requested AND the estimates are present.
    show_kt = sysid and "K_hat" in traj[0]
    if show_kt:
        K_hat = np.array([s["K_hat"] for s in traj], dtype=float)
        T_hat = np.array([s["T_hat"] for s in traj], dtype=float)

    # Reconstruct a 2D path: forward speed along the actual heading (= heading error psi,
    # since the target course is heading 0 along the +x axis).
    x = np.concatenate([[0.0], np.cumsum(forward_speed * np.cos(psi) * dt)])
    y = np.concatenate([[0.0], np.cumsum(forward_speed * np.sin(psi) * dt)])

    n_right = 5 if show_kt else 3
    fig = plt.figure(figsize=(12, 8 if show_kt else 6))
    gs = fig.add_gridspec(n_right, 2, width_ratios=[1.4, 1.0])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_psi = fig.add_subplot(gs[0, 1])
    ax_del = fig.add_subplot(gs[1, 1])
    ax_r = fig.add_subplot(gs[2, 1], sharex=ax_psi)

    title = f"Course Correction — {agent_name}"
    if K is not None and T is not None:
        title += f"   (K={K:.3f}, T={T:.2f})"
    fig.suptitle(title, fontsize=13)

    # --- Left: top-down map ---
    pad_x = max(1.0, 0.05 * (x.max() - x.min() + 1e-9))
    pad_y = max(1.0, 0.20 * (np.abs(y).max() + 1e-9))
    ax_map.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax_map.set_ylim(y.min() - pad_y, y.max() + pad_y)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_title("Top-down View", fontsize=10)
    ax_map.set_xlabel("x axis (m)")
    ax_map.set_ylabel("y axis (m)")
    x_lo, x_hi = ax_map.get_xlim()
    # Target-course reference: a horizontal line carried at the vessel's current y, so the
    # desired heading direction stays next to the boat (arrow aligns with it when on course).
    (course_line,) = ax_map.plot([], [], color="tab:green", ls="--", lw=1.5, label="target course")
    (path_line,) = ax_map.plot([], [], color="tab:blue", lw=2, label="ship path")
    boat = ax_map.quiver([], [], [], [], color="tab:red", scale=18,
                         width=0.008, label="ship heading")
    ax_map.legend(loc="upper right", fontsize=8)

    # Time Series Plot of the heading error, rudder command, and yaw rate.
    def _setup(ax, series, ylabel, color):
        ax.set_xlim(0, max(t[-1], dt))
        lo, hi = float(np.min(series)), float(np.max(series))
        m = 0.1 * (hi - lo) + 1e-3
        ax.set_ylim(lo - m, hi + m)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)
        (line,) = ax.plot([], [], color=color, lw=1.8)
        return line

    # Setup the K/T prediction panels if requested.
    def _setup_pred(ax, pred, true_val, name, color):
        ax.set_xlim(0, max(t[-1], dt))
        vals = np.concatenate([pred, [true_val]])
        lo, hi = float(vals.min()), float(vals.max())
        m = 0.1 * (hi - lo) + 1e-3
        ax.set_ylim(lo - m, hi + m)
        ax.set_ylabel(f"{name} estimate", fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(true_val, color="k", ls="--", lw=1.2, alpha=0.8, label=f"true {name}")
        (line,) = ax.plot([], [], color=color, lw=1.8, label=f"predicted {name}")
        ax.legend(loc="best", fontsize=7)
        return line
    
    ax_psi.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    psi_line = _setup(ax_psi, psi, "heading err ψ (rad)", "tab:blue")
    del_line = _setup(ax_del, delta, "rudder δ (rad)", "tab:orange")
    ax_del.axhline(rudder_bound, color="r", ls=":", lw=1, alpha=0.7)
    ax_del.axhline(-rudder_bound, color="r", ls=":", lw=1, alpha=0.7)
    r_line = _setup(ax_r, r, "yaw rate r (rad/s)", "tab:purple")

    kt_lines = []
    if show_kt:
        ax_K = fig.add_subplot(gs[3, 1], sharex=ax_psi)
        ax_T = fig.add_subplot(gs[4, 1], sharex=ax_psi)
        K_line = _setup_pred(ax_K, K_hat, K, "K", "tab:cyan")
        T_line = _setup_pred(ax_T, T_hat, T, "T", "tab:brown")
        kt_lines = [K_line, T_line]
        ax_T.set_xlabel("time (s)")
    else:
        ax_r.set_xlabel("time (s)")

    def update(frame):
        i = frame + 1  # number of points to show
        path_line.set_data(x[:i + 1], y[:i + 1])
        # Boat marker at the current position, pointing along the current heading.
        boat.set_offsets([[x[i], y[i]]])
        boat.set_UVC(np.cos(psi[frame]), np.sin(psi[frame]))
        # Course line carried at the vessel's current y (target heading = horizontal / +x).
        course_line.set_data([x_lo, x_hi], [y[i], y[i]])
        psi_line.set_data(t[:i], psi[:i])
        del_line.set_data(t[:i], delta[:i])
        r_line.set_data(t[:i], r[:i])
        if show_kt:
            kt_lines[0].set_data(t[:i], K_hat[:i])
            kt_lines[1].set_data(t[:i], T_hat[:i])
        return course_line, path_line, boat, psi_line, del_line, r_line, *kt_lines

    anim = FuncAnimation(fig, update, frames=n, interval=max(1, int(1000 / fps)),
                         blit=False, repeat=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is not None:
        anim.save(save_path, writer="pillow", fps=fps)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return anim
