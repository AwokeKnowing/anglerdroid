"""
main.py - 30 fps loop: frames + tool calls; optional rerun. Entrypoint for anglerdroid v2.
"""

import os
import time
import argparse
import tools
import vision as vision_mod
import navigator

try:
    import rerun as rr
    HAS_RERUN = True
except (ImportError, TypeError):
    rr = None
    HAS_RERUN = False


TARGET_FPS = 30
LOOP_DT = 1.0 / TARGET_FPS
BUDGET_MS = 1000.0 / TARGET_FPS  # 33.33 ms at 30 fps


def main():
    parser = argparse.ArgumentParser(description="AnglerDroid v2 main loop")
    parser.add_argument("--no-wheelbase", action="store_true", help="Do not init real wheelbase (e.g. dev machine)")
    parser.add_argument("--no-rerun", action="store_true", help="Disable rerun logging")
    parser.add_argument("--rs1", default="", help="RealSense 1 serial")
    parser.add_argument("--rs2", default="", help="RealSense 2 serial")
    parser.add_argument("--rgb1", default="/dev/video0", help="RGB camera device (e.g. /dev/video0)")
    parser.add_argument("--gemini-key", default="", help="Gemini API key (or set GEMINI_KEY env)")
    parser.add_argument("--gemini-model", default="", help="Gemini model name (default: gemini-2.5-flash)")
    parser.add_argument("--auth-email", default="", help="Email allowed full control (default: everyone)")
    parser.add_argument("--google-client-id", default="", help="Google OAuth client ID for sign-in")
    parser.add_argument("--http-port", type=int, default=8080, help="HTTP server port")
    parser.add_argument("--ws-port", type=int, default=8081, help="WebSocket server port")
    parser.add_argument("--brain-url", default="",
                        help="Brain server URL (e.g. http://192.168.1.50:8090). Uses local vLLM instead of Gemini.")
    args = parser.parse_args()

    gemini_key = args.gemini_key or os.environ.get("GEMINI_KEY", "")
    brain_url = args.brain_url or os.environ.get("BRAIN_URL", "")

    # Rerun
    if HAS_RERUN and not args.no_rerun:
        rr.init("anglerdroid_v2")
        rr.connect()

    # Wheelbase (real or None)
    wb = None
    if not args.no_wheelbase:
        try:
            import wheelbase as wb_mod
            wb = wb_mod.WheelBase()
        except Exception as e:
            print(f"wheelbase not started: {e}")

    # Vision
    from vision import Vision
    vis = Vision(
        rs1_serial=args.rs1 or "0",
        rs2_serial=args.rs2 or "0",
        rgb1_device_id=args.rgb1,
    )
    vis.set_wheelbase(wb)
    vis.start()

    # UI (web server + Gemini)
    from ui import UI
    u = UI(
        gemini_key=gemini_key,
        gemini_model=args.gemini_model,
        auth_email=args.auth_email,
        google_client_id=args.google_client_id,
        http_port=args.http_port,
        ws_port=args.ws_port,
        brain_url=brain_url,
    )
    u.start()

    tools.init(wheelbase_instance=wb, vision_instance=vis, ui_instance=u)

    print("AnglerDroid v2 main loop (30 fps). Ctrl+C to quit.")
    print("  budget=%.1f ms/frame | every 30 frames: fps, avg process_ms, avg wait_ms" % BUDGET_MS)
    frame_id = 0
    last_report = time.monotonic()
    process_sum = 0.0
    wait_sum = 0.0
    try:
        while True:
            loop_start = time.monotonic()

            # Get latest atlas only (no frame copies)
            atlas, ts = tools.get_atlas()
            if atlas is not None:
                u.send_atlas(atlas)


            if HAS_RERUN and not args.no_rerun:
                rr.set_time_seconds("capture", ts)
                rr.log("vision/atlas", rr.Image(atlas))

            # Propagate debug flags from UI to vision
            vis.debug_depth = u.debug_flags.get("depth", False)

            # Safety override — directional scaling (fwd / bwd / angular independent)
            if wb is not None:
                wb.set_safety_scales(vis.safety_fwd_scale, vis.safety_bwd_scale,
                                     vis.safety_ang_scale)

            # Gamepad + watchdog
            wb = tools.get_wheelbase()
            gamepad_active = False
            if wb is not None:
                wb._check_gamepad_health()
                left_tps = 0.0
                right_tps = 0.0
                if wb.gamepad is not None:
                    vels = wb.gamepad.diffDrive()
                    left_norm = vels.get("left", 0.0)
                    right_norm = vels.get("right", 0.0)
                    if abs(left_norm) < 0.08:
                        left_norm = 0.0
                    if abs(right_norm) < 0.08:
                        right_norm = 0.0
                    left_tps = left_norm * 0.5
                    right_tps = right_norm * 0.5
                gamepad_active = abs(left_tps) > 0 or abs(right_tps) > 0

                if gamepad_active:
                    wb.cancel_twist_for()
                    navigator.clear_goal()
                    tools.set_wheel_vels(left_tps, right_tps)
                elif not wb.is_twist_for_active():
                    twist = navigator.compute_twist(atlas) if atlas is not None else None
                    if twist is not None:
                        tools.twist(twist[0], twist[1])
                    else:
                        tools.set_wheel_vels(0.0, 0.0)

            # Tool calls from agent (only act if gamepad is idle)
            if not gamepad_active:
                pending = tools.get_pending_tool_calls()
                for call in pending:
                    name = call.get("name")
                    cargs = call.get("args", {})
                    if name == "twist_for":
                        fwd = cargs.get("forward_mps", 0)
                        ang = cargs.get("angular_rads", 0)
                        dur = cargs.get("duration_secs", 2.0)
                        sf = wb._safety_fwd if fwd > 0 else wb._safety_bwd if fwd < 0 else 1.0
                        print("exec: twist_for(%.2f, %.2f, %.1fs) safety_fwd=%.2f safety_ang=%.2f" % (
                            fwd, ang, dur, sf, wb._safety_ang))
                        navigator.clear_goal()
                        tools.twist_for(
                            fwd, ang,
                            duration_secs=dur,
                            ramp_in_secs=cargs.get("ramp_in_secs", 0.0),
                            ramp_out_secs=cargs.get("ramp_out_secs", 0.0),
                        )
                    elif name == "stop":
                        if wb:
                            wb.cancel_twist_for()
                        navigator.clear_goal()
                        tools.stop()
                    elif name == "navigate":
                        hdg = cargs.get("heading_deg")
                        if hdg is not None:
                            navigator.set_goal(float(hdg))
                        else:
                            navigator.clear_goal()
                    elif name == "twist":
                        tools.twist(cargs.get("forward_mps", 0), cargs.get("angular_rads", 0))
                    elif name == "set_wheel_vels":
                        tools.set_wheel_vels(cargs.get("left_tps", 0), cargs.get("right_tps", 0))

            # Throttle to 30 fps
            process_sec = time.monotonic() - loop_start
            process_ms = process_sec * 1000.0
            sleep_time = LOOP_DT - process_sec
            if sleep_time > 0:
                time.sleep(sleep_time)
            wait_ms = sleep_time * 1000.0 if sleep_time > 0 else 0.0
            process_sum += process_ms
            wait_sum += wait_ms
            frame_id += 1
            if frame_id % 90 == 0:
                now = time.monotonic()
                elapsed = now - last_report
                actual_fps = 90.0 / elapsed if elapsed > 0 else 0
                avg_process = process_sum / 90.0
                avg_wait = wait_sum / 90.0
                process_sum = 0.0
                wait_sum = 0.0
                last_report = now
                nav_info = ""
                hdg = navigator.get_goal()
                if hdg is not None:
                    nav_info = "  nav=%.0f°" % hdg
                print("  fps=%.1f  process=%.1f ms  wait=%.1f ms  (budget %.1f ms)%s" % (
                    actual_fps, avg_process, avg_wait, BUDGET_MS, nav_info))
    except KeyboardInterrupt:
        pass
    finally:
        vis.stop()
        u.stop()
        navigator.clear_goal()
        if wb:
            wb.shutdown()
        print("main: shutdown complete")


if __name__ == "__main__":
    main()
