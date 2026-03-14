"""
main.py - 30 fps loop: frames + tool calls; optional rerun. Entrypoint for anglerdroid v2.
"""

import os
import time
import argparse
import numpy as np
import cv2

import tools
import vision as vision_mod
import navigator
import goals

try:
    import rerun as rr
    HAS_RERUN = True
except (ImportError, TypeError):
    rr = None
    HAS_RERUN = False


TARGET_FPS = 30
LOOP_DT = 1.0 / TARGET_FPS
BUDGET_MS = 1000.0 / TARGET_FPS  # 33.33 ms at 30 fps
ATLAS_W, ATLAS_H = 640, 480


def main():
    parser = argparse.ArgumentParser(description="AnglerDroid v2 main loop")
    parser.add_argument("--no-wheelbase", action="store_true", help="Do not init real wheelbase (e.g. dev machine)")
    parser.add_argument("--no-rerun", action="store_true", help="Disable rerun logging")
    parser.add_argument("--rs1", default="", help="RealSense 1 serial")
    parser.add_argument("--rs2", default="", help="RealSense 2 serial")
    parser.add_argument("--rgb1", default="/dev/video0", help="RGB camera device (e.g. /dev/video0)")
    parser.add_argument("--no-show", action="store_true", help="Do not show vision atlas window")
    parser.add_argument("--gemini-key", default="", help="Gemini API key (or set GEMINI_KEY env)")
    parser.add_argument("--gemini-model", default="", help="Gemini model name (default: gemini-2.5-flash)")
    parser.add_argument("--auth-email", default="", help="Email allowed full control (default: everyone)")
    parser.add_argument("--google-client-id", default="", help="Google OAuth client ID for sign-in")
    parser.add_argument("--http-port", type=int, default=8080, help="HTTP server port")
    parser.add_argument("--ws-port", type=int, default=8081, help="WebSocket server port")
    args = parser.parse_args()

    gemini_key = args.gemini_key or os.environ.get("GEMINI_KEY", "")

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
    )
    u.start()

    tools.init(wheelbase_instance=wb, vision_instance=vis, ui_instance=u)

    show_ok = not args.no_show
    _OFF_CENTER = 320
    if show_ok:
        cv2.namedWindow("vision atlas", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("td_xoff", "vision atlas", _OFF_CENTER + vision_mod.TD_X_OFFSET, _OFF_CENTER * 2, lambda v: None)

    print("AnglerDroid v2 main loop (30 fps). Ctrl+C to quit.")
    print("  budget=%.1f ms/frame | every 30 frames: fps, avg process_ms, avg wait_ms" % BUDGET_MS)
    frame_id = 0
    last_report = time.monotonic()
    process_sum = 0.0
    wait_sum = 0.0
    try:
        while True:
            loop_start = time.monotonic()

            # Read tuning sliders and update vision module
            if show_ok:
                vision_mod.TD_X_OFFSET = cv2.getTrackbarPos("td_xoff", "vision atlas") - _OFF_CENTER

            # Get latest atlas only (no frame copies)
            atlas, ts = tools.get_atlas()
            if show_ok and atlas is not None:
                try:
                    display = cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR)
                    display = cv2.resize(display, (ATLAS_W * 2, ATLAS_H * 2), interpolation=cv2.INTER_NEAREST)
                    fw_x = vision_mod.TD_X_OFFSET + vision_mod.FW_TD_X_DELTA
                    cv2.putText(display, "td_x=%d  fw_x=%d  floor=%.0fcm" % (
                        vision_mod.TD_X_OFFSET, fw_x,
                        float(vision_mod.TD_FLOOR_CLIP) * 100),
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    navigator.draw_overlay(display)
                    cv2.imshow("vision atlas", display)
                    cv2.waitKey(1)
                except cv2.error:
                    show_ok = False
                    print("vision: display not available, continuing without window")
            # Send atlas to UI at ~1 fps for browser + Gemini
            if frame_id % TARGET_FPS == 0 and atlas is not None:
                u.send_atlas(atlas)


            if HAS_RERUN and not args.no_rerun:
                rr.set_time_seconds("capture", ts)
                rr.log("vision/atlas", rr.Image(atlas))

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
                        navigator.clear_goal()
                        tools.twist_for(
                            cargs.get("forward_mps", 0), cargs.get("angular_rads", 0),
                            duration_secs=cargs.get("duration_secs", 2.0),
                            ramp_in_secs=cargs.get("ramp_in_secs", 1.0),
                            ramp_out_secs=cargs.get("ramp_out_secs", 1.0),
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
                    elif name == "set_visual_goals":
                        landmarks = cargs.get("landmarks", [])
                        if landmarks:
                            goals.set_goals(landmarks)
                            print("goals: set %d landmarks: %s" % (len(landmarks), landmarks))
                    elif name == "goal_reached":
                        nxt = goals.advance()
                        if nxt:
                            print("goals: advanced → %s" % nxt)
                        else:
                            print("goals: all landmarks reached")
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
            if frame_id % 30 == 0:
                now = time.monotonic()
                elapsed = now - last_report
                actual_fps = 30.0 / elapsed if elapsed > 0 else 0
                avg_process = process_sum / 30.0
                avg_wait = wait_sum / 30.0
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
        if show_ok:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        vis.stop()
        u.stop()
        navigator.clear_goal()
        if wb:
            wb.shutdown()
        print("main: shutdown complete")


if __name__ == "__main__":
    main()
