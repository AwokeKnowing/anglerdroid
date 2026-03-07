"""
main.py - 30 fps loop: frames + tool calls; optional rerun. Entrypoint for anglerdroid v2.
"""

import time
import argparse
import numpy as np

import tools

try:
    import rerun as rr
    HAS_RERUN = True
except (ImportError, TypeError):
    rr = None
    HAS_RERUN = False


TARGET_FPS = 30
LOOP_DT = 1.0 / TARGET_FPS


def main():
    parser = argparse.ArgumentParser(description="AnglerDroid v2 main loop")
    parser.add_argument("--no-wheelbase", action="store_true", help="Do not init real wheelbase (e.g. dev machine)")
    parser.add_argument("--no-rerun", action="store_true", help="Disable rerun logging")
    parser.add_argument("--rs1", default="", help="RealSense 1 serial")
    parser.add_argument("--rs2", default="", help="RealSense 2 serial")
    parser.add_argument("--rgb1", default="/dev/video0", help="RGB camera device (e.g. /dev/video0)")
    args = parser.parse_args()

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

    # UI (stub)
    from ui import UI
    u = UI()
    u.start()

    tools.init(wheelbase_instance=wb, vision_instance=vis, ui_instance=u)

    print("AnglerDroid v2 main loop (30 fps). Ctrl+C to quit.")
    t0 = time.monotonic()
    frame_id = 0
    try:
        while True:
            loop_start = time.monotonic()

            # Get latest vision (safe read)
            frames, atlas, ts = tools.get_frames()
            if HAS_RERUN and not args.no_rerun:
                rr.set_time_seconds("capture", ts)
                rr.log("vision/atlas", rr.Image(atlas))

            # Get pending tool calls from agent (e.g. LLM) and execute locally
            pending = tools.get_pending_tool_calls()
            for call in pending:
                # Example: {"name": "set_wheel_vels", "args": {"left_tps": 0.0, "right_tps": 0.0}}
                name = call.get("name")
                args = call.get("args", {})
                if name == "set_wheel_vels":
                    tools.set_wheel_vels(args.get("left_tps", 0), args.get("right_tps", 0))
                elif name == "stop":
                    tools.stop()
                elif name == "twist":
                    tools.twist(args.get("forward_mps", 0), args.get("angular_rads", 0))

            # User text (e.g. for next LLM turn)
            user_text = tools.get_user_text()
            if user_text:
                pass  # Feed to agent when integrated

            # Throttle to 30 fps
            elapsed = time.monotonic() - loop_start
            sleep_time = LOOP_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            frame_id += 1
    except KeyboardInterrupt:
        pass
    finally:
        vis.stop()
        u.stop()
        if wb:
            wb.shutdown()
        print("main: shutdown complete")


if __name__ == "__main__":
    main()
