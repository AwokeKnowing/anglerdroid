# Third-party SLAM candidates on Kevin sensor logs (offline, i777, sim stopped)
Exports: `python -m emu.tools.slamlog.export_rgbd ~/slamlogs/L3 rs2 ~/slamlogs/L3x_rs2` (rgb/*.jpg, depth/*.png16 mm in colour, K.txt, list.txt).
Score any camera trajectory: `python -m emu.tools.slamlog.ext_eval <log> rs2 <export> <poses> <ace0|tum|tum@30> [out.json]`.
venv ~/slamvenv (torch 2.10 cu128, system nvcc 12.0). All CUDA extensions built for sm_86 only (the upstream setup.py
gencode lists of 5-6 archs were why lietorch / backends took >1 h to compile; sm_86-only lietorch builds in ~12 min).
- DROID-SLAM (~/slamcand/DROID-SLAM, droid.pth): `python droid_rgbd.py <export> <out.txt>` (copy of this dir's droid_rgbd.py;
  RGB-D, the repo's ETH3D RGB-D eval settings). torch_scatter built from thirdparty/pytorch_scatter.
- MASt3R-SLAM (~/slamcand/MASt3R-SLAM): `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python main.py --dataset <export>/png
  --config config/base.yaml --calib config/kevin_rs2.yaml --no-viz --save-as <tag>` (<export>/png = *.png symlinks to the
  jpgs, RGBFiles loader; trajectory = keyframes only, stamps = index/30 -> ext_eval fmt tum@30).  Local patches for torch 2.10:
  gn_kernels.cu linalg_norm -> .norm(), matching_kernels.cu .type() -> .scalar_type(); pyrealsense2 / imgui imports optional;
  needs faiss-cpu and asmk (cython/hamming.c regenerated with Cython 3).
- ACE0 (~/slamcand/acezero): `ace_zero.py '<export>/rgb/*.jpg' <out> --depth_files '<export>/depth/*.png' --depth_use_always True
  --use_external_focal_length 625.22 --refine_calibration False` (LD_LIBRARY_PATH=~/slamcand/mm/cv/lib, PATH=~/slamvenv/bin).
  poses_final.txt is world-to-camera; ext_eval keeps frames with confidence >= 500 (ACE0's registration threshold).
