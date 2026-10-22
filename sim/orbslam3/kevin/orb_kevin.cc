// ORB-SLAM3 offline driver for Kevin bench logs (bench/slam/run_orb_rtx.py): stock ORB-SLAM3 System, viewer off,
// no real-time sleep.  Modes: rgbd | rgbd_imu | mono | mono_imu.
//   orb_kevin MODE VOC SETTINGS SEQDIR LIST OUT_PREFIX [IMU_CSV]
// LIST lines: t rgb_rel [depth_rel]      IMU_CSV lines: t gx gy gz ax ay az  (IMU body frame = settings IMU.T_b_c1 body)
// Writes OUT_PREFIX_online.txt (per frame: t + Twc as TUM, only when tracking OK) and OUT_PREFIX_final_*.txt.
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <System.h>
using namespace std;
int main(int argc, char **argv) {
    if (argc < 7) { cerr << "usage: orb_kevin MODE VOC SETTINGS SEQDIR LIST OUT_PREFIX [IMU_CSV]" << endl; return 1; }
    string mode = argv[1], seq = argv[4], outp = argv[6];
    bool imu = mode.find("imu") != string::npos, rgbd = mode.rfind("rgbd", 0) == 0;
    vector<double> T; vector<string> R, D;
    { ifstream f(argv[5]); string l; while (getline(f, l)) { if (l.empty()) continue; stringstream s(l); double t; string a, b; s >> t >> a; if (rgbd) s >> b; T.push_back(t); R.push_back(a); D.push_back(b); } }
    vector<ORB_SLAM3::IMU::Point> I;
    if (imu) { ifstream f(argv[7]); string l; while (getline(f, l)) { if (l.empty() || l[0] == 't') continue; for (auto &c : l) if (c == ',') c = ' ';
        stringstream s(l); double t, gx, gy, gz, ax, ay, az; s >> t >> gx >> gy >> gz >> ax >> ay >> az; I.emplace_back(ax, ay, az, gx, gy, gz, t); } }
    ORB_SLAM3::System::eSensor sens = mode == "rgbd" ? ORB_SLAM3::System::RGBD : mode == "rgbd_imu" ? ORB_SLAM3::System::IMU_RGBD
                                     : mode == "mono" ? ORB_SLAM3::System::MONOCULAR : ORB_SLAM3::System::IMU_MONOCULAR;
    ORB_SLAM3::System SLAM(argv[2], argv[3], sens, false);
    ofstream on(outp + "_online.txt"); on << fixed; size_t ki = 0; double ttot = 0; int lost = 0;
    // first frame: skip IMU before it
    while (imu && ki < I.size() && I[ki].t <= T[0]) ki++;
    for (size_t n = 0; n < T.size(); n++) {
        cv::Mat im = cv::imread(seq + "/" + R[n], cv::IMREAD_UNCHANGED), dm;
        if (rgbd) dm = cv::imread(seq + "/" + D[n], cv::IMREAD_UNCHANGED);
        vector<ORB_SLAM3::IMU::Point> v;
        if (imu && n > 0) while (ki < I.size() && I[ki].t <= T[n]) v.push_back(I[ki++]);
        auto t0 = chrono::steady_clock::now(); Sophus::SE3f Tcw;
        if (rgbd) Tcw = SLAM.TrackRGBD(im, dm, T[n], v); else Tcw = SLAM.TrackMonocular(im, T[n], v);
        ttot += chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        int st = SLAM.GetTrackingState();
        if (st == 2 /*OK*/ || st == 3 /*RECENTLY_LOST, IMU-predicted*/) {
            Sophus::SE3f Twc = Tcw.inverse(); Eigen::Quaternionf q = Twc.unit_quaternion(); Eigen::Vector3f p = Twc.translation();
            on << setprecision(6) << T[n] << " " << setprecision(7) << p.x() << " " << p.y() << " " << p.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        } else lost++;
    }
    on.close(); SLAM.Shutdown();
    if (!imu && rgbd) SLAM.SaveTrajectoryTUM(outp + "_final_tum_cam.txt");
    else SLAM.SaveTrajectoryEuRoC(outp + "_final_euroc.txt");
    SLAM.SaveKeyFrameTrajectoryTUM(outp + "_final_kf.txt");
    cout << "ORB_KEVIN frames " << T.size() << " lost " << lost << " track_ms " << 1e3 * ttot / T.size() << endl;
    return 0;
}
