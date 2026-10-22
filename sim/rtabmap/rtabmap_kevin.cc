// RTAB-Map (library 0.22, default parameters) offline driver for Kevin bench logs (bench/slam/run_rtabmap.py).
//   rtabmap_kevin MODE SEQDIR LIST CAMFILE OUT_PREFIX [ODOMFILE]
// MODE vo   : RTAB-Map's own RGB-D visual odometry (Odometry::create(defaults) = F2M) + RTAB-Map loop closure/graph
// MODE odom : external odometry (ODOMFILE per frame: t x y z qx qy qz qw  + 6 incremental std devs) + RTAB-Map
// CAMFILE: fx fy cx cy w h  then 12 numbers T_base_optical (3x4 row-major)
// The map update (Rtabmap::process) runs at 1 Hz like rtabmap_ros' default detection rate; odometry every frame.
// Output OUT_PREFIX_online.txt: per frame t + (mapCorrection * odom) as TUM (base frame); _graph.txt: final optimized nodes.
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/Signature.h>
#include <rtabmap/utilite/ULogger.h>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
using namespace rtabmap;
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage\n"); return 1; }
    std::string mode = argv[1], seq = argv[2], outp = argv[5];
    ULogger::setType(ULogger::kTypeConsole); ULogger::setLevel(ULogger::kWarning);
    double fx, fy, cx, cy; int w, h; float m[12];
    { std::ifstream f(argv[4]); f >> fx >> fy >> cx >> cy >> w >> h; for (int i = 0; i < 12; i++) f >> m[i]; }
    Transform Tbo(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11]);
    CameraModel cam("rtx_rs2", fx, fy, cx, cy, Tbo, 0, cv::Size(w, h));
    std::vector<double> T; std::vector<std::string> R, D;
    { std::ifstream f(argv[3]); std::string l; while (std::getline(f, l)) { if (l.empty()) continue; std::stringstream s(l); double t; std::string a, b; s >> t >> a >> b; T.push_back(t); R.push_back(a); D.push_back(b); } }
    std::vector<std::vector<double>> O;
    if (mode == "odom") { std::ifstream f(argv[6]); std::string l; while (std::getline(f, l)) { std::stringstream s(l); std::vector<double> v(14); for (auto &x : v) s >> x; O.push_back(v); } }
    ParametersMap P;                                  // defaults
    Odometry *odom = mode == "vo" ? Odometry::create(P) : nullptr;
    Rtabmap rt; rt.init(P, "");                       // in-RAM database
    std::ofstream on(outp + "_online.txt"); on << std::fixed;
    int lost = 0, nproc = 0; double tod = 0, tmap = 0; double lastProc = -1e9;
    Transform lastNodeOdom; cv::Mat accCov = cv::Mat::zeros(6, 6, CV_64FC1);
    for (size_t n = 0; n < T.size(); n++) {
        cv::Mat rgb = cv::imread(seq + "/" + R[n], cv::IMREAD_COLOR), dep = cv::imread(seq + "/" + D[n], cv::IMREAD_UNCHANGED);
        SensorData data(rgb, dep, cam, (int)n + 1, T[n]);
        Transform pose; cv::Mat cov;
        auto t0 = std::chrono::steady_clock::now();
        if (odom) {
            OdometryInfo info; pose = odom->process(data, &info);
            cov = info.reg.covariance.empty() ? cv::Mat::eye(6, 6, CV_64FC1) * 9999.0 : info.reg.covariance;
        } else {
            auto &v = O[n]; pose = Transform(v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            cov = cv::Mat::zeros(6, 6, CV_64FC1); for (int i = 0; i < 6; i++) cov.at<double>(i, i) = v[8 + i] * v[8 + i];
        }
        tod += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (pose.isNull()) { lost++; continue; }
        for (int i = 0; i < 6; i++) accCov.at<double>(i, i) += cov.at<double>(i, i);          // incremental link covariance between map updates
        if (T[n] - lastProc >= 1.0 - 1e-6) {
            auto t1 = std::chrono::steady_clock::now();
            rt.process(data, pose, accCov); nproc++; lastProc = T[n]; accCov = cv::Mat::zeros(6, 6, CV_64FC1);
            tmap += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        }
        Transform Tm = rt.getMapCorrection() * pose; Eigen::Quaternionf q = Tm.getQuaternionf();
        on << std::setprecision(6) << T[n] << " " << Tm.x() << " " << Tm.y() << " " << Tm.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
    on.close();
    std::map<int, Transform> poses; std::multimap<int, Link> links; rt.getGraph(poses, links, true, true);
    std::ofstream g(outp + "_graph.txt"); g << std::fixed; int nloop = 0;
    for (auto &l : links) if (l.second.type() == Link::kGlobalClosure || l.second.type() == Link::kLocalSpaceClosure || l.second.type() == Link::kLocalTimeClosure) nloop++;
    for (auto &p : poses) { if (p.first < 1 || p.first > (int)T.size()) continue; Eigen::Quaternionf q = p.second.getQuaternionf();
        g << std::setprecision(6) << T[p.first - 1] << " " << p.second.x() << " " << p.second.y() << " " << p.second.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n"; }
    printf("RTABMAP_KEVIN frames %zu lost %d map_updates %d nodes %zu loop_links %d odom_ms %.1f map_ms %.1f\n", T.size(), lost, nproc, poses.size(), nloop, 1e3 * tod / T.size(), 1e3 * tmap / std::max(1, nproc));
    rt.close(false); delete odom; return 0;
}
