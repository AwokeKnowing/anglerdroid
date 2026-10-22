// RTAB-Map (library 0.22) STREAMING driver for long runs (bench/slam/life24.py): the harness pipes frames in as the sim
// produces them, so RTAB-Map runs online beside slam2 on the same sensor stream.
//   rtabmap_stream CAMFILE DBPATH [Param=value ...]       (DBPATH "" = in-RAM database; parameters default otherwise)
// CAMFILE: fx fy cx cy w h  then 12 numbers T_base_optical (3x4 row-major)
// stdin protocol (one map update per frame; the harness sends at RTAB-Map's default 1 Hz detection rate):
//   "F t x y yaw sx sy syaw\n" + w*h*3 BGR bytes + w*h*2 uint16 depth (mm, registered to the colour camera)
//        external odometry pose (base, planar) + accumulated std devs since the previous update
//   -> "R t last_id wm loop_id prox_id ms corr_x corr_y corr_yaw\n"
//   "Q\n" -> "G n_nodes n_links\n", n lines "N id x y yaw", links "L from to type x y yaw", "END\n"   (optimized graph)
//   "X\n" -> close (DB flushed) and "BYE\n"
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/Statistics.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/utilite/ULogger.h>
#include <cstdio>
#include <fstream>
#include <chrono>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
using namespace rtabmap;
static bool readn(void *p, size_t n) { return fread(p, 1, n, stdin) == n; }
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: rtabmap_stream CAMFILE DBPATH [k=v...]\n"); return 1; }
    ULogger::setType(ULogger::kTypeConsole); ULogger::setLevel(ULogger::kError);
    double fx, fy, cx, cy; int w, h; float m[12];
    { std::ifstream f(argv[1]); f >> fx >> fy >> cx >> cy >> w >> h; for (int i = 0; i < 12; i++) f >> m[i]; }
    Transform Tbo(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11]);
    CameraModel cam("sim_rs2_color", fx, fy, cx, cy, Tbo, 0, cv::Size(w, h));
    ParametersMap P;
    for (int i = 3; i < argc; i++) { std::string s = argv[i]; size_t k = s.find('='); if (k != std::string::npos) P[s.substr(0, k)] = s.substr(k + 1); }
    std::string db = argv[2];
    Rtabmap rt; rt.init(P, db);
    setvbuf(stdout, nullptr, _IOLBF, 0);
    char line[512]; int n = 0;
    while (fgets(line, sizeof line, stdin)) {
        if (line[0] == 'F') {
            double t, x, y, yaw, sx, sy, sa;
            if (sscanf(line + 1, "%lf %lf %lf %lf %lf %lf %lf", &t, &x, &y, &yaw, &sx, &sy, &sa) != 7) { fprintf(stderr, "bad F\n"); return 2; }
            cv::Mat rgb(h, w, CV_8UC3), dep(h, w, CV_16UC1);
            if (!readn(rgb.data, (size_t)w * h * 3) || !readn(dep.data, (size_t)w * h * 2)) { fprintf(stderr, "short frame\n"); return 3; }
            SensorData data(rgb, dep, cam, ++n, t);
            Transform pose(x, y, 0, 0, 0, yaw);
            cv::Mat cov = cv::Mat::zeros(6, 6, CV_64FC1);
            double sv[6] = {sx, sy, 1e-4, 1e-4, 1e-4, sa};
            for (int i = 0; i < 6; i++) cov.at<double>(i, i) = sv[i] * sv[i];
            auto t0 = std::chrono::steady_clock::now();
            int exc = 0;
            try { rt.process(data, pose, cov); }
            catch (const std::exception &ex) {       // UASSERT inside the library: log, count, keep the session (reported by the harness)
                exc = 1; fprintf(stderr, "EXC frame %d t %.3f pose %.4f %.4f %.5f sd %g %g %g: %s\n", n, t, x, y, yaw, sx, sy, sa, ex.what());
                char fn[256]; snprintf(fn, sizeof fn, "exc_%d_rgb.png", n); cv::imwrite(fn, rgb); snprintf(fn, sizeof fn, "exc_%d_depth.png", n); cv::imwrite(fn, dep);
            }
            double ms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const Statistics &st = rt.getStatistics(); Transform C = rt.getMapCorrection();
            auto stv = [&](const std::string &k) { auto it = st.data().find(k); return it == st.data().end() ? -1.f : it->second; };
            printf("R %.4f %d %d %d %d %.2f %.5f %.5f %.6f %d %.4f %.0f %.0f %.0f\n", t, rt.getLastLocationId(), rt.getWMSize(), st.loopClosureId(), st.proximityDetectionId(), ms, C.x(), C.y(), C.theta(), exc,
                   stv(Statistics::kLoopHighest_hypothesis_value()), stv(Statistics::kLoopHighest_hypothesis_id()), stv(Statistics::kLoopRejectedHypothesis()), stv(Statistics::kLoopVisual_inliers()));
        } else if (line[0] == 'Q') {
            std::map<int, Transform> poses; std::multimap<int, Link> links; rt.getGraph(poses, links, true, true);
            printf("G %zu %zu\n", poses.size(), links.size());
            for (auto &p : poses) printf("N %d %.5f %.5f %.6f\n", p.first, p.second.x(), p.second.y(), p.second.theta());
            for (auto &l : links) { const Transform &T = l.second.transform(); printf("L %d %d %d %.5f %.5f %.6f\n", l.second.from(), l.second.to(), (int)l.second.type(), T.x(), T.y(), T.theta()); }
            printf("END\n");
        } else if (line[0] == 'X') { rt.close(true); printf("BYE\n"); return 0; }
    }
    rt.close(true); return 0;
}
