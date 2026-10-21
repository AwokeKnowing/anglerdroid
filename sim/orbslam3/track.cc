// ORB-SLAM3 stereo tracker for Kevin → i777.
// Listens TCP length-prefixed IR pairs; publishes JSON pose on UDP.
// Never talks to the camera. Capture on Orin stays a memcpy + this process.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <MapPoint.h>
#include <System.h>

namespace {

volatile sig_atomic_t g_run = 1;

void on_sig(int) { g_run = 0; }

bool read_exact(int fd, void* buf, size_t n) {
    auto* p = static_cast<uint8_t*>(buf);
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::recv(fd, p + got, n - got, 0);
        if (r == 0) return false;
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        got += static_cast<size_t>(r);
    }
    return true;
}

int listen_tcp(uint16_t port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = INADDR_ANY;
    a.sin_port = htons(port);
    if (bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) < 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 1) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int udp_sock() {
    return ::socket(AF_INET, SOCK_DGRAM, 0);
}

void send_json(int fd, const sockaddr_in& dest, const std::string& s) {
    sendto(fd, s.data(), s.size(), 0,
           reinterpret_cast<const sockaddr*>(&dest), sizeof(dest));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: orbslam3_track voc.txt settings.yaml "
                     "[listen_port=5571] [pose_host=127.0.0.1] [pose_port=5572]\n";
        return 1;
    }
    const std::string voc = argv[1];
    const std::string yaml = argv[2];
    const uint16_t listen_port = (argc > 3) ? static_cast<uint16_t>(std::stoi(argv[3])) : 5571;
    const char* pose_host = (argc > 4) ? argv[4] : "127.0.0.1";
    const uint16_t pose_port = (argc > 5) ? static_cast<uint16_t>(std::stoi(argv[5])) : 5572;
    const char* viewer_env = std::getenv("ORBSLAM3_VIEWER");
    const bool viewer = !(viewer_env && std::string(viewer_env) == "0");

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    // RS1 top-down laser stays on for obstacle depth. RS2 is pitched at the
    // floor, so that speckle is bright in stereo IR and body-fixed as Kevin
    // rolls. Median-kill the dots here (i777), never by turning RS1 off.
    int speckle_k = 5;
    if (const char* envk = std::getenv("ORBSLAM3_SPECKLE_K")) {
        speckle_k = std::atoi(envk);
    }
    if (speckle_k > 0 && (speckle_k % 2) == 0) speckle_k += 1;

    std::cout << "orbslam3_track: voc=" << voc << " yaml=" << yaml
              << " tcp=:" << listen_port << " pose_udp=" << pose_host << ":" << pose_port
              << " viewer=" << (viewer ? "on" : "off")
              << " speckle_k=" << speckle_k << std::endl;

    ORB_SLAM3::System SLAM(voc, yaml, ORB_SLAM3::System::STEREO, viewer);

    int lfd = listen_tcp(listen_port);
    if (lfd < 0) {
        std::cerr << "orbslam3_track: listen failed on " << listen_port << std::endl;
        SLAM.Shutdown();
        return 1;
    }
    int ufd = udp_sock();
    sockaddr_in dest{};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(pose_port);
    inet_pton(AF_INET, pose_host, &dest.sin_addr);

    std::cout << "orbslam3_track: waiting for Kevin stereo TCP…" << std::endl;
    int cfd = -1;
    while (g_run && cfd < 0) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(lfd, &rfds);
        timeval tv{1, 0};
        int sel = select(lfd + 1, &rfds, nullptr, nullptr, &tv);
        if (sel > 0 && FD_ISSET(lfd, &rfds)) {
            cfd = accept(lfd, nullptr, nullptr);
        }
    }
    if (cfd < 0) {
        close(lfd);
        SLAM.Shutdown();
        return 0;
    }
    std::cout << "orbslam3_track: client connected" << std::endl;

    std::vector<uint8_t> body;
    uint64_t nframe = 0;
    while (g_run) {
        uint32_t nbytes = 0;
        if (!read_exact(cfd, &nbytes, 4)) break;
        if (nbytes < 8 + 2 + 2 + 20 || nbytes > 8 * 1024 * 1024) {
            std::cerr << "orbslam3_track: bad frame size " << nbytes << std::endl;
            break;
        }
        body.resize(nbytes);
        if (!read_exact(cfd, body.data(), nbytes)) break;

        uint64_t ts_ns = 0;
        uint16_t w = 0, h = 0;
        float fx, fy, cx, cy, baseline;
        std::memcpy(&ts_ns, body.data(), 8);
        std::memcpy(&w, body.data() + 8, 2);
        std::memcpy(&h, body.data() + 10, 2);
        std::memcpy(&fx, body.data() + 12, 4);
        std::memcpy(&fy, body.data() + 16, 4);
        std::memcpy(&cx, body.data() + 20, 4);
        std::memcpy(&cy, body.data() + 24, 4);
        std::memcpy(&baseline, body.data() + 28, 4);
        const size_t pix = static_cast<size_t>(w) * static_cast<size_t>(h);
        if (static_cast<size_t>(32) + 2 * pix != nbytes) {
            std::cerr << "orbslam3_track: size mismatch w=" << w << " h=" << h
                      << " nbytes=" << nbytes << std::endl;
            break;
        }
        cv::Mat imL(h, w, CV_8UC1, body.data() + 32);
        cv::Mat imR(h, w, CV_8UC1, body.data() + 32 + pix);
        cv::Mat left = imL.clone();
        cv::Mat right = imR.clone();
        if (speckle_k >= 3) {
            cv::medianBlur(left, left, speckle_k);
            cv::medianBlur(right, right, speckle_k);
        }
        (void)fx; (void)fy; (void)cx; (void)cy; (void)baseline;

        const double tframe = static_cast<double>(ts_ns) * 1e-9;
        Sophus::SE3f Tcw = SLAM.TrackStereo(left, right, tframe);
        const int state = SLAM.GetTrackingState();
        const bool ok = (state == 2) && !Tcw.matrix().hasNaN();

        float x = 0, y = 0, z = 0, yaw = 0;
        if (ok) {
            Eigen::Matrix3f R = Tcw.rotationMatrix();
            Eigen::Vector3f t = Tcw.translation();
            Eigen::Vector3f twc = -R.transpose() * t;
            x = twc.x();
            y = twc.y();
            z = twc.z();
            // OpenCV cam: X-right Y-down Z-forward. Yaw around Y.
            yaw = std::atan2(R(0, 2), R(2, 2));
        }

        char buf[320];
        std::snprintf(buf, sizeof(buf),
                      "{\"ok\":%d,\"state\":%d,\"t_ns\":%llu,\"x\":%.4f,\"y\":%.4f,"
                      "\"z\":%.4f,\"yaw\":%.4f,\"n\":%llu}\n",
                      ok ? 1 : 0, state,
                      static_cast<unsigned long long>(ts_ns),
                      x, y, z, yaw,
                      static_cast<unsigned long long>(++nframe));
        send_json(ufd, dest, std::string(buf));

        if (nframe % 2 == 0) {
            auto mps = SLAM.GetTrackedMapPoints();
            std::vector<float> xyz;
            xyz.reserve(1500 * 3);
            for (auto* p : mps) {
                if (!p || p->isBad()) continue;
                Eigen::Vector3f w = p->GetWorldPos();
                xyz.push_back(w.x());
                xyz.push_back(w.y());
                xyz.push_back(w.z());
                if (xyz.size() >= 1500 * 3) break;
            }
            const uint16_t npt = static_cast<uint16_t>(xyz.size() / 3);
            const uint16_t pad = 0;
            std::string pkt("ORBP", 4);
            pkt.append(reinterpret_cast<const char*>(&npt), 2);
            pkt.append(reinterpret_cast<const char*>(&pad), 2);
            if (!xyz.empty()) {
                pkt.append(reinterpret_cast<const char*>(xyz.data()),
                           xyz.size() * sizeof(float));
            }
            send_json(ufd, dest, pkt);

            cv::Mat thumb;
            cv::resize(left, thumb, cv::Size(424, 240), 0, 0, cv::INTER_AREA);
            std::vector<uchar> jpg;
            cv::imencode(".jpg", thumb, jpg,
                         {cv::IMWRITE_JPEG_QUALITY, 55});
            const uint32_t nj = static_cast<uint32_t>(jpg.size());
            std::string ipkt("ORBI", 4);
            ipkt.append(reinterpret_cast<const char*>(&nj), 4);
            ipkt.append(reinterpret_cast<const char*>(jpg.data()), jpg.size());
            send_json(ufd, dest, ipkt);
        }
        if (nframe % 30 == 1) {
            std::cout << "orbslam3_track: " << buf;
        }
        if (nframe % 600 == 0) {
            SLAM.SaveMapPoints("/home/james/.kevin/orbslam3/points.txt");
            SLAM.SaveTrajectoryTUM("/home/james/.kevin/orbslam3/traj_tum.txt");
            std::cout << "orbslam3_track: saved points+traj n=" << nframe << std::endl;
        }
    }

    close(cfd);
    close(lfd);
    close(ufd);
    try {
        SLAM.SaveMapPoints("/home/james/.kevin/orbslam3/points.txt");
        SLAM.SaveTrajectoryTUM("/home/james/.kevin/orbslam3/traj_tum.txt");
    } catch (...) {
    }
    SLAM.Shutdown();
    std::cout << "orbslam3_track: shutdown after " << nframe << " frames" << std::endl;
    return 0;
}
