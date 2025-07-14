#include "ContextManager/ContextManager.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <fstream>
#include <iostream>
#include <filesystem>

#define LOG

namespace fs = std::filesystem;

ScanContextConfig::ScanContextConfig(const std::string& config_path) {
    YAML::Node config = YAML::LoadFile(config_path);
    tile_size = config["tile_size"].as<float>();
    sectors = config["sectors"].as<int>();
    rings = config["rings"].as<int>();
    max_length = config["max_length"].as<float>();
    kdtree_leaf_size = config["kdtree_leaf_size"].as<int>();
    num_candidates = config["num_candidates"].as<int>();
    dist_threshold = config["dist_threshold"].as<float>();
}

ScanContextManager::ScanContextManager(const ScanContextConfig& config)
    : config_(config) {}

void ScanContextManager::tileAndSave(const pcl::PointCloud<PointType>::Ptr& cloud,
                                     const std::string& output_folder) {
    float min_x = FLT_MAX, min_y = FLT_MAX, max_x = -FLT_MAX, max_y = -FLT_MAX;
    for (const auto& pt : cloud->points) {
        min_x = std::min(min_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_x = std::max(max_x, pt.x);
        max_y = std::max(max_y, pt.y);
    }

    fs::create_directories(output_folder);
    int tile_id = 0;
    for (float x = min_x; x < max_x; x += config_.tile_size) {
        for (float y = min_y; y < max_y; y += config_.tile_size) {
            pcl::PointCloud<PointType> tile;
            for (const auto& pt : cloud->points) {
                if (pt.x >= x && pt.x < x + config_.tile_size &&
                    pt.y >= y && pt.y < y + config_.tile_size) {
                    tile.points.push_back(pt);
                }
            }
            if (tile.size() < 10) continue;  // ✅ Skip very small tiles

            pcl::PointCloud<SCPointType> tile_sc;
            pcl::copyPointCloud(tile, tile_sc);
            MatrixXd sc = sc_manager_.makeScancontext(tile_sc);

            if (sc.norm() < 1e-6) continue;

            std::string tile_path = output_folder + "/tile_" + std::to_string(tile_id) + ".pcd";
            pcl::io::savePCDFileBinary(tile_path, tile);

            scan_contexts_.push_back(sc);
            tiles_.push_back(tile);
            tile_id++;
        }
    }
    buildKDTree();
}


void ScanContextManager::saveScanContextsPerTile(const std::string& folder) const {
    fs::create_directories(folder);
    for (size_t i = 0; i < scan_contexts_.size(); ++i) {
        std::string file = folder + "/scancontext_" + std::to_string(i) + ".bin";
        std::ofstream ofs(file, std::ios::binary);
        int rows = scan_contexts_[i].rows(), cols = scan_contexts_[i].cols();
        ofs.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(scan_contexts_[i].data()), sizeof(double) * rows * cols);
    }
}

bool ScanContextManager::loadScanContextsFromFolder(const std::string& folder) {
    if (!fs::exists(folder)) return false;
    scan_contexts_.clear();
    tiles_.clear();

    int idx = 0;
    while (true) {
        std::string file = folder + "/scancontext_" + std::to_string(idx) + ".bin";
        if (!fs::exists(file)) break;

        std::ifstream ifs(file, std::ios::binary);
        int rows, cols;
        ifs.read(reinterpret_cast<char*>(&rows), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&cols), sizeof(int));
        MatrixXd sc(rows, cols);
        ifs.read(reinterpret_cast<char*>(sc.data()), sizeof(double) * rows * cols);
        scan_contexts_.push_back(sc);
        idx++;
    }

    if (scan_contexts_.empty()) return false;

    buildKDTree();
    return true;
}

void ScanContextManager::buildKDTree() {
    keys_ = MatrixXd(scan_contexts_.size(), scan_contexts_[0].rows());
    for (size_t i = 0; i < scan_contexts_.size(); ++i)
        keys_.row(i) = extractRingKey(scan_contexts_[i]);
    kdtree_ = std::make_unique<nanoflann::KDTreeEigenMatrixAdaptor<MatrixXd>>(keys_.cols(), std::cref(keys_), config_.kdtree_leaf_size);
    kdtree_->index->buildIndex();
}

VectorXd ScanContextManager::extractRingKey(const MatrixXd& sc) {
    return sc.rowwise().mean();
}

std::pair<int, float> ScanContextManager::query(const pcl::PointCloud<PointType>::Ptr& query_cloud) {
    pcl::PointCloud<SCPointType> query_cloud_sc;
    pcl::copyPointCloud(*query_cloud, query_cloud_sc);
    MatrixXd query_sc = sc_manager_.makeScancontext(query_cloud_sc);
    VectorXd query_key = extractRingKey(query_sc);

    const int K = config_.num_candidates;
    std::vector<size_t> nn_idx(K);
    std::vector<double> nn_dist_sqr(K);
    nanoflann::KNNResultSet<double> resultSet(K);
    resultSet.init(nn_idx.data(), nn_dist_sqr.data());
    kdtree_->index->findNeighbors(resultSet, query_key.data(), nanoflann::SearchParams(10));

    double best_dist = 1e9;
    int best_idx = -1;
    for (int i = 0; i < K; ++i) {
        double dist = sc_manager_.distanceBtnScanContext(query_sc, scan_contexts_[nn_idx[i]]).first;
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = nn_idx[i];
        }
    }
    if (best_dist > config_.dist_threshold)
        best_idx = -1;

#ifdef LOG
    std::ofstream log_file("/home/hesam/MR-BIM/actual/Log_distances.txt");
    log_file << "Index RingKeyNormDistance FullSCDistance\n";
    for (size_t i = 0; i < scan_contexts_.size(); ++i) {
        VectorXd sc_key = extractRingKey(scan_contexts_[i]);
        double ringkey_dist = (query_key - sc_key).norm();
        double sc_dist = sc_manager_.distanceBtnScanContext(query_sc, scan_contexts_[i]).first;
        log_file << i << " " << ringkey_dist << " " << sc_dist << "\n";
    }
    log_file.close();
    std::cout << "[INFO] Distance log saved to /home/hesam/MR-BIM/actual/Log_distances.txt\n";
#endif

    return {best_idx, static_cast<float>(best_dist)};
}

