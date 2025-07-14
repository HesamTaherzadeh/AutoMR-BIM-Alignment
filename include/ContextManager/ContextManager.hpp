#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <string>
#include <filesystem>
#include <ScanContext/Scancontext.h>
#include "ScanContext/nanoflann.hpp"

using PointType = pcl::PointXYZ;
using namespace Eigen;

struct ScanContextConfig {
    float tile_size = 1.0f;
    int sectors = 60;
    int rings = 20;
    float max_length = 80.0f;
    int kdtree_leaf_size = 10;
    int num_candidates = 10;
    float dist_threshold = 0.35f;

    explicit ScanContextConfig(const std::string& config_path);
};

class ScanContextManager {
public:
    explicit ScanContextManager(const ScanContextConfig& config);

    void tileAndSave(const pcl::PointCloud<PointType>::Ptr& cloud, const std::string& output_folder);
    void saveScanContextsPerTile(const std::string& folder) const;
    bool loadScanContextsFromFolder(const std::string& folder);

    std::pair<int, float> query(const pcl::PointCloud<PointType>::Ptr& query_cloud);

private:
    ScanContextConfig config_;
    SCManager sc_manager_;

    std::vector<MatrixXd> scan_contexts_;
    std::vector<pcl::PointCloud<PointType>> tiles_;
    MatrixXd keys_;

    std::unique_ptr<nanoflann::KDTreeEigenMatrixAdaptor<MatrixXd>> kdtree_;

    void buildKDTree();
    VectorXd extractRingKey(const MatrixXd& sc);
};
