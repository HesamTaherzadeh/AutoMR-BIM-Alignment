#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
#include "ScanContext/nanoflann.hpp"
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

using PointType = pcl::PointXYZ;
using namespace Eigen;

struct ScanContextConfig
{
    float tile_size        = 1.0f;
    int   sectors          = 60;
    int   rings            = 20;
    float max_length       = 80.0f;
    int   kdtree_leaf_size = 10;

    ScanContextConfig() = default;

    explicit ScanContextConfig(const std::string& config_path)
    {
        YAML::Node config = YAML::LoadFile(config_path);
        tile_size         = config["tile_size"].as<float>();
        sectors           = config["sectors"].as<int>();
        rings             = config["rings"].as<int>();
        max_length        = config["max_length"].as<float>();
        kdtree_leaf_size  = config["kdtree_leaf_size"].as<int>();
    }
};

class ScanContextManager
{
   public:
    explicit ScanContextManager(const ScanContextConfig& config);

    void                  tileAndSave(const pcl::PointCloud<PointType>::Ptr& cloud,
                                      const std::string&                     output_folder);
    std::pair<int, float> query(const pcl::PointCloud<PointType>::Ptr& query_cloud);
    void                  saveScanContextsBinary(const std::string& path) const;
    bool                  loadScanContextsBinary(const std::string& path);
    void                  saveScanContextsPerTile(const std::string& folder) const;
    bool                  loadScanContextsFromFolder(const std::string& folder);

   private:
    ScanContextConfig                                              config_;
    std::vector<MatrixXd>                                          scan_contexts_;
    std::vector<pcl::PointCloud<PointType>>                        tiles_;
    std::unique_ptr<nanoflann::KDTreeEigenMatrixAdaptor<MatrixXd>> kdtree_;

    MatrixXd computeScanContext(const pcl::PointCloud<PointType>& cloud);
    void     buildKDTree();
    VectorXd extractRingKey(const MatrixXd& sc);
};
