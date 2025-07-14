#include "ContextManager/ContextManager.hpp"
#include <pcl/io/pcd_io.h>
#include <cfloat>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

ScanContextManager::ScanContextManager(const ScanContextConfig& config) : config_(config) {}

void ScanContextManager::tileAndSave(const pcl::PointCloud<PointType>::Ptr& cloud,
                                     const std::string&                     output_folder)
{
    float min_x = FLT_MAX, min_y = FLT_MAX, max_x = -FLT_MAX, max_y = -FLT_MAX;
    for (const auto& pt : cloud->points)
    {
        min_x = std::min(min_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_x = std::max(max_x, pt.x);
        max_y = std::max(max_y, pt.y);
    }

    fs::create_directories(output_folder);
    int tile_id = 0;

    for (float x = min_x; x < max_x; x += config_.tile_size)
    {
        for (float y = min_y; y < max_y; y += config_.tile_size)
        {
            pcl::PointCloud<PointType> tile;
            for (const auto& pt : cloud->points)
            {
                if (pt.x >= x && pt.x < x + config_.tile_size && pt.y >= y &&
                    pt.y < y + config_.tile_size)
                    tile.points.push_back(pt);
            }
            if (tile.empty()) continue;
            std::string tile_path = output_folder + "/tile_" + std::to_string(tile_id) + ".pcd";
            pcl::io::savePCDFileBinary(tile_path, tile);
            MatrixXd sc = computeScanContext(tile);
            scan_contexts_.push_back(sc);
            tiles_.push_back(tile);
            tile_id++;
        }
    }
    buildKDTree();
}

void ScanContextManager::saveScanContextsBinary(const std::string& path) const
{
    std::ofstream ofs(path, std::ios::binary);
    int           total = scan_contexts_.size();
    ofs.write(reinterpret_cast<const char*>(&total), sizeof(int));

    for (const auto& sc : scan_contexts_)
    {
        int rows = sc.rows(), cols = sc.cols();
        ofs.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(sc.data()), sizeof(double) * rows * cols);
    }
}

void ScanContextManager::saveScanContextsPerTile(const std::string& folder) const
{
    fs::create_directories(folder);
    for (size_t i = 0; i < scan_contexts_.size(); ++i)
    {
        std::string   file = folder + "/scancontext_" + std::to_string(i) + ".bin";
        std::ofstream ofs(file, std::ios::binary);
        const auto&   sc   = scan_contexts_[i];
        int           rows = sc.rows(), cols = sc.cols();
        ofs.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(sc.data()), sizeof(double) * rows * cols);
    }
}

bool ScanContextManager::loadScanContextsFromFolder(const std::string& folder)
{
    if (!fs::exists(folder)) return false;
    scan_contexts_.clear();
    tiles_.clear();

    int idx = 0;
    while (true)
    {
        std::string file = folder + "/scancontext_" + std::to_string(idx) + ".bin";
        if (!fs::exists(file)) break;
        std::ifstream ifs(file, std::ios::binary);
        if (!ifs.is_open()) break;

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

bool ScanContextManager::loadScanContextsBinary(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;

    int total;
    ifs.read(reinterpret_cast<char*>(&total), sizeof(int));
    scan_contexts_.clear();
    tiles_.clear();  // optional: if you're not saving tiles

    for (int i = 0; i < total; ++i)
    {
        int rows, cols;
        ifs.read(reinterpret_cast<char*>(&rows), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&cols), sizeof(int));

        MatrixXd sc(rows, cols);
        ifs.read(reinterpret_cast<char*>(sc.data()), sizeof(double) * rows * cols);
        scan_contexts_.push_back(sc);
    }

    buildKDTree();
    return true;
}

MatrixXd ScanContextManager::computeScanContext(const pcl::PointCloud<PointType>& cloud)
{
    int      sectors = config_.sectors, rings = config_.rings;
    float    max_length = config_.max_length;
    MatrixXd desc       = MatrixXd::Zero(rings, sectors);

    for (const auto& pt : cloud.points)
    {
        float xy_range = sqrt(pt.x * pt.x + pt.y * pt.y);
        if (xy_range > max_length) continue;
        float angle = atan2(pt.y, pt.x) * 180.0 / M_PI;
        if (angle < 0) angle += 360.0;

        int ring_idx   = std::min(int((xy_range / max_length) * rings), rings - 1);
        int sector_idx = std::min(int((angle / 360.0) * sectors), sectors - 1);

        desc(ring_idx, sector_idx) = std::max<double>(desc(ring_idx, sector_idx),
                                                      static_cast<double>(pt.z));
    }
    return desc;
}

VectorXd ScanContextManager::extractRingKey(const MatrixXd& sc)
{
    return sc.rowwise().mean();
}

void ScanContextManager::buildKDTree()
{
    MatrixXd keys(scan_contexts_.size(), scan_contexts_[0].rows());
    for (size_t i = 0; i < scan_contexts_.size(); ++i)
        keys.row(i) = extractRingKey(scan_contexts_[i]);
    kdtree_ = std::make_unique<nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd>>(
        keys.cols(), std::cref(keys), config_.kdtree_leaf_size);
    kdtree_->index->buildIndex();
}

std::pair<int, float> ScanContextManager::query(const pcl::PointCloud<PointType>::Ptr& query_cloud)
{
    auto                            query_sc  = computeScanContext(*query_cloud);
    VectorXd                        query_key = extractRingKey(query_sc);
    size_t                          ret_index;
    double                          out_dist_sqr;
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_index, &out_dist_sqr);
    kdtree_->index->findNeighbors(resultSet, query_key.data(), nanoflann::SearchParams(10));
    return {int(ret_index), std::sqrt(out_dist_sqr)};
}
