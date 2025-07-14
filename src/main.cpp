#include "ContextManager/ContextManager.hpp"
#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    ScanContextConfig config("/home/hesam/MR-BIM/actual/Config/cfg.yaml");
    ScanContextManager manager(config);

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    if (pcl::io::loadPCDFile("/home/hesam/MR-BIM/actual/kitti2011_09_30_0018.pcd", *cloud) == -1) {
        std::cerr << "Couldn't read file input.pcd\n";
        return -1;
    }

    if (manager.loadScanContextsFromFolder("tiles/scancontexts")) {
        std::cout << "Loaded scan contexts from folder.\n";
    } else {
        manager.tileAndSave(cloud, "tiles");
        manager.saveScanContextsPerTile("tiles/scancontexts");
        std::cout << "Saved scan contexts per tile.\n";
    }

    pcl::PointCloud<PointType>::Ptr query_cloud(new pcl::PointCloud<PointType>());
    if (pcl::io::loadPCDFile("/home/hesam/MR-BIM/actual/tiles/tile_24.pcd", *query_cloud) == -1) {
        std::cerr << "Couldn't read file query.pcd\n";
        return -1;
    }

    std::cout << "Queryting output" << std::endl;
    
    auto result = manager.query(query_cloud);
    std::cout << "Best match index: " << result.first
              << ", distance: " << result.second << std::endl;

    return 0;
}
