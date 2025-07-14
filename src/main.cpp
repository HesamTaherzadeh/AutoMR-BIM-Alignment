#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>
#include "ContextManager/ContextManager.hpp"
#include <iostream>

int main()
{
    ScanContextConfig config("Config/cfg.yaml");

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    if (pcl::io::loadPCDFile("/home/hesam/MR-BIM/actual/kitti2011_09_30_0018.pcd", *cloud) == -1)
    {
        std::cerr << "Couldn't read file input.pcd\n";
        return -1;
    }

    ScanContextManager manager(config);
    manager.tileAndSave(cloud, "tiles");
    manager.saveScanContextsPerTile("tiles/scancontexts");

    auto result = manager.query(cloud);
    std::cout << "Best match index: " << result.first << ", distance: " << result.second
              << std::endl;

    return 0;
}