#ifndef EGO_LANE_DETECTION_PARAMETER_ESTIMATION_H_
#define EGO_LANE_DETECTION_PARAMETER_ESTIMATION_H_

#include <dirent.h>
#include <fstream>
#include <iostream>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class ParameterEstimation {
 public:
  ParameterEstimation();

  Eigen::MatrixXf Estimation(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &feature_points,
      const Eigen::MatrixXf &osm_points);

 private:
  // Point Cloud Voxel filter
  void CloudVoxelFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud,
                        float voxel_size);

  // Parameter estimation using search based optimization
  Eigen::Vector2f SearchBasedOptimization(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr feature_points,
      const Eigen::MatrixXf &osm_points);

  // Transform OSM points given parameters
  Eigen::MatrixXf TransformOSMPoints(const Eigen::MatrixXf &osm_points,
                                     const float &delta_y,
                                     const float &delta_yaw);

  // Compute distance between feature points and OSM points
  float FeatureOSMDistance(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr feature_points,
      const Eigen::MatrixXf &osm_points, int &count);

  // Compute distance between point and line
  float PointLineDistance(const Eigen::Vector2f &ego_point,
                          const Eigen::Vector2f &start_point,
                          const Eigen::Vector2f &end_point);
};

#endif  // EGO_LANE_DETECTION_PARAMETER_ESTIMATION_H_
