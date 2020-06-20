#include "ego_lane_detection/parameter_estimation.h"

ParameterEstimation::ParameterEstimation() {}

Eigen::MatrixXf ParameterEstimation::Estimation(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &feature_points,
    const Eigen::MatrixXf &osm_points) {
  // Down sample feature points
  pcl::PointCloud<pcl::PointXYZ>::Ptr feature_points_filter(
      new pcl::PointCloud<pcl::PointXYZ>);
  CloudVoxelFilter(feature_points, feature_points_filter, 40);

  // Parameter estimation using search based optimization
  Eigen::Vector2f optimal_parameter = Eigen::Vector2f::Zero();
  if (feature_points->points.size() > 50 ||
      feature_points_filter->points.size() > 3) {
    optimal_parameter =
        SearchBasedOptimization(feature_points_filter, osm_points);
  }

  // Transform OSM points given optimal parameters
  Eigen::MatrixXf trans_osm_points = TransformOSMPoints(
      osm_points, optimal_parameter(0), optimal_parameter(1));

  return trans_osm_points;
}

void ParameterEstimation::CloudVoxelFilter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud, float voxel_size) {
  pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
  voxel_grid.setInputCloud(in_cloud);
  voxel_grid.setLeafSize((float)voxel_size, (float)voxel_size,
                         (float)voxel_size);
  voxel_grid.filter(*out_cloud);
}

Eigen::Vector2f ParameterEstimation::SearchBasedOptimization(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr feature_points,
    const Eigen::MatrixXf &osm_points) {
  float min_distance = std::numeric_limits<float>::max();
  int max_count = 0;
  Eigen::Vector2f optimal_parameter = Eigen::Vector2f::Zero();
#pragma omp parallel for num_threads(8)
  for (int delta_y = -100; delta_y < 100; delta_y += 5) {
    for (float delta_yaw = -0.1; delta_yaw < 0.1; delta_yaw += 0.005) {
      // Transform osm points
      Eigen::MatrixXf trans_osm_points =
          TransformOSMPoints(osm_points, delta_y, delta_yaw);
      // Compute distance
      int count = 0;
      float distance =
          FeatureOSMDistance(feature_points, trans_osm_points, count);
      // Find minimum distance
      if (count > max_count) {
        max_count = count;
        optimal_parameter = Eigen::Vector2f(delta_y, delta_yaw);
        min_distance = distance;
      } else if (count == max_count && min_distance > distance) {
        min_distance = distance;
        optimal_parameter = Eigen::Vector2f(delta_y, delta_yaw);
      }
    }
  }

  return optimal_parameter;
}

Eigen::MatrixXf ParameterEstimation::TransformOSMPoints(
    const Eigen::MatrixXf &osm_points, const float &delta_y,
    const float &delta_yaw) {
  // Combine parameters to a matrix
  Eigen::MatrixXf tranform_matrix(2, 2);
  tranform_matrix << cos(delta_yaw), sin(delta_yaw), -sin(delta_yaw),
      cos(delta_yaw);

  // Rotation
  Eigen::MatrixXf transform_points = tranform_matrix * osm_points;

  // Translation
  transform_points.row(1).array() += delta_y;

  return transform_points;
}

float ParameterEstimation::FeatureOSMDistance(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr feature_points,
    const Eigen::MatrixXf &osm_points, int &count) {
  double sum_distance = 0.0;
  count = 0;
  for (const auto &pt : feature_points->points) {
    float distance = 0;
    for (int i = 0; i < osm_points.cols(); ++i) {
      if (pt.x > osm_points.col(i)(0)) {
        distance = PointLineDistance(Eigen::Vector2f(pt.x, pt.y),
                                     osm_points.col(i - 1), osm_points.col(i));
        break;
      }
    }
    if (distance < 1.0) count++;
    sum_distance += distance;
  }

  return sum_distance / feature_points->points.size();
}

float ParameterEstimation::PointLineDistance(const Eigen::Vector2f &ego_point,
                                             const Eigen::Vector2f &start_point,
                                             const Eigen::Vector2f &end_point) {
  float num =
      fabs((end_point(1) - start_point(1)) * ego_point(0) -
           (end_point(0) - start_point(0)) * ego_point(1) +
           end_point(0) * start_point(1) - end_point(1) * start_point(0));
  float den =
      sqrt((end_point(1) - start_point(1)) * (end_point(1) - start_point(1)) +
           (end_point(0) - start_point(0)) * (end_point(0) - start_point(0)));
  return num / den;
}
