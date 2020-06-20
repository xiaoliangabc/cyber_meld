#ifndef EGO_LANE_DETECTION_COMMON_UTILS_H_
#define EGO_LANE_DETECTION_COMMON_UTILS_H_

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>

#include "include_fade2d/Fade_2D.h"
#include "include_fade2d/Triangle2.h"

// Cloud index map size
const int MinRing = 0;
const int MaxRing = 64;
const int VerticalRes = MaxRing - MinRing;
const int HorizontalRes = 500;
// BEV image size
const int BEVImageHeight = 800;
const int BEVImageWidth = 400;

class CommonUtils {
 public:
  CommonUtils();

  // Get all files given path
  std::vector<std::string> GetPathFiles(std::string path);

  // Read calibration matrixs from input file
  void ReadCalibMatrixs(const std::string &calib_file);

  // Read point cloud data form file and generate index map
  cv::Mat ReadPointCloud(const std::string &in_file,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud);

  // Read OSM points form file
  Eigen::MatrixXf ReadOSMPoints(const std::string &file);

  // Compute BEV look up table
  void ComputeBEVLookUpTable();

  // Transform perspective image to brid's eye view image
  void PerspectiveToBEV(const cv::Mat &per_image, cv::Mat &bev_image);

  // Project cloud to perspective image and upsampling
  void ProjectUpsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_cloud,
                       cv::Mat &in_image);

  // Show detection result
  void ShowDetectionResult(const cv::Mat &lane_bev_image,
                           cv::Mat &raw_bev_image);

  cv::Point VeloPointToBEVImage(const pcl::PointXYZ &pt);

 private:
  // Calibration matrixs
  Eigen::MatrixXf trans_rectcam_to_image_;
  Eigen::MatrixXf trans_cam_to_rectcam_;
  Eigen::MatrixXf trans_velo_to_cam_;
  Eigen::MatrixXf trans_imu_to_velo_;
  Eigen::MatrixXf trans_cam_to_road_;
  Eigen::Matrix3f trans_bev_image_to_road_;
  Eigen::Matrix4f trans_road_to_bev_image_;

  // Look up table index for transform perspective to bev
  Eigen::MatrixXf bev_index_;
  Eigen::MatrixXf per_index_;

  // Point Cloud Voxel filter
  void CloudVoxelFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud,
                        float voxel_size);

  // Transform point cloud to image and return
  std::vector<GEOM_FADE2D::Point2> TransformCloudToImage(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud,
      const cv::Size &image_size);

  // Unsampling using Delaunay Triangulation
  void DelaunayUpsampling(const std::vector<GEOM_FADE2D::Point2> &points,
                          cv::Mat &image);

  // Get lines number given a file
  int FileLinesNumber(std::ifstream &file);

  // Judge whether point in image view
  bool IsPointInImageView(const int &row, const int &col, const int &height,
                          const int &width);
};

#endif  // EGO_LANE_DETECTION_COMMON_UTILS_H_
