#ifndef EGO_LANE_DETECTION_COMMON_UTILS_H_
#define EGO_LANE_DETECTION_COMMON_UTILS_H_

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>

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

  // Read OSM points form file
  Eigen::MatrixXf ReadOSMPoints(const std::string &file);

  // Compute BEV look up table
  void ComputeBEVLookUpTable();

  // Transform perspective image to brid's eye view image
  void PerspectiveToBEV(const cv::Mat &per_image, cv::Mat &bev_image);

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

  // Get lines number given a file
  int FileLinesNumber(std::ifstream &file);

  // Judge whether point in image view
  bool IsPointInImageView(const int &row, const int &col, const int &height,
                          const int &width);
};

#endif  // EGO_LANE_DETECTION_COMMON_UTILS_H_
