#ifndef EGO_LANE_DETECTION_LANE_SEGMENTATION_H_
#define EGO_LANE_DETECTION_LANE_SEGMENTATION_H_

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>

#include "ego_lane_detection/common_utils.h"

class LaneSegmentation {
 public:
  LaneSegmentation();

  void Segmentation(const cv::Mat& roi_bev_image,
                    const Eigen::MatrixXf& trans_osm_points,
                    cv::Mat& lane_bev_image);

 private:
  // Draw OSM point in image
  void DrawOSMPoints(const Eigen::MatrixXf& trans_osm_points,
                     const int& row_offset, const int& col_offset,
                     cv::Mat& lane_bev_image);

  // Lane prune using up edge
  int UpEdgePrune(const int& lane_width, cv::Mat& lane_bev_image);

  // Lane prune using right edge(road curb)
  int RightEdgePrune(const cv::Mat& roi_bev_image, const int& up_edge_row,
                     cv::Mat& lane_bev_image);

  // Lane completion for upper rows
  void LaneCompletion(const cv::Mat& roi_bev_image,
                      const Eigen::MatrixXf& trans_osm_points,
                      const int& lane_width, cv::Mat& lane_bev_image);
};

#endif  // EGO_LANE_DETECTION_LANE_SEGMENTATION_H_
