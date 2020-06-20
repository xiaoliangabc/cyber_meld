#ifndef EGO_LANE_DETECTION_ROI_SELECTION_H_
#define EGO_LANE_DETECTION_ROI_SELECTION_H_

#include <dirent.h>
#include <fstream>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common_utils.h"

class ROISelection {
 public:
  ROISelection(CommonUtils &common_utils);

  // Selection
  void Selection(const pcl::PointCloud<pcl::PointXYZ>::Ptr &raw_cloud,
                 const cv::Mat &index_map, const cv::Size &per_image_size,
                 cv::Mat &roi_bev_image, cv::Mat &vertical_slope_map);

 private:
  // Common utils
  CommonUtils common_utils_;

  // Compute horizontal slope feature map
  cv::Mat HorizontalSlopeFeature(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &raw_cloud,
      const cv::Mat &index_map, const int &window_size, const float &max_slope);

  // Compute vertical slope feature map
  cv::Mat VerticalSlopeFeature(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &raw_cloud,
      const cv::Mat &index_map, const float &max_slope);

  // Region grow in both horizontal and vertical direction
  void HorizontalVerticalRegionGrow(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &raw_cloud,
      const cv::Mat &index_map, const cv::Mat &slope_map, const int &max_slope,
      pcl::PointCloud<pcl::PointXYZ>::Ptr &roi_cloud);

  // Logistic function
  float Logistic(float x);
};

#endif  // EGO_LANE_DETECTION_ROI_SELECTION_H_
