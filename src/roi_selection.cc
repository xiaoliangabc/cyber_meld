#include "ego_lane_detection/roi_selection.h"

ROISelection::ROISelection(CommonUtils &common_utils)
    : common_utils_(common_utils) {}

void ROISelection::Selection(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &raw_cloud,
    const cv::Mat &index_map, const cv::Size &per_image_size,
    cv::Mat &roi_bev_image, cv::Mat &vertical_slope_map) {
  // Horizontal slope feature maps
  cv::Mat horizontal_slope_map =
      HorizontalSlopeFeature(raw_cloud, index_map, 3, 5.0);

  // Vertical slope feature maps
  vertical_slope_map = VerticalSlopeFeature(raw_cloud, index_map, 0.20);

  // Add two feature map
  cv::Mat weight_map = horizontal_slope_map + vertical_slope_map;

  // Region grow in both horizontal and vertical direction
  pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  HorizontalVerticalRegionGrow(raw_cloud, index_map, weight_map, 20, roi_cloud);

  // Project roi cloud to image and upsampling
  cv::Mat roi_per_image = cv::Mat::zeros(per_image_size, CV_8UC1);
  common_utils_.ProjectUpsample(roi_cloud, roi_per_image);

  // TRansform perspective image to bev image
  roi_bev_image = cv::Mat::zeros(BEVImageHeight, BEVImageWidth, CV_8UC1);
  common_utils_.PerspectiveToBEV(roi_per_image, roi_bev_image);
}

cv::Mat ROISelection::HorizontalSlopeFeature(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &raw_cloud,
    const cv::Mat &index_map, const int &window_size, const float &max_slope) {
  cv::Mat feature_map = cv::Mat::zeros(VerticalRes, HorizontalRes, CV_8UC1);
#pragma omp parallel for num_threads(8)
  // Loop over each row
  for (int i = 0; i < feature_map.rows; ++i) {
    // Loop over each col
    for (int j = 0; j < feature_map.cols; ++j) {
      std::vector<cv::Point2f> window_points;
      // Push back all window points to a vector
      for (int k = j - window_size; k <= j + window_size; ++k) {
        const auto &point_index = index_map.at<short>(i, k);
        if (k < 0 || k >= feature_map.cols || point_index < 0) continue;
        window_points.push_back(cv::Point2f(raw_cloud->points[point_index].y,
                                            raw_cloud->points[point_index].x));
      }
      // Check whether valid
      if (window_points.size() > 3) {
        // Fit all points to a line
        cv::Vec4f line;
        cv::fitLine(cv::Mat(window_points), line, cv::DIST_L2, 0, 1e-2, 1e-2);
        // Compute slope
        float slope = fabs(line[1] / line[0]);
        // Normalize slope
        float slope_norm = Logistic(slope / max_slope) * 255;
        feature_map.at<uchar>(i, j) = static_cast<int>(slope_norm);
      }
    }
  }

  return feature_map;
}

cv::Mat ROISelection::VerticalSlopeFeature(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &raw_cloud,
    const cv::Mat &index_map, const float &max_slope) {
  cv::Mat feature_map = cv::Mat::zeros(VerticalRes, HorizontalRes, CV_8UC1);
#pragma omp parallel for num_threads(8)
  // Loop over each cols
  for (int i = 0; i < feature_map.cols; ++i) {
    // Push back all valid index to a vector
    std::vector<int> valid_index;
    for (int j = 0; j < feature_map.rows; ++j) {
      if (index_map.at<short>(j, i) > -1) valid_index.push_back(j);
    }
    // Loop over all valid points
    for (int j = 0; j < valid_index.size(); ++j) {
      // Get firt ring point
      const auto &pt1 =
          raw_cloud->points[index_map.at<short>(valid_index[j], i)];
      float r1 = sqrt(pt1.x * pt1.x + pt1.y * pt1.y);
      float z1 = pt1.z;
      // Get second ring number
      int k = j + 1;
      if (valid_index[j] < 5)
        k += 11;
      else if (valid_index[j] < 10)
        k += 9;
      else if (valid_index[j] < 15)
        k += 7;
      else if (valid_index[j] < 20)
        k += 5;
      else if (valid_index[j] < 25)
        k += 3;
      else if (valid_index[j] < 30)
        k += 1;
      if (k < valid_index.size()) {
        // Get second ring point
        const auto &pt2 =
            raw_cloud->points[index_map.at<short>(valid_index[k], i)];
        float r2 = sqrt(pt2.x * pt2.x + pt2.y * pt2.y);
        float z2 = pt2.z;
        // Compute slope
        float r_diff = r1 - r2;
        float z_diff = z1 - z2;
        float slope = fabs(z_diff) / fabs(r_diff);
        // Normalize slope
        float slope_norm = Logistic(slope / max_slope) * 255;
        feature_map.at<uchar>(valid_index[k], i) = static_cast<int>(slope_norm);
      }
    }
  }

  return feature_map;
}

void ROISelection::HorizontalVerticalRegionGrow(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &raw_cloud,
    const cv::Mat &index_map, const cv::Mat &slope_map, const int &max_slope,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &roi_cloud) {
  // Find middle of road for grow starting
  int middle_edge = 0;
  int max_middle_size = 0;
  // Loop through form left to middle
  for (int j = 100; j < slope_map.cols / 2; ++j) {
    int middle_size = 0;
    for (int i = 0; i < slope_map.rows; ++i) {
      if (static_cast<int>(slope_map.at<uchar>(i, j)) < 100)
        middle_size++;
      else
        break;
    }
    if (middle_size >= max_middle_size) {
      max_middle_size = middle_size;
      middle_edge = j;
    }
  }
  // Loop through form right to middle
  for (int j = slope_map.cols - 100; j >= slope_map.cols / 2; --j) {
    int middle_size = 0;
    for (int i = 0; i < slope_map.rows; ++i) {
      if (static_cast<int>(slope_map.at<uchar>(i, j)) < 100)
        middle_size++;
      else
        break;
    }
    if (middle_size >= max_middle_size) {
      max_middle_size = middle_size;
      middle_edge = j;
    }
  }

  std::vector<int> left_edges;
  std::vector<int> right_edges;
  // Region grow in horizontal direction
  for (int i = 0; i < slope_map.rows; ++i) {
    // Slid right
    for (int j = middle_edge; j < slope_map.cols; ++j) {
      if (static_cast<int>(slope_map.at<uchar>(i, j)) < max_slope) {
        if (index_map.at<short>(i, j) > -1) {
          roi_cloud->points.push_back(
              raw_cloud->points[index_map.at<short>(i, j)]);
        }
      } else {
        left_edges.push_back(j);
        break;
      }
    }
    // Slid left
    for (int j = middle_edge; j >= 0; --j) {
      if (static_cast<int>(slope_map.at<uchar>(i, j)) < max_slope) {
        if (index_map.at<short>(i, j) > -1) {
          roi_cloud->points.push_back(
              raw_cloud->points[index_map.at<short>(i, j)]);
        }
      } else {
        right_edges.push_back(j);
        break;
      }
    }
  }

  // Region grow in vertical direction
  for (int j = right_edges[0]; j < left_edges[0]; ++j) {
    for (int i = 0; i < slope_map.rows; ++i) {
      if (static_cast<int>(slope_map.at<uchar>(i, j)) < max_slope) {
        if (index_map.at<short>(i, j) > -1) {
          roi_cloud->points.push_back(
              raw_cloud->points[index_map.at<short>(i, j)]);
        }
      } else {
        break;
      }
    }
  }
}

inline float ROISelection::Logistic(float x) {
  // definition domain [0, 1], value range [0, 1]
  return 1.0 / (1.0 + exp(5.0 - 10.0 * x));
}
