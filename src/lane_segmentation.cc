#include "ego_lane_detection/lane_segmentation.h"

LaneSegmentation::LaneSegmentation() {}

void LaneSegmentation::Segmentation(
    CommonUtils& common_utils,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& raw_cloud,
    const cv::Mat& index_map, const cv::Mat& vertical_slope_map,
    const cv::Mat& roi_bev_image, const Eigen::MatrixXf& trans_osm_points,
    cv::Mat& lane_bev_image) {
  // Draw OSM points in image
  DrawOSMPoints(trans_osm_points, 0, 2, lane_bev_image);

  // Lane prune using up edge
  int up_edge_row = UpEdgePrune(common_utils, trans_osm_points, raw_cloud,
                                index_map, vertical_slope_map, lane_bev_image);

  // Lane prune using right edge(road curb)
  int lane_width = RightEdgePrune(
      roi_bev_image, up_edge_row >= 0 ? up_edge_row : 0, lane_bev_image);

  // Lane completion for upper rows
  LaneCompletion(trans_osm_points, up_edge_row, lane_width, lane_bev_image);
}

void LaneSegmentation::DrawOSMPoints(const Eigen::MatrixXf& trans_osm_points,
                                     const int& row_offset,
                                     const int& col_offset,
                                     cv::Mat& lane_bev_image) {
#pragma omp parallel for num_threads(8)
  for (int i = 0; i < trans_osm_points.cols() - 1; ++i) {
    cv::Point start_point(trans_osm_points.col(i)(1) + col_offset,
                          trans_osm_points.col(i)(0) + row_offset);
    cv::Point end_point(trans_osm_points.col(i + 1)(1) + col_offset,
                        trans_osm_points.col(i + 1)(0) + row_offset);
    line(lane_bev_image, start_point, end_point, cv::Scalar(255, 255, 255), 1,
         8);
  }
}

int LaneSegmentation::UpEdgePrune(
    CommonUtils& common_utils, const Eigen::MatrixXf& trans_osm_points,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& raw_cloud,
    const cv::Mat& index_map, const cv::Mat& vertical_slope_map,
    cv::Mat& lane_bev_image) {
  // Add padding for bev image
  int up_row_padding = 200;

  cv::Mat mock_lane_image =
      cv::Mat::zeros(BEVImageHeight + up_row_padding, BEVImageWidth, CV_8UC1);

  // Draw left edge
  DrawOSMPoints(trans_osm_points, up_row_padding, 2, mock_lane_image);

// Mark points two memters right of the lane line as the lane
#pragma omp parallel for num_threads(8)
  for (int row = 0; row < mock_lane_image.rows; ++row) {
    for (int col = 0; col < mock_lane_image.cols; ++col) {
      if (mock_lane_image.at<uchar>(row, col) == 255) {
        // Remove left edge
        for (int lane_col = col;
             lane_col < col + 5 && lane_col < mock_lane_image.cols;
             ++lane_col) {
          mock_lane_image.at<uchar>(row, lane_col) = 0;
        }
        // Mock lane
        for (int lane_col = col + 20;
             lane_col < col + 40 && lane_col < mock_lane_image.cols;
             ++lane_col) {
          mock_lane_image.at<uchar>(row, lane_col) = 255;
        }
        break;
      }
    }
  }

  // Compute up edge using point cloud
  int up_edge_row = 0;
#pragma omp parallel for num_threads(8)
  for (int i = 0; i < vertical_slope_map.rows; ++i) {
    for (int j = 0; j < vertical_slope_map.cols; ++j) {
      if (vertical_slope_map.at<uchar>(i, j) > 150) {
        // Candidate obstacle
        const auto& pt = raw_cloud->points[index_map.at<short>(i, j)];
        cv::Point bev_point = common_utils.VeloPointToBEVImage(pt);
        int row = static_cast<int>(bev_point.x) + up_row_padding;
        int col = static_cast<int>(bev_point.y);
        if (row < 0 || row >= mock_lane_image.rows || col < 0 ||
            col >= mock_lane_image.cols)
          continue;
        if (mock_lane_image.at<uchar>(row, col) == 255) {
          if (pt.x > 10.0 && row > up_edge_row) {
            up_edge_row = row;
          }
        }
      }
    }
  }
  up_edge_row -= up_row_padding;

// Lane prune using up edge
#pragma omp parallel for num_threads(8)
  for (int row = 0; row <= up_edge_row; ++row) {
    for (int col = 0; col < lane_bev_image.cols; ++col) {
      lane_bev_image.at<uchar>(row, col) = 0;
    }
  }

  return up_edge_row;
}

int LaneSegmentation::RightEdgePrune(const cv::Mat& roi_bev_image,
                                     const int& up_edge_row,
                                     cv::Mat& lane_bev_image) {
// Mark all points right of the lane line as the lane
#pragma omp parallel for num_threads(8)
  for (int row = 0; row < lane_bev_image.rows; ++row) {
    for (int col = 0; col < lane_bev_image.cols; ++col) {
      if (lane_bev_image.at<uchar>(row, col) == 255) {
        for (; col < lane_bev_image.cols; ++col) {
          lane_bev_image.at<uchar>(row, col) = 255;
        }
        break;
      }
    }
  }

  // Combine roi image to lane image
  bitwise_and(roi_bev_image, lane_bev_image, lane_bev_image);

  // Compute lane width
  long int width_sum = 0;
  int valid_size = 0;
#pragma omp parallel for num_threads(8)
  for (int row = up_edge_row; row < lane_bev_image.rows; ++row) {
    int start_col = -1, end_col = -1;
    for (int col = 0; col < lane_bev_image.cols - 1; ++col) {
      // Get left edge for each row
      if (lane_bev_image.at<uchar>(row, col) == 0 &&
          lane_bev_image.at<uchar>(row, col + 1) == 255) {
        start_col = col;
      }
      // Get right edge for each row
      if (lane_bev_image.at<uchar>(row, col) == 255 &&
          lane_bev_image.at<uchar>(row, col + 1) == 0) {
        end_col = col;
        if (start_col != -1) {
          int width = end_col - start_col;
          // Check if valid
          if (width > 40 && width < 100) {
#pragma omp critical
            width_sum += width;
#pragma omp critical
            valid_size += 1;
          }
          start_col = -1;
          end_col = -1;
        }
      }
    }
  }

  // Compute mean width using all rows
  int width_mean = 0;
  if (valid_size != 0) {
    width_mean = width_sum / valid_size + 2;
  }

// Lane prune using width
#pragma omp parallel for num_threads(8)
  for (int row = up_edge_row; row < lane_bev_image.rows; ++row) {
    for (int col = 0; col < lane_bev_image.cols; ++col) {
      if (lane_bev_image.at<uchar>(row, col) == 255) {
        for (int col_new = col + width_mean; col_new < lane_bev_image.cols;
             ++col_new) {
          lane_bev_image.at<uchar>(row, col_new) = 0;
        }
        break;
      }
    }
  }

  return width_mean;
}

void LaneSegmentation::LaneCompletion(const Eigen::MatrixXf& trans_osm_points,
                                      const int& up_edge_row,
                                      const int& lane_width,
                                      cv::Mat& lane_bev_image) {
  // No obstacle
  if (up_edge_row <= -200) {
    // Remove in valid
    bool invalid = true;
    for (int col = 0; col < lane_bev_image.cols; ++col) {
      if (lane_bev_image.at<uchar>(150, col) == 255) {
        invalid = false;
        break;
      }
    }
    if (invalid) return;

    cv::Mat lane_edge_image = cv::Mat::zeros(lane_bev_image.size(), CV_8UC1);

    // Draw left lane
    DrawOSMPoints(trans_osm_points, 0, 0, lane_edge_image);

    // Draw right lane
    DrawOSMPoints(trans_osm_points, 0, lane_width, lane_edge_image);

    // Only for upper rows
    for (int row = 300; row >= 0; --row) {
      // Get left edge
      int left_edge = 0;
      for (int col = 0; col < lane_edge_image.cols; ++col) {
        if (lane_edge_image.at<uchar>(row, col) == 255) {
          left_edge = col + 5;
          break;
        }
      }
      for (int col = left_edge - 5; col < left_edge + 5; ++col) {
        if (col < 0 || col >= lane_edge_image.cols) continue;
        if (lane_bev_image.at<uchar>(row, col) == 255) {
          left_edge = col;
          break;
        }
      }
      // Get right edge
      int right_edge = 399;
      for (int col = lane_edge_image.cols - 1; col >= 0; --col) {
        if (lane_edge_image.at<uchar>(row, col) == 255) {
          right_edge = col - 5;
          break;
        }
      }
      for (int col = right_edge + 5; col < right_edge - 5; --col) {
        if (col < 0 || col >= lane_edge_image.cols) continue;
        if (lane_bev_image.at<uchar>(row, col) == 255) {
          right_edge = col;
          break;
        }
      }
      if (left_edge > right_edge) {
        if (left_edge < 200)
          left_edge = 0;
        else
          right_edge = 399;
      }
      // Set as lane
      for (int col = left_edge; col < right_edge; ++col)
        lane_bev_image.at<uchar>(row, col) = 255;
    }
  }
}
