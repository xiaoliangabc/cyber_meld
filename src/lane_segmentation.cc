#include "ego_lane_detection/lane_segmentation.h"

LaneSegmentation::LaneSegmentation() {}

void LaneSegmentation::Segmentation(const cv::Mat& roi_bev_image,
                                    const Eigen::MatrixXf& trans_osm_points,
                                    cv::Mat& lane_bev_image) {
  // Draw OSM points in image
  DrawOSMPoints(trans_osm_points, 0, 2, lane_bev_image);

  // Lane prune using right edge(road curb)
  int lane_width = RightEdgePrune(roi_bev_image, 0, lane_bev_image);

  // Lane prune using up edge
  int up_edge_row = UpEdgePrune(lane_width, lane_bev_image);

  // Lane completion for upper rows
  LaneCompletion(roi_bev_image, trans_osm_points, lane_width, lane_bev_image);
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

int LaneSegmentation::UpEdgePrune(const int& lane_width,
                                  cv::Mat& lane_bev_image) {
  // Compute up edge
  int up_edge_row = 0;
  for (int i = 0; i < lane_bev_image.rows - 100; ++i) {
    int edge_size = 0;
    int edge_col = -100;
    // Find edge
    for (int j = 0; j < lane_bev_image.cols - 1; ++j) {
      if (lane_bev_image.at<uchar>(i, j) == 0 &&
          lane_bev_image.at<uchar>(i, j + 1) == 255 && j - edge_col > 5) {
        edge_size++;
        edge_col = j;
      }
      if (lane_bev_image.at<uchar>(i, j) == 255 &&
          lane_bev_image.at<uchar>(i, j + 1) == 0 && j - edge_col > 5) {
        edge_size++;
        edge_col = j;
      }
    }
    // Take the largest row
    if (edge_size >= 4 && i > up_edge_row) {
      up_edge_row = i;
    }
  }

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

void LaneSegmentation::LaneCompletion(const cv::Mat& roi_bev_image,
                                      const Eigen::MatrixXf& trans_osm_points,
                                      const int& lane_width,
                                      cv::Mat& lane_bev_image) {
  cv::Mat lane_edge_image = cv::Mat::zeros(lane_bev_image.size(), CV_8UC1);

  // Draw left lane
  DrawOSMPoints(trans_osm_points, 0, 0, lane_edge_image);

  // Draw right lane
  DrawOSMPoints(trans_osm_points, 0, lane_width, lane_edge_image);

  for (int row = 0; row < lane_edge_image.rows; ++row) {
    // Get left edge
    int left_edge = 0;
    for (int col = 0; col < lane_edge_image.cols; ++col) {
      if (lane_edge_image.at<uchar>(row, col) == 255) {
        left_edge = col;
        break;
      }
    }
    // Get right edge
    int right_edge = 399;
    for (int col = lane_edge_image.cols - 1; col >= 0; --col) {
      if (lane_edge_image.at<uchar>(row, col) == 255) {
        right_edge = col;
        break;
      }
    }
    if (abs(left_edge - right_edge) < 3 && left_edge < 200) {
      for (int col = 0; col < right_edge; ++col) {
        if (roi_bev_image.at<uchar>(row, col) == 255) {
          lane_bev_image.at<uchar>(row, col) = 255;
        }
      }
    }
  }
}
