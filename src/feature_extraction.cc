#include "ego_lane_detection/feature_extraction.h"

FeatureExtraction::FeatureExtraction() {}

void FeatureExtraction::Extraction(
    const cv::Mat &raw_image, const cv::Mat &roi_image,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &feature_points) {
  // Generate mask image
  cv::Mat mask_image = roi_image.clone();
  GenerateMask(mask_image);

  // Detect lane line by convolution in mask image
  cv::Mat lane_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
  Convolution(raw_image, mask_image, lane_image);

  // Find contours in lane image
  std::vector<std::vector<cv::Point> > contours;
  findContours(lane_image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

  // Remove outliers
  RemoveOutliers(contours, lane_image);

  // Find all feature points
  GetFeaturePoints(lane_image, contours, feature_points);
}

void FeatureExtraction::GenerateMask(cv::Mat &mask_image) {
#pragma omp parallel for num_threads(8)
  for (int row = 0; row < mask_image.rows; ++row) {
    // Get left edge
    int start_col = 0;
    for (int col = 0; col < mask_image.cols; ++col) {
      if (mask_image.at<uchar>(row, col) == 255) {
        start_col = col;
        break;
      }
    }
    // Get right edge
    int end_col = 0;
    for (int col = mask_image.cols - 1; col >= 0; --col) {
      if (mask_image.at<uchar>(row, col) == 255) {
        end_col = col;
        break;
      }
    }
    // Compute road width
    int road_width = end_col - start_col;
    // Compute erode width
    int erode_width = road_width * 0.3;
    if (erode_width < 30) erode_width = 30;
    // Erode
    for (int col = start_col;
         col < start_col + erode_width && col < mask_image.cols; ++col) {
      mask_image.at<uchar>(row, col) = 0;
    }
    for (int col = end_col; col >= end_col - erode_width && col >= 0; --col) {
      mask_image.at<uchar>(row, col) = 0;
    }
  }
}

void FeatureExtraction::Convolution(const cv::Mat &raw_gray_image,
                                    const cv::Mat &mask_image,
                                    cv::Mat &lane_image) {
#pragma omp parallel for num_threads(8)
  for (int row = 0; row < raw_gray_image.rows; ++row) {
    for (int col = 4; col < raw_gray_image.cols - 5; ++col) {
      // Compute the sum of left 3 pixels' intensity
      int left_intensity = 0;
      for (int k = col - 4; k < col - 1; ++k)
        left_intensity += raw_gray_image.at<uchar>(row, k) * -1;
      // Compute the sum of middle 3 pixels' intensity
      int middle_intensity = 0;
      for (int k = col - 1; k < col + 2; ++k)
        middle_intensity += raw_gray_image.at<uchar>(row, k) * 2;
      // Compute the sum of right 3 pixels' intensity
      int right_intensity = 0;
      for (int k = col + 2; k < col + 5; ++k)
        right_intensity += raw_gray_image.at<uchar>(row, k) * -1;

      // Judge for lane line
      if (fabs(left_intensity - right_intensity) > 100) continue;
      if (middle_intensity < 400) continue;
      int gredient = left_intensity + middle_intensity + right_intensity;
      if (gredient > 200 && mask_image.at<uchar>(row, col) == 255) {
        lane_image.at<uchar>(row, col) = 255;
      }
    }
  }
}

void FeatureExtraction::RemoveOutliers(
    const std::vector<std::vector<cv::Point> > &contours, cv::Mat &lane_image) {
  if (contours.size() > 10) {
#pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < contours.size(); ++i) {
      // Fit bounding box for contour
      cv::Rect bounding_rect = boundingRect(cv::Mat(contours[i]));
      // Remove contours with small size
      if (bounding_rect.area() < 10) {
        for (int row = bounding_rect.x;
             row < bounding_rect.x + bounding_rect.width; ++row) {
          for (int col = bounding_rect.y;
               col < bounding_rect.y + bounding_rect.height; ++col) {
            lane_image.at<uchar>(col, row) = 0;
          }
        }
      }
    }
  }
}

void FeatureExtraction::GetFeaturePoints(
    const cv::Mat &lane_image,
    const std::vector<std::vector<cv::Point> > &contours,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &feature_points) {
#pragma omp parallel for num_threads(8)
  // Loop over all contours
  for (size_t i = 0; i < contours.size(); ++i) {
    // Fit bounding box for contour
    cv::Rect bounding_rect = boundingRect(cv::Mat(contours[i]));
    // Loop over all points in bounding box
    for (int row = bounding_rect.x; row < bounding_rect.x + bounding_rect.width;
         ++row) {
      for (int col = bounding_rect.y;
           col < bounding_rect.y + bounding_rect.height; ++col) {
        // Store feature point to pcl
        if (lane_image.at<uchar>(col, row) == 255) {
          pcl::PointXYZ pt;
          pt.x = col;
          pt.y = row;
          pt.z = 0.0;
#pragma omp critical
          feature_points->points.push_back(pt);
        }
      }
    }
  }
}
