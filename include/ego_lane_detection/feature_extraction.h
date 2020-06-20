#ifndef EGO_LANE_DETECTION_FEATURE_EXTRACTION_H_
#define EGO_LANE_DETECTION_FEATURE_EXTRACTION_H_

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

class FeatureExtraction {
 public:
  FeatureExtraction();

  // Extraction lane line feature
  void Extraction(const cv::Mat &raw_image, const cv::Mat &roi_image,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr &feature_points);

 private:
  // Generate mask image
  void GenerateMask(cv::Mat &mask_image);

  // Convolution in mask image
  void Convolution(const cv::Mat &raw_gray_image, const cv::Mat &mask_image,
                   cv::Mat &lane_image);

  // Remove outliers
  void RemoveOutliers(const std::vector<std::vector<cv::Point> > &contours,
                      cv::Mat &lane_image);

  // Find all feature points
  void GetFeaturePoints(const cv::Mat &lane_image,
                        const std::vector<std::vector<cv::Point> > &contours,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr &feature_points);
};

#endif  // EGO_LANE_DETECTION_FEATURE_EXTRACTION_H_
