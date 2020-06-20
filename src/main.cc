#include <unistd.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#undef NDEBUG
#include <assert.h>

#include "ego_lane_detection/common_utils.h"
#include "ego_lane_detection/feature_extraction.h"
#include "ego_lane_detection/lane_segmentation.h"
#include "ego_lane_detection/parameter_estimation.h"
#include "ego_lane_detection/roi_selection.h"

int frame_size = 95;

// 3D point cloud
std::string cloud_path;
std::vector<std::string> cloud_files;
// Raw per image
std::string per_image_path;
std::vector<std::string> per_image_files;
// Calibration parameters
std::string calib_path;
std::vector<std::string> calib_files;
// OSM points
std::string osm_path;
std::vector<std::string> osm_files;
// Result
std::string result_path;

// Common utils calss
CommonUtils common_utils;

// Runtime for all frames
double runtime = 0.0;

class Timer {
 public:
  // Start Timer
  void start() { start_time_ = std::chrono::steady_clock::now(); }

  // Get elapsed time
  double getElapsedTime() {
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_time_);
    return double(elapsed_time.count()) / CLOCKS_PER_SEC;
  }

 private:
  std::chrono::_V2::steady_clock::time_point start_time_;
};

void Run(const int& frame) {
  printf("Processing frame %d start...\n", frame);

  // Timer
  Timer timer;
  timer.start();

  // Read point cloud data from file and generate index map
  std::string cloud_file = cloud_path + cloud_files[frame];
  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  cv::Mat index_map = common_utils.ReadPointCloud(cloud_file, raw_cloud);

  // Read raw per image
  std::string per_image_file = per_image_path + per_image_files[frame];
  cv::Mat raw_per_image =
      cv::imread(per_image_file.c_str(), cv::IMREAD_GRAYSCALE);

  // Read calib matrixs from file
  std::string calib_file = calib_path + calib_files[frame];
  common_utils.ReadCalibMatrixs(calib_file);

  // Read OSM points from file
  std::string osm_file = osm_path + osm_files[frame];
  Eigen::MatrixXf osm_points = common_utils.ReadOSMPoints(osm_file);

  // Compute BEV look up table
  common_utils.ComputeBEVLookUpTable();

  // Transform per image to bev image
  cv::Mat raw_bev_image;
  common_utils.PerspectiveToBEV(raw_per_image, raw_bev_image);

  // ROI selection
  ROISelection roi_selection(common_utils);
  cv::Mat roi_bev_image, vertical_slope_map;
  roi_selection.Selection(raw_cloud, index_map, raw_per_image.size(),
                          roi_bev_image, vertical_slope_map);

  // Feature extraction
  FeatureExtraction feature_extraction;
  pcl::PointCloud<pcl::PointXYZ>::Ptr feature_points(
      new pcl::PointCloud<pcl::PointXYZ>);
  feature_extraction.Extraction(raw_bev_image, roi_bev_image, feature_points);

  // Parameter estimation
  ParameterEstimation parameter_estimation;
  Eigen::MatrixXf trans_osm_points =
      parameter_estimation.Estimation(feature_points, osm_points);

  // Lane segmentation
  LaneSegmentation lane_segmentation;
  cv::Mat lane_bev_image =
      cv::Mat::zeros(BEVImageHeight, BEVImageWidth, CV_8UC1);
  lane_segmentation.Segmentation(common_utils, raw_cloud, index_map,
                                 vertical_slope_map, roi_bev_image,
                                 trans_osm_points, lane_bev_image);

  // Save result
  std::ostringstream file_number;
  file_number << std::setfill('0') << std::setw(6) << frame;
  std::string result_file =
      result_path + "um_lane_" + file_number.str() + ".png";
  cv::imwrite(result_file, lane_bev_image);

  runtime += timer.getElapsedTime();

  printf("Processing frame %d finish with %f seconds\n", frame,
         timer.getElapsedTime());
}

void BuildUpDataPath(const std::string& dataset) {
  // Get current working directory
  char cur_path[256];
  if (!getcwd(cur_path, 256)) {
    std::cerr << "[Error] can not get current path" << std::endl;
    exit(0);
  }

  // Get project path
  std::string project_path(cur_path);
  int pos = project_path.find_last_of('/');
  project_path = project_path.substr(0, pos);

  // Build up cloud path
  cloud_path = project_path + "/data/" + dataset + "/velodyne/";
  cloud_files = common_utils.GetPathFiles(cloud_path);
  assert(cloud_files.size() == frame_size);

  // Build up per image path
  per_image_path = project_path + "/data/" + dataset + "/per_image/";
  per_image_files = common_utils.GetPathFiles(per_image_path);
  assert(per_image_files.size() == frame_size);

  // Build up calibration path
  calib_path = project_path + "/data/" + dataset + "/calib/";
  calib_files = common_utils.GetPathFiles(calib_path);
  assert(calib_files.size() == frame_size);

  // Build up osm path
  osm_path = project_path + "/data/" + dataset + "/osm/";
  osm_files = common_utils.GetPathFiles(osm_path);
  assert(osm_files.size() == frame_size);

  // Build result path
  result_path = project_path + "/data/" + dataset + "/result/";
  if (access(result_path.c_str(), 0) == -1) mkdir(result_path.c_str(), S_IRWXU);
}

int main(int argc, char** argv) {
  // Check input
  if (argc != 2) {
    std::cerr << "[Error] Need to specify the dataset" << std::endl;
    std::cerr << "-----------------------------------------------" << std::endl;
    std::cerr << "Usage: ego_lane_detection dataset" << std::endl;
    std::cerr << "-----------------------------------------------" << std::endl;
    exit(0);
  }

  // Parse dataset and build up data path
  const std::string dataset = argv[1];
  if (dataset == "training" || dataset == "testing") {
    if (dataset == "testing") frame_size = 96;
    BuildUpDataPath(dataset);
  } else {
    std::cerr << "[Error] Invalid dataset: " << dataset << std::endl;
    exit(0);
  }

  for (int frame = 0; frame < frame_size; ++frame) {
    Run(frame);
  }
  printf("Run time: %f\n", runtime);

  return 0;
}
