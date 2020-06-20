#include "ego_lane_detection/common_utils.h"

CommonUtils::CommonUtils() {
  // Initial BEV index
  bev_index_ = Eigen::MatrixXf::Ones(3, BEVImageHeight * BEVImageWidth);
  for (int row = 0; row < BEVImageHeight; ++row) {
    for (int col = 0; col < BEVImageWidth; ++col) {
      bev_index_(0, row * BEVImageWidth + col) = row;
      bev_index_(1, row * BEVImageWidth + col) = col;
    }
  }

  // Initial per index
  per_index_ = Eigen::MatrixXf::Ones(2, BEVImageHeight * BEVImageWidth);

  // Compute transform matrix from BEV image to road (3*3)
  trans_bev_image_to_road_ << 0.0, 0.05, -9.975, -0.05, 0.0, 45.975, 0.0, 0.0,
      1.0;

  // Compute transform matrix from road to BEV image (4*4)
  trans_road_to_bev_image_ << 0.0, 0.0, -20.0, 45.975 * 20.0, 20.0, 0.0, 0.0,
      9.975 * 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}

std::vector<std::string> CommonUtils::GetPathFiles(std::string path) {
  DIR *dir;
  struct dirent *ent;
  std::vector<std::string> files;
  if (dir = opendir(path.c_str())) {
    // Print all the files and directories within directory
    while ((ent = readdir(dir)) != NULL) {
      if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
        continue;
      files.push_back(ent->d_name);
    }
    closedir(dir);
  } else {
    std::cerr << "Could not open directory: " << path << std::endl;
    exit(0);
  }
  std::sort(files.begin(), files.end());

  return files;
}

void CommonUtils::ReadCalibMatrixs(const std::string &calib_file) {
  // Open calib file
  std::ifstream calib_ifs(calib_file);

  // Loop over all lines, each line represent one matrix
  std::string temp_str;
  while (std::getline(calib_ifs, temp_str)) {
    std::istringstream iss(temp_str);
    // Get matrix name
    std::string matrix_name;
    iss >> matrix_name;
    if (matrix_name == "P2:") {
      trans_rectcam_to_image_ = Eigen::MatrixXf::Zero(3, 4);
      float temp_float;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
          iss >> temp_float;
          trans_rectcam_to_image_(i, j) = temp_float;
        }
      }
    } else if (matrix_name == "R0_rect:") {
      trans_cam_to_rectcam_ = Eigen::MatrixXf::Zero(4, 4);
      float temp_float;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          iss >> temp_float;
          trans_cam_to_rectcam_(i, j) = temp_float;
        }
      }
      trans_cam_to_rectcam_(3, 3) = 1.0;
    } else if (matrix_name == "Tr_velo_to_cam:") {
      trans_velo_to_cam_ = Eigen::MatrixXf::Zero(4, 4);
      float temp_float;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
          iss >> temp_float;
          trans_velo_to_cam_(i, j) = temp_float;
        }
      }
      trans_velo_to_cam_(3, 3) = 1.0;
    } else if (matrix_name == "Tr_imu_to_velo:") {
      trans_imu_to_velo_ = Eigen::MatrixXf::Zero(4, 4);
      float temp_float;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
          iss >> temp_float;
          trans_imu_to_velo_(i, j) = temp_float;
        }
      }
      trans_imu_to_velo_(3, 3) = 1.0;
    } else if (matrix_name == "Tr_cam_to_road:") {
      trans_cam_to_road_ = Eigen::MatrixXf::Zero(4, 4);
      float temp_float;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
          iss >> temp_float;
          trans_cam_to_road_(i, j) = temp_float;
        }
      }
      trans_cam_to_road_(3, 3) = 1.0;
    }
  }
}

cv::Mat CommonUtils::ReadPointCloud(
    const std::string &in_file, pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud) {
  // Load point cloud from .bin file
  std::fstream input(in_file.c_str(), std::ios::in | std::ios::binary);
  if (!input.good()) {
    std::cerr << "[Error] Could not read file: " << in_file << std::endl;
    exit(0);
  }
  input.seekg(0, std::ios::beg);

  int cloud_size = 0;
  int cur_ring = 63;
  int pre_col = -1;
  float pre_theta = 0.0;
  cv::Mat index_map = cv::Mat_<short>(VerticalRes, HorizontalRes, -1);
  // Transform .bin file to pcl cloud
  for (size_t i = 0; input.good() && !input.eof(); i++) {
    pcl::PointXYZ pt;
    // Read data
    input.read((char *)&pt.x, 3 * sizeof(float));
    float intensity;
    input.read((char *)&intensity, sizeof(float));
    // Calculate the ring that the point belong to
    float theta = atan2(pt.y, pt.x) * 180.0 / M_PI;
    if (theta > -1e-9 && theta < 1e-9) continue;
    if (theta < -1e-2) theta += 360;
    if (fabs(theta - pre_theta) > 200) cur_ring--;
    pre_theta = theta;
    // Only remain point in front of vehicle
    if (theta < 315 && theta > 45.0) continue;
    // Remove points too far away to vehicle
    if (cur_ring > MaxRing) continue;
    // Remove points too close to vehicle
    if (cur_ring < MinRing) continue;
    // Remove points very close to vehicle
    if (pt.x * pt.x + pt.y * pt.y < 9.0) continue;
    // Remove points too low or too high
    if (pt.z < -3.0 || pt.z > -0.5) continue;
    // Set index map
    if (theta > 180) theta -= 360;
    int col = static_cast<int>((theta + 45.0) / 90.0 * HorizontalRes);
    if (col == pre_col) continue;
    int row = cur_ring - MinRing;
    index_map.at<short>(row, col) = cloud_size;
    cloud_size++;
    pre_col = col;
    raw_cloud->points.push_back(pt);
  }
  input.close();
  return index_map;
}

Eigen::MatrixXf CommonUtils::ReadOSMPoints(const std::string &file) {
  std::ifstream ifs;
  ifs.open(file.c_str(), std::ios::in);

  // Initial Matrix
  int points_num = FileLinesNumber(ifs);
  Eigen::MatrixXf imu_points = Eigen::MatrixXf::Zero(4, points_num);

  // Read OSM points line by line (in imu coordinate)
  int count = 0;
  std::string line_str;
  while (std::getline(ifs, line_str)) {
    // Assign std::string to std::istringstream
    std::istringstream line_ss(line_str);
    // Parse all data in line
    Eigen::Vector4f imu_point;
    line_ss >> imu_point(0);
    line_ss >> imu_point(1);
    imu_point(2) = 0.0;
    imu_point(3) = 1.0;
    imu_points.col(count) = imu_point;
    count++;
  }

  // Transform point from imu to bev image
  Eigen::MatrixXf trans_imu_to_bev_image =
      trans_road_to_bev_image_ * trans_cam_to_road_ * trans_velo_to_cam_ *
      trans_imu_to_velo_;
  Eigen::MatrixXf bev_points = trans_imu_to_bev_image * imu_points;

  return bev_points.topRows(2);
}

void CommonUtils::ComputeBEVLookUpTable() {
  // Compute transform matrix from road to per image
  Eigen::MatrixXf trans_road_to_per_image = trans_rectcam_to_image_ *
                                            trans_cam_to_rectcam_ *
                                            trans_cam_to_road_.inverse();
  Eigen::Matrix3f trans_road_to_per_image33;
  trans_road_to_per_image33.col(0) = trans_road_to_per_image.col(0);
  trans_road_to_per_image33.col(1) = trans_road_to_per_image.col(2);
  trans_road_to_per_image33.col(2) = trans_road_to_per_image.col(3);

  // Transform BEV index to per index
  Eigen::MatrixXf trans_bev_image_to_per_image =
      trans_road_to_per_image33 * trans_bev_image_to_road_;
  Eigen::MatrixXf per_index3 = trans_bev_image_to_per_image * bev_index_;
  per_index_ = per_index3.colwise().hnormalized();
}

void CommonUtils::PerspectiveToBEV(const cv::Mat &per_image,
                                   cv::Mat &bev_image) {
  // Get per image size
  int per_image_height = per_image.rows;
  int per_image_width = per_image.cols;

  // Create bev image
  bev_image = cv::Mat::zeros(BEVImageHeight, BEVImageWidth, CV_8UC1);

#pragma omp parallel for num_threads(8)
  // Loop over all indices
  for (int i = 0; i < BEVImageHeight * BEVImageWidth; ++i) {
    // Get per index
    int per_col = static_cast<int>(per_index_(0, i)) - 1;
    int per_row = static_cast<int>(per_index_(1, i)) - 1;
    // Check per index valid
    if (!IsPointInImageView(per_row, per_col, per_image_height,
                            per_image_width))
      continue;
    // Get bev index
    int bev_col = static_cast<int>(bev_index_(1, i));
    int bev_row = static_cast<int>(bev_index_(0, i));
    // Assign
    bev_image.at<uchar>(bev_row, bev_col) =
        per_image.at<uchar>(per_row, per_col);
  }
}

void CommonUtils::ProjectUpsample(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_cloud, cv::Mat &in_image) {
  // Down sample point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr filter_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  CloudVoxelFilter(in_cloud, filter_cloud, 0.1);

  // Transform point cloud to image
  std::vector<GEOM_FADE2D::Point2> points =
      TransformCloudToImage(filter_cloud, in_image.size());

  // Unsampling using Delaunay Triangulation
  DelaunayUpsampling(points, in_image);
}

void CommonUtils::ShowDetectionResult(const cv::Mat &lane_bev_image,
                                      cv::Mat &raw_bev_image) {
  for (int row = 0; row < lane_bev_image.rows; ++row) {
    for (int col = 0; col < lane_bev_image.cols; ++col) {
      if (lane_bev_image.at<uchar>(row, col) == 255) {
        raw_bev_image.at<cv::Vec3b>(row, col)[1] = 255;
      }
    }
  }
}

cv::Point CommonUtils::VeloPointToBEVImage(const pcl::PointXYZ &pt) {
  Eigen::MatrixXf trans_velo_to_bev_image =
      trans_road_to_bev_image_ * trans_cam_to_road_ * trans_velo_to_cam_;
  Eigen::Vector4f velo_point(pt.x, pt.y, pt.z, 1.0);
  Eigen::VectorXf bev_point = trans_velo_to_bev_image * velo_point;
  return cv::Point(bev_point(0), bev_point(1));
}

void CommonUtils::CloudVoxelFilter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud, float voxel_size) {
  pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
  voxel_grid.setInputCloud(in_cloud);
  voxel_grid.setLeafSize((float)voxel_size, (float)voxel_size,
                         (float)voxel_size);
  voxel_grid.filter(*out_cloud);
}

std::vector<GEOM_FADE2D::Point2> CommonUtils::TransformCloudToImage(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud,
    const cv::Size &image_size) {
  // Transformed points
  std::vector<GEOM_FADE2D::Point2> transformed_points;

  // Transform matrix from velo to image
  Eigen::MatrixXf trans_velo_to_image =
      trans_rectcam_to_image_ * trans_cam_to_rectcam_ * trans_velo_to_cam_;

#pragma omp parallel for num_threads(8)
  // Transform all point
  for (int i = 0; i < in_cloud->points.size(); ++i) {
    const auto &pt = in_cloud->points[i];
    // Transform point from velo coordinates to the image plane
    Eigen::Vector4f cloud_point(pt.x, pt.y, pt.z, 1);
    Eigen::Vector3f image_point3 = trans_velo_to_image * cloud_point;
    Eigen::Vector2f image_point2 = image_point3.colwise().hnormalized();
    // Check if image point is valid
    if (IsPointInImageView(image_point2(1), image_point2(0), image_size.height,
                           image_size.width)) {
#pragma omp critical
      transformed_points.push_back(
          GEOM_FADE2D::Point2(image_point2(0), image_point2(1)));
    }
  }

  return transformed_points;
}

void CommonUtils::DelaunayUpsampling(
    const std::vector<GEOM_FADE2D::Point2> &points, cv::Mat &image) {
  // Insert all points to FADE2D
  GEOM_FADE2D::Fade_2D *pDt = new GEOM_FADE2D::Fade_2D();
  pDt->insert(points);
#pragma omp parallel for num_threads(8)
  // Loop over all image points
  for (int i = 180; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      // Get a pointer to a triangle which contains point
      GEOM_FADE2D::Triangle2 *locatetran =
          pDt->locate(GEOM_FADE2D::Point2(j, i));
      if (locatetran != NULL) {
        // Check if valid
        if (locatetran->getArea2D() > 100.0) continue;
        if (sqrt(locatetran->getMaxSqEdgeLen2D()) > 20.0) continue;
        image.at<uchar>(i, j) = 255;
      }
    }
  }
}

int CommonUtils::FileLinesNumber(std::ifstream &file) {
  std::string line;
  int i;
  for (i = 0; std::getline(file, line); ++i)
    ;
  file.clear();
  file.seekg(0, std::ios::beg);
  return i;
}

inline bool CommonUtils::IsPointInImageView(const int &row, const int &col,
                                            const int &height,
                                            const int &width) {
  return row >= 0 && row < height && col >= 0 && col < width;
}