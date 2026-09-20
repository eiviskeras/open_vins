/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "TrackCheckerboard.h"

#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/opencv_lambda_body.h"

#include <opencv2/calib3d.hpp>

using namespace ov_core;

void TrackCheckerboard::feed_new_camera(const CameraData &message) {

  // Error check that we have all the data
  if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != message.masks.size()) {
    PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
    PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
    PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
    PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
    std::exit(EXIT_FAILURE);
  }

  // The corner ids are known, so each image is processed independently
  size_t num_images = message.images.size();
  parallel_for_(cv::Range(0, (int)num_images), LambdaBody([&](const cv::Range &range) {
                  for (int i = range.start; i < range.end; i++) {
                    perform_tracking(message.timestamp, message.images.at(i), message.sensor_ids.at(i), message.masks.at(i));
                  }
                }));
}

void TrackCheckerboard::perform_tracking(double timestamp, const cv::Mat &imgin, size_t cam_id, const cv::Mat &maskin) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Lock this data feed for this camera
  std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

  // Histogram equalize
  cv::Mat img;
  if (histogram_method == HistogramMethod::HISTOGRAM) {
    cv::equalizeHist(imgin, img);
  } else if (histogram_method == HistogramMethod::CLAHE) {
    double eq_clip_limit = 10.0;
    cv::Size eq_win_size = cv::Size(8, 8);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
    clahe->apply(imgin, img);
  } else {
    img = imgin;
  }

  // Select the configured region before detection. The corners are translated
  // back into the full image coordinate frame below.
  const int roi_x = std::max(0, std::min(img.cols - 1, (int)std::floor(detection_roi[0] * img.cols)));
  const int roi_y = std::max(0, std::min(img.rows - 1, (int)std::floor(detection_roi[1] * img.rows)));
  const int roi_width = std::max(1, std::min(img.cols - roi_x, (int)std::ceil(detection_roi[2] * img.cols)));
  const int roi_height = std::max(1, std::min(img.rows - roi_y, (int)std::ceil(detection_roi[3] * img.rows)));
  cv::Mat detection_image = img(cv::Rect(roi_x, roi_y, roi_width, roi_height));

  // If enabled, downsize only the selected region
  cv::Mat img0;
  if (do_downsizing) {
    cv::pyrDown(detection_image, img0, cv::Size(detection_image.cols / 2, detection_image.rows / 2));
  } else {
    img0 = detection_image;
  }

  //===================================================================================
  //===================================================================================

  // Detect the full board. The sector-based detector is sub-pixel accurate; fall back
  // to the classic detector (with a fast pre-check) if it does not find the board.
  std::vector<cv::Point2f> pts;
  bool found = false;
#if CV_MAJOR_VERSION >= 4
  found = cv::findChessboardCornersSB(img0, board_size, pts, cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_ACCURACY);
#endif
  if (!found) {
    pts.clear();
    found = cv::findChessboardCorners(img0, board_size, pts,
                                      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
    if (found) {
      cv::cornerSubPix(img0, pts, cv::Size(5, 5), cv::Size(-1, -1),
                       cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.01));
    }
  }
  found = found && (int)pts.size() == board_size.width * board_size.height;
  rT2 = boost::posix_time::microsec_clock::local_time();

  DetectionResult result;
  result.timestamp = timestamp;
  result.camera_id = cam_id;
  result.found = found;

  std::vector<size_t> ids_new;
  std::vector<cv::KeyPoint> pts_new;

  if (found) {

    // Restore detections to full-image coordinates
    const float coordinate_scale = do_downsizing ? 2.0F : 1.0F;
    for (auto &pt : pts) {
      pt.x = coordinate_scale * pt.x + roi_x;
      pt.y = coordinate_scale * pt.y + roi_y;
    }

    // Refine at full resolution if we detected on the downsized image
    if (do_downsizing) {
      cv::cornerSubPix(img, pts, cv::Size(5, 5), cv::Size(-1, -1),
                       cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.01));
    }

    // Normalize the 180 degree ordering ambiguity: the first corner must be the
    // upper-left one in the image (row direction pointing right)
    const int cols = board_size.width;
    const int rows = board_size.height;
    cv::Point2f row_dir = pts.at(cols - 1) - pts.at(0);
    if (row_dir.x < 0.0F) {
      std::reverse(pts.begin(), pts.end());
      row_dir = pts.at(cols - 1) - pts.at(0);
    }
    cv::Point2f col_dir = pts.at((rows - 1) * cols) - pts.at(0);
    const bool orientation_ok = row_dir.x > 0.0F && col_dir.y > 0.0F;

    // Quality gates: full-board homography consistency and apparent size
    result.fit_error_px = grid_fit_error(cam_id, pts);
    result.min_spacing_px = min_neighbour_spacing(pts);
    result.accepted = orientation_ok && result.fit_error_px >= 0.0 && result.fit_error_px <= max_fit_error &&
                      result.min_spacing_px >= min_spacing;

    if (!result.accepted) {
      PRINT_DEBUG("[CHECKERBOARD]: cam %zu board rejected (orientation %d, fit %.3f px, spacing %.1f px)\n", cam_id, (int)orientation_ok,
                  result.fit_error_px, result.min_spacing_px);
    } else {
      for (size_t i = 0; i < pts.size(); i++) {
        const cv::Point2f &pt = pts.at(i);
        if (pt.x < 0.0F || pt.y < 0.0F || pt.x >= (float)maskin.cols || pt.y >= (float)maskin.rows)
          continue;
        // NOTE: mask has max value of 255 (white) if it should be masked out
        if (maskin.at<uint8_t>((int)pt.y, (int)pt.x) > 127)
          continue;
        cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(pt);
        size_t tmp_id = i; // row * cols + col
        database->update_feature(tmp_id, timestamp, cam_id, pt.x, pt.y, npt_l.x, npt_l.y);
        cv::KeyPoint kpt;
        kpt.pt = pt;
        ids_new.push_back(tmp_id);
        pts_new.push_back(kpt);
      }
    }
  }

  // Retain one diagnostic result for every image, including images with no detection
  {
    std::lock_guard<std::mutex> lock(detection_results_mtx);
    detection_results.push_back(result);
    if (detection_results.size() > 1000)
      detection_results.pop_front();
  }

  // Move forward in time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_mask_last[cam_id] = maskin;
    ids_last[cam_id] = ids_new;
    pts_last[cam_id] = pts_new;
    corners_last[cam_id] = found ? pts : std::vector<cv::Point2f>();
    accepted_last[cam_id] = result.accepted;
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Timing information
  PRINT_ALL("[TIME-CHECKERBOARD]: %.4f seconds for detection\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-CHECKERBOARD]: %.4f seconds for feature DB update (%d features)\n", (rT3 - rT2).total_microseconds() * 1e-6,
            (int)ids_new.size());
  PRINT_ALL("[TIME-CHECKERBOARD]: %.4f seconds for total\n", (rT3 - rT1).total_microseconds() * 1e-6);
}

double TrackCheckerboard::grid_fit_error(size_t cam_id, const std::vector<cv::Point2f> &corners) const {
  // Undistort into the pinhole image (normalized coordinates scaled by K) so a planar
  // board maps to the image through an exact homography
  const cv::Matx33d K = camera_calib.at(cam_id)->get_K();
  std::vector<cv::Point2f> grid, undist;
  grid.reserve(corners.size());
  undist.reserve(corners.size());
  for (int r = 0; r < board_size.height; r++) {
    for (int c = 0; c < board_size.width; c++) {
      grid.emplace_back((float)c, (float)r);
      cv::Point2f n = camera_calib.at(cam_id)->undistort_cv(corners.at((size_t)(r * board_size.width + c)));
      undist.emplace_back((float)(K(0, 0) * n.x + K(0, 2)), (float)(K(1, 1) * n.y + K(1, 2)));
    }
  }
  cv::Mat H = cv::findHomography(grid, undist, 0);
  if (H.empty())
    return -1.0;
  std::vector<cv::Point2f> proj;
  cv::perspectiveTransform(grid, proj, H);
  double sum_sq = 0.0;
  for (size_t i = 0; i < proj.size(); i++) {
    const cv::Point2f d = proj.at(i) - undist.at(i);
    sum_sq += (double)d.x * d.x + (double)d.y * d.y;
  }
  return std::sqrt(sum_sq / (double)proj.size());
}

double TrackCheckerboard::min_neighbour_spacing(const std::vector<cv::Point2f> &corners) const {
  double min_d = std::numeric_limits<double>::max();
  const int cols = board_size.width;
  const int rows = board_size.height;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      const cv::Point2f &p = corners.at((size_t)(r * cols + c));
      if (c + 1 < cols)
        min_d = std::min(min_d, cv::norm(corners.at((size_t)(r * cols + c + 1)) - p));
      if (r + 1 < rows)
        min_d = std::min(min_d, cv::norm(corners.at((size_t)((r + 1) * cols + c)) - p));
    }
  }
  return min_d;
}

void TrackCheckerboard::display_active(cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2, std::string overlay) {

  // Cache the images to prevent other threads from editing while we viz (which can be slow)
  std::map<size_t, cv::Mat> img_last_cache, img_mask_last_cache;
  std::unordered_map<size_t, std::vector<cv::Point2f>> corners_cache;
  std::unordered_map<size_t, bool> accepted_cache;
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last_cache = img_last;
    img_mask_last_cache = img_mask_last;
    corners_cache = corners_last;
    accepted_cache = accepted_last;
  }

  // Get the largest width and height
  int max_width = -1;
  int max_height = -1;
  for (auto const &pair : img_last_cache) {
    if (max_width < pair.second.cols)
      max_width = pair.second.cols;
    if (max_height < pair.second.rows)
      max_height = pair.second.rows;
  }

  // Return if we didn't have a last image
  if (img_last_cache.empty() || max_width == -1 || max_height == -1)
    return;

  // If the image is "small" thus we should use smaller display codes
  bool is_small = (std::min(max_width, max_height) < 400);

  // If the image is "new" then draw the images from scratch
  bool image_new = ((int)img_last_cache.size() * max_width != img_out.cols || max_height != img_out.rows);
  if (image_new)
    img_out = cv::Mat(max_height, (int)img_last_cache.size() * max_width, CV_8UC3, cv::Scalar(0, 0, 0));

  // Loop through each image, and draw
  int index_cam = 0;
  for (auto const &pair : img_last_cache) {
    cv::Mat img_temp;
    if (image_new)
      cv::cvtColor(img_last_cache[pair.first], img_temp, cv::COLOR_GRAY2RGB);
    else
      img_temp = img_out(cv::Rect(max_width * index_cam, 0, max_width, max_height));
    // Accepted boards in colour order, rejected ones in red
    const auto &corners = corners_cache[pair.first];
    if (!corners.empty()) {
      if (accepted_cache[pair.first]) {
        cv::drawChessboardCorners(img_temp, board_size, corners, true);
      } else {
        for (const auto &pt : corners)
          cv::circle(img_temp, pt, 3, cv::Scalar(0, 0, 255), 1);
      }
    }
    auto txtpt = (is_small) ? cv::Point(10, 30) : cv::Point(30, 60);
    if (overlay == "") {
      cv::putText(img_temp, "CAM:" + std::to_string((int)pair.first), txtpt, cv::FONT_HERSHEY_COMPLEX_SMALL, (is_small) ? 1.5 : 3.0,
                  cv::Scalar(0, 255, 0), 3);
    } else {
      cv::putText(img_temp, overlay, txtpt, cv::FONT_HERSHEY_COMPLEX_SMALL, (is_small) ? 1.5 : 3.0, cv::Scalar(0, 0, 255), 3);
    }
    cv::Mat mask = cv::Mat::zeros(img_mask_last_cache[pair.first].rows, img_mask_last_cache[pair.first].cols, CV_8UC3);
    mask.setTo(cv::Scalar(0, 0, 255), img_mask_last_cache[pair.first]);
    cv::addWeighted(mask, 0.1, img_temp, 1.0, 0.0, img_temp);
    img_temp.copyTo(img_out(cv::Rect(max_width * index_cam, 0, img_last_cache[pair.first].cols, img_last_cache[pair.first].rows)));
    index_cam++;
  }
}
