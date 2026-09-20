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

#ifndef OV_CORE_TRACK_CHECKERBOARD_H
#define OV_CORE_TRACK_CHECKERBOARD_H

#include "TrackBase.h"

#include <array>
#include <deque>

namespace ov_core {

/**
 * @brief Tracking of a single planar checkerboard.
 *
 * The inner corners of one checkerboard (e.g. 5x7 inner corners) are detected with
 * OpenCV's chessboard detector and fed to the feature database as persistent
 * fiducial features with fixed ids `row * cols + col`. They therefore occupy the same
 * id range (< 4 * numaruco) as ArUco corners and are treated by the estimator exactly
 * like ArUco features (never marginalized, own pixel noise).
 *
 * A frame only contributes measurements when the *whole* board is found and passes
 * two quality gates: the RMS residual of a homography fitted between the board grid and
 * the undistorted corners must stay below `max_fit_error_px`, and the smallest spacing
 * between neighbouring corners must exceed `min_spacing_px`.
 *
 * Boards with an even number of squares in both directions (odd x odd inner corners)
 * are symmetric under a 180 degree rotation, so OpenCV's corner ordering may flip
 * between frames. The ordering is normalized so that the first corner is the upper-left
 * one in the image, which is stable for an approximately upright camera.
 */
class TrackCheckerboard : public TrackBase {

public:
  struct DetectionResult {
    double timestamp = 0.0;
    size_t camera_id = 0;
    bool found = false;
    bool accepted = false;
    double fit_error_px = -1.0;
    double min_spacing_px = -1.0;
  };

  /**
   * @brief Public constructor with configuration variables
   * @param cameras camera calibration object which has all camera intrinsics in it
   * @param numaruco max fiducial id range (features below 4*numaruco are treated as fiducials)
   * @param stereo unused (each image is processed independently, ids are known)
   * @param histmethod what type of histogram pre-processing should be done (histogram eq?)
   * @param downsize scale the detection image by 1/2 (corners are refined at full resolution)
   * @param cols number of inner corners per row
   * @param rows number of inner corners per column
   * @param max_fit_error_px reject detections whose grid-homography RMS residual exceeds this
   * @param min_spacing_px reject detections whose smallest neighbour spacing is below this
   * @param roi normalized x, y, width, and height of the image region to scan
   */
  explicit TrackCheckerboard(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numaruco, bool stereo,
                             HistogramMethod histmethod, bool downsize, int cols, int rows, double max_fit_error_px,
                             double min_spacing_px, const std::array<double, 4> &roi = {0.0, 0.0, 1.0, 1.0})
      : TrackBase(cameras, 0, numaruco, stereo, histmethod), board_size(cols, rows), do_downsizing(downsize),
        max_fit_error(max_fit_error_px), min_spacing(min_spacing_px), detection_roi(roi) {
    if (cols < 2 || rows < 2) {
      PRINT_ERROR(RED "[ERROR]: checkerboard needs at least 2x2 inner corners (got %dx%d)\n" RESET, cols, rows);
      std::exit(EXIT_FAILURE);
    }
    if (cols * rows > 4 * numaruco) {
      PRINT_ERROR(RED "[ERROR]: checkerboard corner ids (%d) exceed the fiducial id range 4*num_aruco (%d)\n" RESET, cols * rows,
                  4 * numaruco);
      std::exit(EXIT_FAILURE);
    }
  }

  /**
   * @brief Process a new image
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_new_camera(const CameraData &message) override;

  /** Return and clear all per-frame detector results accumulated since the previous call. */
  std::vector<DetectionResult> consume_detection_results() {
    std::lock_guard<std::mutex> lock(detection_results_mtx);
    std::vector<DetectionResult> results(detection_results.begin(), detection_results.end());
    detection_results.clear();
    return results;
  }

  /**
   * @brief We override the display equation so we can show the board we extract.
   */
  void display_active(cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2, std::string overlay = "") override;

protected:
  /**
   * @brief Process a new monocular image
   * @param timestamp timestamp the new image occurred at
   * @param imgin new cv:Mat grayscale image
   * @param cam_id the camera id that this new image corresponds too
   * @param maskin tracking mask for the given input image
   */
  void perform_tracking(double timestamp, const cv::Mat &imgin, size_t cam_id, const cv::Mat &maskin);

  /**
   * @brief Fit a homography between the board grid and the undistorted corners.
   * @return RMS residual in (undistorted) pixels, or a negative value on failure
   */
  double grid_fit_error(size_t cam_id, const std::vector<cv::Point2f> &corners) const;

  /// Smallest distance between horizontally or vertically adjacent corners
  double min_neighbour_spacing(const std::vector<cv::Point2f> &corners) const;

  // Inner corner grid (cols x rows)
  cv::Size board_size;

  // If we should downsize the image for the detection step
  bool do_downsizing;

  // Quality gates
  double max_fit_error;
  double min_spacing;

  // Normalized x, y, width, and height of the image region used for detection
  std::array<double, 4> detection_roi;

  // Per-frame results consumed by ROS diagnostics. This includes empty scans.
  std::deque<DetectionResult> detection_results;
  std::mutex detection_results_mtx;

  // Last detection per camera (full-image coordinates) for visualization
  std::unordered_map<size_t, std::vector<cv::Point2f>> corners_last;
  std::unordered_map<size_t, bool> accepted_last;
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_CHECKERBOARD_H */
