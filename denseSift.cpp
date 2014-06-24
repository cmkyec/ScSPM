#include "denseSift.h"

using namespace gentech;

static int g_maxImSize = 300;
// dense sample pixel position
static const int g_gridSpacing = 6;
static const int g_patchSize = 16;
// image filter kernel
static const double g_sigma = 0.8;

// sift descriptor extract
static const int g_num_angles = 8;
static const double g_angles[8] = { 0., 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978 };
static const int g_num_bins = 4;
static const int g_num_samples = g_num_bins * g_num_bins;
static const int g_alpha = 9;

static void gen_dgauss(double sigma, cv::Mat& GX, cv::Mat& GY)
{
	int f_wid = 4 * cvCeil(sigma) + 1;
	cv::Mat kernel_separate = cv::getGaussianKernel(f_wid, sigma, CV_64F);
	cv::Mat kernel = kernel_separate * kernel_separate.t();
	GX.create(kernel.size(), kernel.type());
	GY.create(kernel.size(), kernel.type());
	for (int r = 0; r < kernel.rows; ++r) {
		for (int c = 0; c < kernel.cols; ++c) {
			if (c == 0) {
				GX.at<double>(r, c) = kernel.at<double>(r, c + 1) - kernel.at<double>(r, c);
			}
			else if (c == kernel.cols - 1) {
				GX.at<double>(r, c) = kernel.at<double>(r, c) - kernel.at<double>(r, c - 1);
			}
			else {
				GX.at<double>(r, c) = (kernel.at<double>(r, c + 1) -
					kernel.at<double>(r, c - 1)) / 2;
			}
			if (r == 0) {
				GY.at<double>(r, c) = kernel.at<double>(r + 1, c) - kernel.at<double>(r, c);
			}
			else if (r == kernel.rows - 1) {
				GY.at<double>(r, c) = kernel.at<double>(r, c) - kernel.at<double>(r - 1, c);
			}
			else {
				GY.at<double>(r, c) = (kernel.at<double>(r + 1, c) -
					kernel.at<double>(r - 1, c)) / 2;
			}
		}
	}
	GX = GX * 2 / cv::sum(cv::abs(GX))[0];
	GY = GY * 2 / cv::sum(cv::abs(GY))[0];
}

void initWeightMatrix(cv::Mat& weights)
{
	weights.create(16, 256, CV_64F);
	cv::Mat weights_x(16, 256, CV_64F);
	cv::Mat weights_y(16, 256, CV_64F);

	double sample_x_t[16] = { 1.5, 1.5, 1.5, 1.5,
				  5.5, 5.5, 5.5, 5.5,
				  9.5, 9.5, 9.5, 9.5,
				  13.5, 13.5, 13.5, 13.5 };
	double sample_y_t[16] = { 1.5, 5.5, 9.5, 13.5,
				  1.5, 5.5, 9.5, 13.5,
				  1.5, 5.5, 9.5, 13.5,
				  1.5, 5.5, 9.5, 13.5 };

	double* pweight_x = (double*)weights_x.data;
	double* pweight_y = (double*)weights_y.data;
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			for (int k = 0; k < 16; ++k) {
				double tmp = std::abs(j * 1.f - sample_x_t[i]) / 4;
				tmp = 1 - tmp;
				*pweight_x++ = tmp > 0 ? tmp : 0;
				tmp = std::abs(k * 1.f - sample_y_t[i]) / 4;
				tmp = 1 - tmp;
				*pweight_y++ = tmp > 0 ? tmp : 0;
			}
		}
	}
	weights = weights_x.mul(weights_y);
}

enum ConvolutionType
{
	/* Return the full convolution, including border */
	CONVOLUTION_FULL,
	/* Return only the part that corresponds to the original image */
	CONVOLUTION_SAME,
	/* Return only the submatrix containing elements that were not influenced by the border */
	CONVOLUTION_VALID
};
static void filter2(cv::Mat& src, cv::Mat& dst, cv::Mat& kernel, int type)
{
	cv::Mat source = src;
	//if (CONVOLUTION_FULL == type) {
	//	source = cv::Mat();
	//	const int additionalRows = kernel.rows - 1, additionalCols = kernel.cols - 1;
	//	cv::copyMakeBorder(src, source, (additionalRows + 1) / 2, additionalRows / 2,
	//		(additionalCols + 1) / 2, additionalCols / 2, cv::BORDER_CONSTANT, cv::Scalar(0));
	//}

	cv::Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
	int borderMode = cv::BORDER_CONSTANT;
	cv::filter2D(source, dst, CV_64F, kernel, anchor, 0, borderMode);

	//if (CONVOLUTION_VALID == type) {
	//	dst = dst.colRange((kernel.cols - 1) / 2, dst.cols - kernel.cols / 2)
	//		.rowRange((kernel.rows - 1) / 2, dst.rows - kernel.rows / 2);
	//}
}

static void magnitude(const cv::Mat& I_X, const cv::Mat& I_Y, cv::Mat& I_mag)
{
	CV_Assert(I_X.type() == CV_64F && I_X.type() == I_Y.type());

	I_mag = I_X.mul(I_X) + I_Y.mul(I_Y);
	int s = I_mag.rows * I_mag.cols;
	double* p = (double*)I_mag.data;
	for (int i = 0; i < s; ++i) {
		*p = cv::sqrt(*p);
		p++;
	}
}

static void orientation(const cv::Mat& I_X, const cv::Mat& I_Y, cv::Mat& I_theta)
{
	CV_Assert(I_X.type() == CV_64F && I_X.type() == I_Y.type());

	I_theta.create(I_X.size(), I_X.type());
	const double* pdx = (double*)I_X.data;
	const double* pdy = (double*)I_Y.data;
	double* ptheta = (double*)I_theta.data;
	for (int i = 0, s = I_X.rows * I_X.cols; i < s; ++i) {
		*ptheta++ = atan2(*pdy++, *pdx++);
	}
}

static void weightOrientation(const cv::Mat& I_theta, const cv::Mat& I_mag, cv::Mat& I_orientation)
{
	CV_Assert(I_theta.size() == I_mag.size());
	CV_Assert(I_theta.type() == CV_64F && I_theta.type() == I_mag.type());
	I_orientation.create(I_theta.size(), CV_64FC(g_num_angles));

	const double* ptheta = (double*)I_theta.data;
	const double* pmag = (double*)I_mag.data;
	double* phist = (double*)I_orientation.data;
	for (int i = 0, s = I_theta.cols * I_theta.rows; i < s; ++i) {
		for (int j = 0; j < g_num_angles; ++j) {
			double tmp = std::pow((std::cos(*ptheta - g_angles[j])), g_alpha);
			if (tmp < 0) tmp = 0;
			*phist++ = tmp * (*pmag);
		}
		ptheta++; pmag++;
	}
}

static void siftForSinglePoint(const cv::Mat& weights, const std::vector<cv::Mat>& I_orientations, 
			       const cv::Point& point, cv::Mat const& descriptor)
{
	CV_Assert(weights.size() == cv::Size(256, 16));
	CV_Assert(descriptor.cols == 128);

	cv::Rect rect(point.x, point.y, 16, 16);
	cv::Mat descriptor_tmp(g_num_angles, g_num_samples, CV_64F);
	double* pdescriptor_tmp = (double*)descriptor_tmp.data;
	for (int i = 0; i < g_num_angles; ++i) {
		cv::Mat hist = I_orientations[i](rect).t();
		for (int r = 0; r < weights.rows; ++r) {
			double sum = 0;
			const double* phist = (double*)hist.data;
			const double* pweights = weights.ptr<double>(r);
			for (int c = 0; c < weights.cols; ++c) {
				sum += (*pweights++) * (*phist++);
			}
			*pdescriptor_tmp++ = sum;
		}
	}
	descriptor_tmp = descriptor_tmp.t();
	memcpy(descriptor.data, descriptor_tmp.data, descriptor_tmp.step * descriptor_tmp.rows);
}

/*************************** help function end *********************************/

void CDenseSIFT::init()
{
	initWeightMatrix(m_weights);
	gen_dgauss(g_sigma, m_G_X, m_G_Y);
}

void CDenseSIFT::sp_find_sift_grid(cv::Mat& img, cv::Mat& siftArr)
{
	cv::Mat I;
	if (img.channels() == 3) {
		cv::cvtColor(img, I, CV_BGR2GRAY);
	}
	else {
		img.copyTo(I);
	}
	I.convertTo(I, CV_64F, 1.0 / 255);	

	/*
	if (std::max(I.rows, I.cols) > g_maxImSize) {
		double scale = g_maxImSize * 1.0 / std::max(I.rows, I.cols);
		cv::resize(I, I, cv::Size(0, 0), scale, scale, cv::INTER_CUBIC);
	}
	*/
	if (std::max(I.rows, I.cols) > g_maxImSize) return;
	
	cv::Mat I_X, I_Y;
	filter2(I, I_X, m_G_X, CONVOLUTION_SAME);
	filter2(I, I_Y, m_G_Y, CONVOLUTION_SAME);

	cv::Mat I_mag, I_theta;
	magnitude(I_X, I_Y, I_mag);
	orientation(I_X, I_Y, I_theta);

	cv::Mat I_orientation;
	weightOrientation(I_theta, I_mag, I_orientation);
	std::vector<cv::Mat> I_orientations;
	cv::split(I_orientation, I_orientations);

	int remX = (I.cols - g_patchSize) % g_gridSpacing;
	int offsetX = cvFloor(remX * 1.0 / 2) + 1;
	int remY = (I.rows - g_patchSize) % g_gridSpacing;
	int offsetY = cvFloor(remY * 1.0 / 2) + 1;
	int num_patches = ((I.cols - g_patchSize + 1 - offsetX) / g_gridSpacing + 1) *
		((I.rows - g_patchSize + 1 - offsetY) / g_gridSpacing + 1);
	siftArr.create(num_patches, g_num_angles * g_num_samples, CV_64F);
	num_patches = 0;
	for (int x = offsetX - 1; x <= I.cols - g_patchSize; x += g_gridSpacing) {
		for (int y = offsetY - 1; y <= I.rows - g_patchSize; y += g_gridSpacing) {
			cv::Point point(x, y);
			siftForSinglePoint(m_weights, I_orientations, point, siftArr.row(num_patches++));
		}
	}
}

void CDenseSIFT::sp_normalize_sift(cv::Mat& siftArr, double threshold)
{
	for (int r = 0; r < siftArr.rows; ++r) {
		cv::Mat row = siftArr.row(r);
		double mag = std::sqrt(cv::sum(row.mul(row))[0]);
		double* pdata = (double*)row.data;
		if (mag >= threshold) {
			for (int c = 0; c < row.cols; ++c) {
				pdata[c] /= mag;
				if (pdata[c] > 0.2) pdata[c] = 0.2;
			}
			double s = std::sqrt(cv::sum(row.mul(row))[0]);
			for (int c = 0; c < row.cols; ++c) pdata[c] /= s;
		}
		else {
			for (int c = 0; c < row.cols; ++c) {
				pdata[c] /= threshold;
				if (pdata[c] > 0.2) pdata[c] = 0.2;
			}
		}
	}
}

void CDenseSIFT::CalculateSiftDescriptor(cv::Mat& img, cv::Mat& siftArr)
{
	sp_find_sift_grid(img, siftArr);
	sp_normalize_sift(siftArr);
}
