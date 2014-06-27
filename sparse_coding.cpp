#include "sparse_coding.h"
#include "utility.h"

using namespace gentech;

static inline void maxValueAndPos(cv::Mat& grad, cv::Mat& x, double& ma, int& mi)
{
	CV_Assert(grad.size() == x.size());
	CV_Assert(grad.type() == x.type() && grad.type() == CV_64F);
	ma = 0; mi = 0;
	double* pgrad = (double*)grad.data;
	double* px = (double*)x.data;
	for (int i = 0, s = grad.cols; i < s; ++i) {
		if ((*px == 0) && (std::abs(*pgrad) > ma)) {
			ma = std::abs(*pgrad);
			mi = i;
		}
		px++; pgrad++;
	}
}

static inline void matMaskAssign(cv::Mat& src, std::vector<double>& dst, std::vector<int>& a)
{
	double* pdst = (double*)(&dst[0]);
	if (src.cols == 1) {
		for (int i = 0; i < (int)a.size(); ++i) {
			*pdst++ = src.at<double>(a[i], 0);
		}
		return;
	}
	if (src.rows == 1) {
		for (int i = 0; i < (int)a.size(); ++i) {
			*pdst++ = src.at<double>(0, a[i]);
		}
		return;
	}
	for (int i = 0; i < (int)a.size(); ++i) {
		for (int j = 0; (int)j < a.size(); ++j) {
			*pdst++ = src.at<double>(a[i], a[j]);
		}
	}
}

static inline bool operator == (cv::Mat& a, double b)
{
	CV_Assert(a.type() == CV_64F);
	int i = 0, s = a.cols;
	double* pa = (double*)a.data;
	for (; i < s; ++i) {
		if (a.at<double>(0, i) != b) return false;
		if (pa[i] != b) return false;
	}
	return true;
}

static inline void nonZeroMask(cv::Mat& x, std::vector<int>& a)
{
	double* px = (double*)x.data;
	int i = 0, s = x.cols;
	for (; i < s; ++i) {
		if (px[i] != 0) a.push_back(i);
	}
}

static inline std::vector<int> find(std::vector<double>& m, int size)
{
	std::vector<int> idx;
	for (int i = 0; i < size; ++i) {
		if (m[i] != 0) idx.push_back(i);
	}
	return idx;
}

static void solve(std::vector<double>& Aa, std::vector<double>& vect,
		  std::vector<double>& x_new, int size)
{
	cv::Mat Aa_mat(size, size, CV_64F, &(Aa[0]));
	cv::Mat vect_mat(size, 1, CV_64F, &(vect[0]));
	cv::Mat x_new_mat(size, 1, CV_64F, &(x_new[0]));
	cv::solve(Aa_mat, vect_mat, x_new_mat);
}

static double os_calculate(std::vector<double>& Aa, std::vector<double>& x_s,
			   std::vector<double>& ba, std::vector<int>& idx, int AaWidth)
{
	int size = (int)idx.size();
	cv::Mat Aa_mat(size, size, CV_64F);
	cv::Mat x_s_mat(size, 1, CV_64F);
	cv::Mat ba_mat(size, 1, CV_64F);

	for (int r = 0; r < size; ++r) {
		for (int c = 0; c < size; ++c) {
			Aa_mat.at<double>(r, c) = Aa[idx[r] * AaWidth + idx[c]];
		}
		x_s_mat.at<double>(r, 0) = x_s[idx[r]];
		ba_mat.at<double>(r, 0) = ba[idx[r]];
	}
	
	cv::Mat res_mat = x_s_mat.t() * (Aa_mat * x_s_mat * 0.5 + ba_mat);
	double res = res_mat.at<double>(0, 0);
	return res;
}

void L1QP_FeatureSign_yang(cv::Mat& A, cv::Mat& b, cv::Mat& x, double lambda = 0.15)
{

	static double EPS = 1.0e-9;
	static cv::Mat grad(1, 1024, CV_64F);
	double ma = 0;
	int mi = 0;

	//x = cv::Mat::zeros(1, A.rows, CV_64F);
	CV_Assert(x.cols == A.rows && x.rows == 1);
	b.copyTo(grad);
	maxValueAndPos(grad, x, ma, mi);

	double* px = (double*)x.data;
	double* pgrad = (double*)grad.data;
	std::vector<int> xFlag;
	static std::vector<double> Aa(4096), ba(512), xa(512);
	static std::vector<double> vect(512), x_new(512), xa_x_new(512);
	static std::vector<double> x_min(512), d(512), t(512);
	static std::vector<double> x_s(512);
	while (true) {
		if (pgrad[mi] > lambda + EPS) {
			px[mi] = (lambda - pgrad[mi]) / A.at<double>(mi, mi);
			// based on the definition of maxValueAndPos, mi could not be in the xFlag
			xFlag.push_back(mi);
		}
		else if (pgrad[mi] < -lambda - EPS) {
			px[mi] = (-lambda - pgrad[mi]) / A.at<double>(mi, mi);
			// based on the definition of maxValueAndPos, mi could not be in the xFlag
			xFlag.push_back(mi);
		}
		else if (xFlag.size() == 0) {
			break;
		}
		while (true) {
			// the elements in x may be replaced within the loop
			// here \a is necessary, can not just use \xFlag
			std::vector<int> a;
			nonZeroMask(x, a);  
			int size = (int)a.size();
			matMaskAssign(A, Aa, a);
			matMaskAssign(b, ba, a);
			matMaskAssign(x, xa, a);
			for (int i = 0; i < size; ++i) {
				vect[i] = xa[i] > 0 ? (-lambda - ba[i]) : (lambda - ba[i]);
			}
			solve(Aa, vect, x_new, size);
			std::vector<int> idx = find(x_new, size);
			double o_new = 0.0, o_new_tmp = 0.0;
			for (std::size_t i = 0; i < idx.size(); ++i) {
				o_new = o_new + (vect[idx[i]] / 2 + ba[idx[i]]) * x_new[idx[i]];
				o_new_tmp += std::abs(x_new[idx[i]]);
			}
			o_new += lambda * o_new_tmp;
			for (int i = 0; i < size; ++i) {
				xa_x_new[i] = (xa[i] * x_new[i] <= 0) ? 1 : 0;
			}
			std::vector<int> s = find(xa_x_new, size);
			double loss = 0;
			if (s.empty()) {
				for (int i = 0; i < size; ++i) {
					px[a[i]] = x_new[i];
				}
				loss = o_new;
				break;
			}
			for (int i = 0; i < size; ++i) x_min[i] = x_new[i];
			double o_min = o_new;
			for (int i = 0; i < size; ++i) {
				d[i] = x_new[i] - xa[i];
				t[i] = d[i] / xa[i];
			}
			for (std::size_t i = 0; i < s.size(); ++i) {
				for (int j = 0; j < size; ++j) {
					x_s[j] = xa[j] - d[j] / t[s[i]];
				}
				x_s[s[i]] = 0;
				idx = find(x_s, size);
				double o_s = 0, o_s_tmp = 0;
				o_s = os_calculate(Aa, x_s, ba, idx, size);
				for (std::size_t j = 0; j < idx.size(); ++j) {
					o_s_tmp += std::abs(x_s[idx[j]]);
				}
				o_s += lambda * o_s_tmp;
				if (o_s < o_min) {
					for (int j = 0; j < size; ++j) x_min[j] = x_s[j];
					o_min = o_s;
				}
			}
			for (int i = 0; i < size; ++i) {
				px[a[i]] = x_min[i];
			}
			loss = o_min;
		}
		b.copyTo(grad);
		// A is symmetric, A.row(i) == A.col(i)
		for (int i = 0; i < (int)xFlag.size(); ++i) {
			grad = grad + A.row(xFlag[i]) * px[xFlag[i]];
		}
		maxValueAndPos(grad, x, ma, mi);
		if (ma <= lambda + EPS) break;
	}
}

/************************* L1QP_FeatureSign_yang end ***************************/


void absMat(cv::Mat& m)
{
	CV_Assert(m.type() == CV_64F);
	double* pdata = (double*)m.data;
	for (int i = 0, s = m.rows * m.cols; i < s; ++i) {
		*pdata = std::abs(*pdata);
		pdata++;
	}
}

// which sub region each sift point belongs to
void siftCenterPosition(int height, int width, 
			std::vector<double>& x_distribution, 
			std::vector<double>& y_distribution, 
			int patchSize = 16, int gridSpacing = 6)
{
	int remX = (width - patchSize) % gridSpacing;
	int offsetX = cvFloor(remX * 1.0 / 2) + 1;
	int remY = (height - patchSize) % gridSpacing;
	int offsetY = cvFloor(remY * 1.0 / 2) + 1;
	int num_patches = ((width - patchSize + 1 - offsetX) / gridSpacing + 1) *
		((height - patchSize + 1 - offsetY) / gridSpacing + 1);
	x_distribution.resize(num_patches);
	y_distribution.resize(num_patches);
	double halfPathchSize = patchSize * 1.0 / 2 - 0.5;
	int index = 0;
	for (int x = offsetX - 1; x <= width - patchSize + 1; x += gridSpacing) {
		for (int y = offsetY - 1; y <= height - patchSize + 1; y += gridSpacing) {
			x_distribution[index] = x + halfPathchSize;
			y_distribution[index] = y + halfPathchSize;
			index++;
		}
	}
	CV_Assert(index == num_patches);
}

void siftCenterDistribution(int height, int width,
			    std::vector<double>& x_distribution,
			    std::vector<double>& y_distribution,
			    std::vector<int>& idxBin,
			    int pyramid)
{
	CV_Assert(x_distribution.size() == idxBin.size());
	double wUnit = width * 1.0 / pyramid;
	double yUnit = height * 1.0 / pyramid;
	for (std::size_t i = 0; i < idxBin.size(); ++i) {
		int xBin = cvCeil((x_distribution[i] + 1) / wUnit);
		int yBin = cvCeil((y_distribution[i] + 1) / yUnit);
		idxBin[i] = (yBin - 1) * pyramid + xBin - 1;
	}
}

void sparseCodingEachLevel(cv::Mat& sc_codes, std::vector<int>& idxBin, int pyramid, cv::Mat& beta)
{
	pyramid = pyramid * pyramid;
	CV_Assert(beta.cols == pyramid && beta.rows == sc_codes.rows);
	std::vector<double> maxLevel(pyramid, 0.);
	for (int r = 0; r < sc_codes.rows; ++r) {
		double* pcodes = sc_codes.ptr<double>(r);
		for (int c = 0; c < sc_codes.cols; ++c) {
			if (pcodes[c] > maxLevel[idxBin[c]]) maxLevel[idxBin[c]] = pcodes[c];
		}
		double* pbeta = beta.ptr<double>(r);
		for (int i = 0; i < pyramid; ++i) {
			pbeta[i] = maxLevel[i];
			maxLevel[i] = 0.;
		}
	}
}

inline void betaMatPosProcess(cv::Mat& beta)
{
	double* pbeta = (double*)beta.data;
	double sum = 0.;
	int s = beta.rows * beta.cols;
	for (int i = 0; i < s; ++i) {
		sum += ((*pbeta) * (*pbeta));
		pbeta++;
	}
	sum = std::sqrt(sum);
	pbeta = (double*)beta.data;
	for (int i = 0; i < s; ++i) *pbeta++ /= sum;
	beta = beta.t();
	beta = beta.reshape(0, s);
}


/************************* help function end ***************************/

CSparseCoding::CSparseCoding(const char* dicMatFilePath, const char* varName)
{
	readMatFile(dicMatFilePath, varName, m_B);
}

void CSparseCoding::sc_pooling(cv::Mat& siftArr, cv::Size imgSize, cv::Mat& betaMat, double gamma)
{	
	static double beta = 1e-4;
	static cv::Mat A = m_B.t() * m_B + 2 * beta * cv::Mat::eye(m_B.cols, m_B.cols, CV_64F);

	cv::Mat Q = -1 * siftArr * m_B;
	cv::Mat sc_codes = cv::Mat::zeros(siftArr.rows, A.rows, CV_64F);
	for (int i = 0; i < sc_codes.rows; ++i) {
		cv::Mat Qrow = Q.row(i);
		cv::Mat sc_codesrow = sc_codes.row(i);
		//L1QP_FeatureSign_yang(A, Q.row(i), sc_codes.row(i));
		L1QP_FeatureSign_yang(A, Qrow, sc_codesrow);
	}
	sc_codes = sc_codes.t();
	absMat(sc_codes);
	
	std::vector<double> x_distribution, y_distribution;
	siftCenterPosition(imgSize.height, imgSize.width, x_distribution, y_distribution);
	std::vector<int> idxBin(x_distribution.size());
	
	betaMat.create(sc_codes.rows, 21, CV_64F);
	int pyramid[3] = { 1, 2, 4 };
	for (int level = 0; level < 3; ++level){
		siftCenterDistribution(imgSize.height, imgSize.width,
				       x_distribution, y_distribution, idxBin, pyramid[level]);
		int startCol = 0;
		if (level == 1) startCol = 1;
		if (level == 2) startCol = 5;
		int endCol = startCol + pyramid[level] * pyramid[level];
		//sparseCodingEachLevel(sc_codes, idxBin, pyramid[level], betaMat.colRange(startCol, endCol));
		cv::Mat betaMatRange = betaMat.colRange(startCol, endCol);
		sparseCodingEachLevel(sc_codes, idxBin, pyramid[level], betaMatRange);
	}
	betaMatPosProcess(betaMat);
}
