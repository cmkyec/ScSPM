/**
 * @brief sparse coding for the dense sift descriptor
 */

#ifndef _SPARSE_CODING_H_
#define _SPARSE_CODING_H_

#include <opencv2/opencv.hpp>

namespace gentech
{

class CSparseCoding
{
public:
	CSparseCoding(const char* dicMatFilePath, const char* varName = "B");

	CSparseCoding() {}

	void sc_pooling(cv::Mat& siftArr, cv::Size imgSize, cv::Mat& betaMat, double gamma = 0.15);

	~CSparseCoding() {}
private:
	cv::Mat m_B;  // trained dictionary
};

}

#endif /* _SPARSE_CODING_H_ */
