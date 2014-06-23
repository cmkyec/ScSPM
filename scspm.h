#ifndef _CLASSIFY_H_
#define _CLASSIFY_H_

#include <opencv2/opencv.hpp>
#include "denseSift.h"
#include "sparse_coding.h"

namespace gentech
{

class CScSPM
{
public:
	CScSPM(const char* dictMatPath, const char* dictMatName,
	       const char* wMatPath, double b = 1.1722);

	~CScSPM() {}

	double classify(cv::Mat& img);
private:
	CScSPM() {}
private:
	CDenseSIFT m_denseSIFT;
	CSparseCoding m_sparseCoding;
	cv::Mat m_w;
	double m_b;
};

}


#endif /* _CLASSIFY_H_ */
