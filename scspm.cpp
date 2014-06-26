#include "scspm.h"
#include "utility.h"

using namespace gentech;

CScSPM::CScSPM(const char* dictMatPath, const char* dictMatName,
	       const char* wMatPath, const char* wMatName, 
	       const char* bMatPath, const char* bMatName) : m_sparseCoding(dictMatPath, dictMatName)
{
	m_denseSIFT.init();
	readMatFile(wMatPath, wMatName, m_w);
	readMatFile(bMatPath, bMatName, m_b);
}

cv::Mat CScSPM::classify(cv::Mat& img)
{
	if (img.cols > 300 || img.rows > 300) return cv::Mat();
	cv::Mat siftArr;
	m_denseSIFT.CalculateSiftDescriptor(img, siftArr);
	
	cv::Mat betaMat;
	m_sparseCoding.sc_pooling(siftArr, img.size(), betaMat);

	CV_Assert(betaMat.rows == m_w.rows);
	cv::Mat result = betaMat.t() * m_w;
	if (result.rows == m_b.rows) {
		result = result + m_b;
	}
	else {
		result = result + m_b.t();
	}
	return result;
}
