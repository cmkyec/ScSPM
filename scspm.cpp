#include "scspm.h"
#include "utility.h"

using namespace gentech;

CScSPM::CScSPM(const char* dictMatPath, const char* dictMatName,
	const char* wMatPath, double b) : m_sparseCoding(dictMatPath, dictMatName)
{
	m_denseSIFT.init();
	readMatFile(wMatPath, "w", m_w);
	m_b = b;
}

double CScSPM::classify(cv::Mat& img)
{
	if (img.rows > 300 || img.cols > 300) return 0;
	cv::Mat siftArr;
	m_denseSIFT.CalculateSiftDescriptor(img, siftArr);
	
	cv::Mat betaMat;
	m_sparseCoding.sc_pooling(siftArr, img.size(), betaMat);

	double result = 0;
	double* pbetaMat = (double*)betaMat.data;
	double* pw = (double*)m_w.data;
	int s = betaMat.rows;
	for (int i = 0; i < s; ++i) {
		result += (*pbetaMat++) * (*pw++);
	}
	return result + m_b;
}
