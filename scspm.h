#ifndef _CLASSIFY_H_
#define _CLASSIFY_H_

#include <opencv2\opencv.hpp>
#include "denseSift.h"
#include "sparse_coding.h"

namespace gentech
{

class CScSPM
{
public:
	/**
	 * @param dictMatPath 字典mat file系统路径
	 * @param dictMatName 字典mat对应的matlab变量名称
	 * @param wMatPath    权重mat file系统路径
	 * @param wMatName    权重mat对应的matlab变量名称
	 * @param bMatPath    /b mat file系统路径
	 * @param bMatName    /b 对应的matlab变量名称
	 */
	CScSPM(const char* dictMatPath, const char* dictMatName,
	       const char* wMatPath, const char* wMatName, 
	       const char* bMatPath, const char* bMatName);

	~CScSPM() {}

	cv::Mat classify(cv::Mat& img);
private:
	CScSPM() {}
private:
	CDenseSIFT m_denseSIFT;
	CSparseCoding m_sparseCoding;
	cv::Mat m_w;
	cv::Mat m_b;
};

}


#endif /* _CLASSIFY_H_ */