#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <opencv2/opencv.hpp>
#include <matio.h>

/** @brief read matric from the matlab mat file.
 * 
 *  @param matFilePath the file path of the mat file
 *  @param varName the variable name which saved in the mat file
 *  @param mat the opencv mat save the matrix data of the matlab mat in the mat file
 */
void readMatFile(const char* matFilePath, const char* varName, cv::Mat& mat);

// just for test, ignore the following
struct nonZeroElem
{
	int pos;
	double value;
};

void findNonZeroElems(cv::Mat& mat, std::vector<nonZeroElem>& elems);

#endif /* _UTILITY_H_ */
