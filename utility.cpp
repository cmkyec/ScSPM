#include "utility.h"
#include <iostream>
#include <cassert>

void readMatFile(const char* matFilePath, const char* varName, cv::Mat& mat)
{
	mat_t* matfp = NULL;
	matfp = Mat_Open(matFilePath, MAT_ACC_RDONLY);
	if (!matfp) {
		std::cerr << "can not read the mat file: " << matFilePath << std::endl;
		return;
	}
	matvar_t* matVar = NULL;
	matVar = Mat_VarRead(matfp, varName);
	if (!matVar) {
		std::cerr << "can not read the variable: " << varName << std::endl;
		Mat_Close(matfp);
		return;
	}
	assert(matVar->data_type == MAT_T_DOUBLE);
	assert(matVar->rank == 2);
	int height = (int)matVar->dims[0];
	int width = (int)matVar->dims[1];
	mat.create(height, width, CV_64F);
	// matlab column major form, opencv row major form
	cv::Mat tmp(width, height, CV_64F);
	memcpy(tmp.data, matVar->data, matVar->nbytes);
	mat = tmp.t();

	Mat_VarFree(matVar);
	Mat_Close(matfp);
}

void findNonZeroElems(cv::Mat& mat, std::vector<nonZeroElem>& elems)
{
	for (int i = 0; i < mat.cols; ++i) {
		if (mat.at<double>(0, i) != 0) {
			struct nonZeroElem elem;
			elem.pos = i;
			elem.value = mat.at<double>(0, i);
			elems.push_back(elem);
		}
	}
}