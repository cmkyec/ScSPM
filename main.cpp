#include <iostream>
#include <fstream>
#include "scspm.h"

// 将mat数据存储至txt文件中，在matlab中import data,以与matlab数据比较。
void outputMatrix(cv::Mat& mat, const char* filename)
{
	std::ofstream outfile(filename);
	outfile << cv::format(mat, "csv") << std::endl;
}

/*
int main()
{
	cv::Mat img = cv::imread("./testRes/car1.jpg");
	gentech::CDenseSIFT denseSIFT;
	denseSIFT.init();  // 计算sift特征前，必须调用一次init
	
	// sift descriptor
	cv::Mat siftArr;
	denseSIFT.CalculateSiftDescriptor(img, siftArr);

	// sparse coding
	const char* matFilePath = "./testRes/dict_Caltech101_1024.mat";
	gentech::CSparseCoding sparseCoding(matFilePath);
	cv::Mat betaMat;
	int64 start = cv::getTickCount();
	sparseCoding.sc_pooling(siftArr, img.size(), betaMat);
	int64 end = cv::getTickCount();
	std::cout << (end - start) * 1.0 / cv::getTickFrequency() << std::endl;
	system("pause");
	return 0;
}
*/

int main()
{
	const char* dictMatPath = "./testRes/dict_Caltech101_1024.mat";
	const char* dictMatName = "B";
	const char* wMatPath = "./testRes/w.mat";
	const char* wMatName = "w";
	const char* bMatPath = "./testRes/b.mat";
	const char* bMatName = "b";
	gentech::CScSPM scspm(dictMatPath, dictMatName, wMatPath, wMatName, bMatPath, bMatName);
	cv::Mat img = cv::imread("./testRes/image_0012.jpg");
	cv::Mat res = scspm.classify(img);
	system("pause");
	return 0;
}

