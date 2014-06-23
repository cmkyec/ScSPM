#include <iostream>
#include <fstream>
#include "scspm.h"

// output opencv matrix into the text file, to compare with the matlab result
void outputMatrix(cv::Mat& mat, const char* filename)
{
	std::ofstream outfile(filename);
	outfile << cv::format(mat, "csv") << std::endl;
}

int main()
{
	cv::Mat img = cv::imread("./testRes/image_0001.jpg");
	gentech::CDenseSIFT denseSIFT;
	denseSIFT.init();  // 
	
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
	std::cout<<"sc_pooling time is: "<<(end - start) * 1.0 / cv::getTickFrequency()<<std::endl;

	return 0;
}

/*
int main()
{
	const char* dictMatPath = "./testRes/dict_Caltech101_1024.mat";
	const char* dictMatName = "B";
	const char* wMatPath = "./testRes/w.mat";
	gentech::CScSPM scspm(dictMatPath, dictMatName, wMatPath);
	cv::Mat img = cv::imread("./testRes/image_0012.jpg");
	double score = scspm.classify(img);
	std::cout<<"score is: "<<score<<std::endl;
	return 0;
}
*/
