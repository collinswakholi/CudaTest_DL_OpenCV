#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
# include <chrono>
#include <ratio>

using namespace cv;

int main() {
	Mat img = imread("image.png",IMREAD_GRAYSCALE);
	cuda::GpuMat dst, src;
	src.upload(img);

	Mat result;
	auto start = std::chrono::high_resolution_clock::now();//tic

	// cpu
	//Ptr<CLAHE> ptr_clahe = createCLAHE(5.0, Size(8, 8));
	//ptr_clahe->apply(img,result);
	
	//gpu
	Ptr<cuda::CLAHE> ptr_CLAHE = cuda::createCLAHE(5.0, Size(8, 8));
	ptr_CLAHE->apply(src, dst);
	auto stop = std::chrono::high_resolution_clock::now();

	//Mat result;
	dst.download(result);

	auto diff = std::chrono::duration_cast<std::chrono::milliseconds> (stop-start);
	std::chrono::duration<double, std::micro> diff_n = diff;

	std::cout<<"Elapsed time (ms) = "<< diff_n.count()<<std::endl;

	imshow("CLAHE result", result);
	//imshow("Original", img);
	waitKey();

	return 0;
}