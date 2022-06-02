#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "opencv2/imgproc.hpp"

# include <chrono>
#include <ratio>
# include <iostream>
# include <fstream>

using namespace cv;
using namespace std;

Mat detectSeed(Mat Img) {
	Mat rgb[3];
	split(Img, rgb);
	Mat b_band = rgb[0];
	Mat res = b_band < 220;
	return res;
}

void countSeed(Mat Img, int& count) {
	Mat output;
	count = connectedComponents(Img, output, 8, CV_16U);
	
	count = count - 1;
	//return 0;
}

int main() {

	Mat img = imread("image.png");

	if (img.empty()) {
		cout << "Could Not find image" << endl;
		cin.get();
		return -1;
	}
	Mat img_r;
	resize(img, img_r, Size(), 0.5, 0.5, INTER_LINEAR);
	int count = 0;
	

	auto start = chrono::high_resolution_clock::now(); // tic
	Mat img2 = detectSeed(img_r);

	countSeed(img2, count);
	auto stop = chrono::high_resolution_clock::now(); // tic


	auto diff = chrono::duration_cast<chrono::microseconds>(stop - start);

	cout << "Number of seeds = " << count << endl;
	cout << " detection time = " << diff.count() << endl;

	imshow("Original Image", img_r);
	imshow("Detected seeds", img2);
	waitKey(0);
}