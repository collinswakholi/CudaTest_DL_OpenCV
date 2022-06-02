// includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
# include<opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
# include <iostream>
# include <fstream>
# include <chrono>
#include <Windows.h>

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"


using namespace cv;
using namespace std;

// Global declarations
const vector<Scalar> colors = { Scalar(152,63,144),Scalar(0,255,0), Scalar(29,147,247), Scalar(0, 0, 255) };
const float Im_W = 512;
const float Im_H = 512;
const float Score_threshold = 0.48;
const float NMS_threshold = 0.35;
const float Conf_threshold = 0.45;

struct Detection {
	int class_id;
	float confidence;
	Rect box;
	Point cent;
};

// Functions

vector<string> load_class_list() { // load class names
	vector<string> class_list;
	ifstream ifs("../img_labels.txt");
	string line;
	while (getline(ifs, line)) {
		class_list.push_back(line);
	}
	return class_list;
}

void load_net(dnn::Net& Net, bool is_cuda) { // load yolo network
	auto result = dnn::readNet("../model.onnx");
	if (is_cuda) {
		cout << "Attempting to use CUDA\n";
		result.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
		result.setPreferableTarget(dnn::DNN_TARGET_CUDA_FP16);
	}
	else {
		cout << "Running on CPU\n";
		result.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
		result.setPreferableTarget(dnn::DNN_TARGET_CPU);
	}
	Net = result;
}

Mat format_yolov5(const Mat& source) { // Format image to yolov5 input format
	int col = source.cols;
	int row = source.rows;
	int _max = MAX(col, row);
	Mat Res = Mat::zeros(_max, _max, CV_8UC3);
	source.copyTo(Res(Rect(0, 0, col, row)));

	return Res;
}

void detect(Mat& Image, dnn::Net& net, vector<Detection>& output, const vector<string>& className) { // detection function
	Mat blob; // reserve meomory if not reserved
	vector<Mat> outputs;
	Mat input_img = format_yolov5(Image);

	dnn::blobFromImage(input_img, blob, 1. / 255., Size(Im_W, Im_H), Scalar(), true, false);
	net.setInput(blob);
	net.forward(outputs, net.getUnconnectedOutLayersNames());

	float x_factor = input_img.cols / Im_W;
	float y_factor = input_img.rows / Im_H;

	float* data = (float*)outputs[0].data;

	const int dimension = 8;
	const int rows = 16128; //////////////////////////////////////////////////////////////////////////////////////////////

	vector<int> class_ids;
	vector<float> confidences;
	vector<Rect> boxes;
	Point class_id;
	double max_class_score;

	// reserve memory
	boxes.reserve(300);
	confidences.reserve(300);
	class_ids.reserve(300);


	#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		float confidence = data[4];
		switch (confidence >= Conf_threshold) {
		case true:
			{
				float* classes_scores = data + 5; 

				Mat scores(1, className.size(), CV_32FC1, classes_scores);

				minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

				switch((max_class_score > Score_threshold) && (data[2] < 70)) {
					case true:
					{
						confidences.push_back(confidence);
						class_ids.push_back(class_id.x);

						boxes.push_back(Rect(int((data[0] - 0.5 * data[2]) * x_factor),
							int((data[1] - 0.5 * data[3]) * y_factor),
							int(data[2] * x_factor), int(data[3] * y_factor)));
					}
				}
			}
		}
		data += 8;
	}
	vector<int> nms_result;
	dnn::NMSBoxes(boxes, confidences, Score_threshold, NMS_threshold, nms_result);
	output.clear();
	Detection result;
	for (int i = 0; i < nms_result.size(); i++) { // put in GPU
		int idx = nms_result[i];
		
		result.class_id = class_ids[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		result.cent = Point((boxes[idx].x + 0.5 * boxes[idx].width), (boxes[idx].y + 0.5 * boxes[idx].height));
		output.push_back(result);
	}
}


int main(int argc, char** argv) {

	vector<string> class_list = load_class_list();

	vector<String> fn;
	glob("../Images/*.png", fn, false);

	Mat frame;
	vector<Detection> output;

	cuda::printShortCudaDeviceInfo(cuda::getDevice());
	int cuda_devices_number = cuda::getCudaEnabledDeviceCount();
	cuda::DeviceInfo _deviceInfo;
	bool _isd_evice_compatible = _deviceInfo.isCompatible();

	dnn::Net net;
	load_net(net, _isd_evice_compatible);

	auto start = chrono::high_resolution_clock::now();
	int frame_count = 0;
	float fps = -1;
	int total_frames = 0;
	bool runn = true;

	while (runn) {
		switch (total_frames >= fn.size() - 1) {
			case true: {
				runn = false;
				cout << "Finished" << endl;
			}
		}

		frame = imread(fn[total_frames]);

		detect(frame, net, output, class_list);

		frame_count++;
		total_frames++;

		int detections = output.size();
		cout << "Number of Detections per image: " << detections << endl;
		String my_label;
		for (int i = 0; i < detections; ++i) {
			auto detection = output[i];
			auto box = detection.box;
			auto classId = detection.class_id;
			const auto color = colors[classId % colors.size()];
			rectangle(frame, box, color, 2);

			rectangle(frame, Point(box.x, box.y - 20), Point(box.x + box.width, box.y), color, FILLED);
			my_label =( class_list[classId]+": 0."+ to_string(int(detection.confidence*100))).c_str();
			putText(frame, my_label, Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			//cout << "centroid = " << detection.cent.x << "," << detection.cent.y << endl;
		}

		
		auto end = chrono::high_resolution_clock::now();
		fps = frame_count * 1000.0 / chrono::duration_cast<chrono::milliseconds>(end - start).count();

		frame_count = 0;
		start = chrono::high_resolution_clock::now();
		

		if (fps > 0)
		{
			ostringstream fps_label;
			fps_label << fixed << setprecision(2);
			fps_label << "FPS: " << fps;
			string fps_label_str = fps_label.str();

			putText(frame, fps_label_str.c_str(), Point(10, 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
		}
		//Mat dst;
		//vconcat(frame, frame, dst);
		//Mat fff = dst(Range(0, 99), Range(0, 511));
		imshow("Output", frame);
		waitKey(0);

		//Sleep(200);

		cout << "FPS: " << fps << "\n";
		cout << "Total frame handled: " << total_frames << "\n";
		
	}
	cout << "Total frame handled: " << total_frames << "\n";
	
	return 0;
}

