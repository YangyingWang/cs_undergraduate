#include "pch.h"
#include "ImgGray.h"
#include <omp.h>

Mat ImgGray::RGB_2_Gray(Mat src) // RGB转换成GRAY
{   
    Mat result;
    cvtColor(src, result, COLOR_BGR2GRAY);
    return result;
}

void ImgGray::Show_Histogram(Mat src)
{
	Mat image_gray, hist;   //定义输入图像，灰度图像, 直方图
	cvtColor(src, image_gray, COLOR_BGR2GRAY);  //灰度化

	//获取图像直方图
	int histsize = 256;
	float ranges[] = { 0,256 };
	const float* histRanges = { ranges };
	calcHist(&image_gray, 1, 0, Mat(), hist, 1, &histsize, &histRanges, true, false);

	//创建直方图显示图像
	int hist_h = 300;//直方图的图像的高
	int hist_w = 512; //直方图的图像的宽
	int bin_w = hist_w / histsize;//直方图的等级
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));//绘制直方图显示的图像

	//绘制并显示直方图
	normalize(hist, hist, 0, hist_h, NORM_MINMAX, -1, Mat());//归一化直方图
	for (int i = 1; i < histsize; i++)
	{
		line(histImage, Point((i - 1) * bin_w, hist_h - cvRound(hist.at<float>(i - 1))),
			Point((i)*bin_w, hist_h - cvRound(hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
	}
	imshow("histImage", histImage);

	waitKey(0);  //暂停，保持图像显示，等待按键结束
}

Mat ImgGray::EqualizeHist(Mat src)
{
	Mat image_gray, image_enhanced;   //定义输入图像，灰度图像, 直方图
	cvtColor(src, image_gray, COLOR_BGR2GRAY);  //灰度化

	equalizeHist(image_gray, image_enhanced);//直方图均衡化
	return image_enhanced;
}

Mat ImgGray::GrayscaleLinearTransform(Mat inputImage, double alpha, double beta)
{
	omp_set_num_threads(2);

	Mat grayImage;
	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
	Mat transformedImage = Mat::zeros(grayImage.size(), CV_8U);

	// 线性变换：new_pixel_value = alpha * original_pixel_value + beta
	// y = kx + b
#pragma omp parallel for
	for (int y = 0; y < grayImage.rows; ++y) {
		for (int x = 0; x < grayImage.cols; ++x) {
			int originalPixelValue = static_cast<int>(grayImage.at<uchar>(y, x));
			int newPixelValue = saturate_cast<uchar>(alpha * originalPixelValue + beta);
			transformedImage.at<uchar>(y, x) = newPixelValue;
		}
	}
	return transformedImage;
}

Mat ImgGray::GrayscaleLogTransform(Mat inputImage, double c)
{
	omp_set_num_threads(2);
	Mat grayImage;
	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

	// 对数变换：new_pixel_value = c * log(1 + original_pixel_value)
	Mat transformedImage = Mat::zeros(grayImage.size(), CV_8U);
#pragma omp parallel for
	for (int y = 0; y < grayImage.rows; ++y) {
		for (int x = 0; x < grayImage.cols; ++x) {
			int originalPixelValue = static_cast<int>(grayImage.at<uchar>(y, x));
			int newPixelValue = saturate_cast<uchar>(c * std::log(1 + originalPixelValue));
			transformedImage.at<uchar>(y, x) = newPixelValue;
		}
	}
	return transformedImage;
}

Mat ImgGray::GrayscaleGammaTransform(Mat inputImage, double gamma)
{
	omp_set_num_threads(2);
	Mat grayImage;
	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

	// 伽马变换：new_pixel_value = 255 * (original_pixel_value / 255) ^ gamma
	Mat transformedImage = Mat::zeros(grayImage.size(), CV_8U);

#pragma omp parallel for
	for (int y = 0; y < grayImage.rows; ++y) {
		for (int x = 0; x < grayImage.cols; ++x) {
			int originalPixelValue = static_cast<int>(grayImage.at<uchar>(y, x));
			int newPixelValue = saturate_cast<uchar>(255 * std::pow(originalPixelValue / 255.0, gamma));
			transformedImage.at<uchar>(y, x) = newPixelValue;
		}
	}
	return transformedImage;
}

Mat ImgGray::ImageErosion(Mat inputImage, int erosionSize)
{
	Mat grayImage;
	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

	// 定义腐蚀核（结构元素）
	Mat element = cv::getStructuringElement(MORPH_RECT, Size(2 * erosionSize + 1, 2 * erosionSize + 1),
		Point(erosionSize, erosionSize));

	// 进行腐蚀操作
	Mat erodedImage;
	erode(grayImage, erodedImage, element);
	return erodedImage;
}

Mat ImgGray::ImageDilation(Mat inputImage, int dilationSize)
{
	Mat grayImage;
	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

	// 定义膨胀核（结构元素）
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * dilationSize + 1, 2 * dilationSize + 1),
		Point(dilationSize, dilationSize));

	// 进行膨胀操作
	Mat dilatedImage;
	dilate(grayImage, dilatedImage, element);
	return dilatedImage;
}

Mat ImgGray::imageThresholdSegmentation(Mat inputImage, int threshold)
{
	Mat grayImage;
	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

	// 进行固定阈值分割
	cv::Mat segmentedImage;
	cv::threshold(grayImage, segmentedImage, threshold, 255, cv::THRESH_BINARY);
	return segmentedImage;
}

Mat ImgGray::adaptiveThresholdSegmentation(Mat inputImage, int blockSize, int subtractValue)
{
	Mat grayImage;
	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

	// 进行自适应阈值分割
	cv::Mat segmentedAdaptive;
	cv::adaptiveThreshold(grayImage, segmentedAdaptive, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
		cv::THRESH_BINARY, blockSize, subtractValue);
	return segmentedAdaptive;
}

Mat ImgGray::regionGrowingSegmentation(Mat inputImage, int seedX, int seedY, int threshold)
{
	Mat grayImage;
	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

	// 初始化分割图像
	Mat segmentedImage = cv::Mat::zeros(grayImage.size(), CV_8U);

	// 获取种子点的初始灰度值
	int seedValue = static_cast<int>(grayImage.at<uchar>(seedY, seedX));

	// 创建队列用于保存待处理的像素坐标
	std::queue<std::pair<int, int>> pixelQueue;
	pixelQueue.push(std::make_pair(seedX, seedY));

	// 开始区域生长
	while (!pixelQueue.empty()) {
		int x = pixelQueue.front().first;
		int y = pixelQueue.front().second;
		pixelQueue.pop();

		// 检查当前像素是否已被处理过
		if (segmentedImage.at<uchar>(y, x) == 0) {
			// 计算当前像素与种子点的灰度差
			int pixelValue = static_cast<int>(grayImage.at<uchar>(y, x));
			int diff = std::abs(seedValue - pixelValue);

			// 如果灰度差小于阈值，则加入分割区域
			if (diff <= threshold) {
				segmentedImage.at<uchar>(y, x) = 255;

				// 将当前像素的邻域加入队列
				for (int dy = -1; dy <= 1; ++dy) {
					for (int dx = -1; dx <= 1; ++dx) {
						int newX = x + dx;
						int newY = y + dy;
						if (newX >= 0 && newX < grayImage.cols && newY >= 0 && newY < grayImage.rows) {
							pixelQueue.push(std::make_pair(newX, newY));
						}
					}
				}
			}
		}
	}
	return segmentedImage;
}

Mat ImgGray::sobelEdgeDetection(Mat inputImage, int kernelSize)
{
	Mat grayImage;
	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

	// 进行Sobel边缘检测
	cv::Mat gradientX, gradientY;
	cv::Sobel(grayImage, gradientX, CV_16S, 1, 0, kernelSize);
	cv::Sobel(grayImage, gradientY, CV_16S, 0, 1, kernelSize);

	// 计算梯度幅值
	cv::Mat absGradientX, absGradientY;
	cv::convertScaleAbs(gradientX, absGradientX);
	cv::convertScaleAbs(gradientY, absGradientY);

	// 合并X和Y梯度幅值
	cv::Mat gradientImage;
	cv::addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0, gradientImage);

	return gradientImage;
}

Mat ImgGray::cannyEdgeDetection(Mat inputImage, double threshold1, double threshold2)
{
	Mat grayImage;
	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

	// 进行Canny边缘检测
	cv::Mat edgeImage;
	cv::Canny(grayImage, edgeImage, threshold1, threshold2);

	return edgeImage;
}

Mat ImgGray::meanFilter(Mat inputImage, int kernelSize)
{
	cv::Mat filteredImage;
	// 使用均值滤波
	cv::blur(inputImage, filteredImage, cv::Size(kernelSize, kernelSize));
	return filteredImage;
}

Mat ImgGray::medianFilter(Mat inputImage, int kernelSize)
{
	cv::Mat filteredImage;
	// 使用中值滤波
	cv::medianBlur(inputImage, filteredImage, kernelSize);
	return filteredImage;
}

Mat ImgGray::gaussianFilter(Mat inputImage, int kernelSize, double sigma)
{
	cv::Mat filteredImage;
	// 使用高斯滤波
	cv::GaussianBlur(inputImage, filteredImage, cv::Size(kernelSize, kernelSize), sigma);
	return filteredImage;
}

Mat ImgGray::detectFaces(Mat inputImage)
{
	//1.加载人脸检测器 
	CascadeClassifier cascade;
	const string path = "D:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml";
	if (!cascade.load(path))
	{
		AfxMessageBox(TEXT("cascade load failed!"));
	}

	//2.人脸检测
	vector<Rect> faces(0);
	cascade.detectMultiScale(inputImage, faces, 1.1, 2, 0, Size(30, 30));

	//3.显示人脸矩形框 
	CString str;
	if (faces.size() > 0)
	{
		str.Format(TEXT("检测到的人脸数量为 : %d"), faces.size());
		for (size_t i = 0; i < faces.size(); i++)
			rectangle(inputImage, faces[i], Scalar(150, 0, 0), 3, 8, 0);
	}
	else str.Format(TEXT("未检测到人脸"));
	AfxMessageBox(str);

	return inputImage;
}
