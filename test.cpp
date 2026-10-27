#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
#include "speedUp.h"
#include <xmmintrin.h>
#include <pmmintrin.h>

using namespace std;
using namespace cv;

static std::string prefix = "E:/cwn/ProjectTest/shapeMatch/shape_based_matching-master-TEST/shape_based_matching-master/test/";

class Timer
{
public:
	Timer() : beg_(clock_::now()) {}
	void reset() { beg_ = clock_::now(); }
	double elapsed() const {
		return std::chrono::duration_cast<second_>
			(clock_::now() - beg_).count();
	}
	void out(std::string message = "") {
		double t = elapsed();
		std::cout << message << "\nelasped time:" << t << "s" << std::endl;
		reset();
	}
private:
	typedef std::chrono::high_resolution_clock clock_;
	typedef std::chrono::duration<double, std::ratio<1> > second_;
	std::chrono::time_point<clock_> beg_;
};
// NMS, got from cv::dnn so we don't need opencv contrib
// just collapse it
namespace  cv_dnn {
namespace
{

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

} // namespace

inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

template <typename BoxType>
inline void NMSFast_(const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
{
    CV_Assert(bboxes.size() == scores.size());
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep)
            indices.push_back(idx);
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}


// copied from opencv 3.4, not exist in 3.0
template<typename _Tp> static inline
double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
    _Tp Aa = a.area();
    _Tp Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    double Aab = (a & b).area();
    // distance = 1 - jaccard_index
    return 1.0 - Aab / (Aa + Ab - Aab);
}

template <typename T>
static inline float rectOverlap(const T& a, const T& b)
{
    return 1.f - static_cast<float>(jaccardDistance__(a, b));
}

void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices, const float eta=1, const int top_k=0)
{
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
}

}

void scale_test(string mode = "test"){
    int num_feature = 150;

    // feature numbers(how many ori in one templates?)
    // two pyramids, lower pyramid(more pixels) in stride 4, lower in stride 8
    line2Dup::Detector detector(num_feature, {4, 8});

//    mode = "test";
    if(mode == "train"){
        //Mat img = cv::imread(prefix+"case0/templ/circle.png");
		Mat img = imread(prefix + "case1/train.png");
		Rect roi(2505, 1428, 520, 520);
		img = img(roi).clone();
        assert(!img.empty() && "check your img path");
        shape_based_matching::shapeInfo_producer shapes(img);

        shapes.scale_range = {0.1f, 1};
        shapes.scale_step = 0.01f;
        shapes.angle_range = {-2, 2};
        shapes.angle_step = 1;
        shapes.produce_infos();

        std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
        string class_id = "circle";
        for(auto& info: shapes.infos){

            // template img, id, mask,
            //feature numbers(missing it means using the detector initial num)
            int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info),
                                                int(num_feature*info.scale));
            std::cout << "templ_id: " << templ_id << std::endl;

            // may fail when asking for too many feature_nums for small training img
            if(templ_id != -1){  // only record info when we successfully add template
                infos_have_templ.push_back(info);
            }
        }

        // save templates
        detector.writeClasses(prefix+"case0/%s_templ.yaml");

        // save infos,
        // in this simple case infos are not used
        shapes.save_infos(infos_have_templ, prefix + "case0/circle_info.yaml");
        std::cout << "train end" << std::endl << std::endl;

    }else if(mode=="test"){
        std::vector<std::string> ids;

        // read templates
        ids.push_back("circle");
        detector.readClasses(ids, prefix+"case0/%s_templ.yaml");

        //Mat test_img = imread(prefix+"case0/1.jpg");
        Mat test_img = imread(prefix + "case1/abc.png", IMREAD_GRAYSCALE);
        assert(!test_img.empty() && "check your img path");

        // make the img having 32*n width & height
        // at least 16*n here for two pyrimads with strides 4 8
        int stride = 32;
        int n = test_img.rows/stride;
        int m = test_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);
        Mat img = test_img(roi).clone();
        assert(img.isContinuous());

        Timer timer;
        // match, img, min socre, ids
        auto matches = detector.match(img, 90, ids);
        // one output match:
        // x: top left x
        // y: top left y
        // template_id: used to find templates
        // similarity: scores, 100 is best
        timer.out();

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 5;
        if(top5 > matches.size()) 
            top5 = matches.size();
        for(size_t i = 0; i < top5; i++)
        {
            auto match = matches[i];
            auto templ = detector.getTemplates("circle",
                                               match.template_id);
            // template:
            // nums: num_pyramids * num_modality (modality, depth or RGB, always 1 here)
            // template[0]: lowest pyrimad(more pixels)
            // template[0].width: actual width of the matched template
            // template[0].tl_x / tl_y: topleft corner when cropping templ during training
            // In this case, we can regard width/2 = radius
            int x =  templ[0].width/2 + match.x;
            int y = templ[0].height/2 + match.y;
            int r = templ[0].width/2;
            Scalar color(255, rand()%255, rand()%255);

            cv::putText(img, to_string(int(round(match.similarity))),
                        Point(match.x+r-10, match.y-3), FONT_HERSHEY_PLAIN, 2, color);
            cv::circle(img, {x, y}, r, color, 2);
        }

		cv::namedWindow("img", WINDOW_NORMAL);
        imshow("img", img);
        waitKey(0);

        std::cout << "test end" << std::endl << std::endl;
    }
}

void angle_test(string mode = "test", bool use_rot = true){
    line2Dup::Detector detector(64, {4, 8});

    if(mode != "test"){
        Mat img = imread(prefix+"case1/train.png");
        assert(!img.empty() && "check your img path");

		//Rect roi(130, 110, 270, 270);
		Rect roi(2505, 1428, 520, 520);
        img = img(roi).clone();
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        // padding to avoid rotating out
        int padding = 100;
        cv::Mat padded_img = cv::Mat(img.rows + 2*padding, img.cols + 2*padding, img.type(), cv::Scalar::all(0));
        img.copyTo(padded_img(Rect(padding, padding, img.cols, img.rows)));

        cv::Mat padded_mask = cv::Mat(mask.rows + 2*padding, mask.cols + 2*padding, mask.type(), cv::Scalar::all(0));
        mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));

        shape_based_matching::shapeInfo_producer shapes(padded_img, padded_mask);
        shapes.angle_range = {-10, 10};
        // Coarse angle library (scheme A). Fine 0.1deg search runs at match time.
        shapes.angle_step = 1.0;

        shapes.scale_range = {1}; // support just one
        shapes.produce_infos();
        std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
        string class_id = "test";

        bool is_first = true;

        // for other scales you want to re-extract points: 
        // set shapes.scale_range then produce_infos; set is_first = false;

        int first_id = 0;
        float first_angle = 0;
        for(auto& info: shapes.infos)
        {
            Mat to_show = shapes.src_of(info);

            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            int templ_id;

            if(is_first)
            {
                templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
                first_id = templ_id;
                first_angle = info.angle;

                if(use_rot) is_first = false;
            }else
            {
                templ_id = detector.addTemplate_rotate(class_id, first_id,
                                                       info.angle-first_angle,
                                                {shapes.src.cols/2.0f, shapes.src.rows/2.0f});
            }

            auto templ = detector.getTemplates("test", templ_id);
            for(int i=0; i<templ[0].features.size(); i++)
            {
                auto feat = templ[0].features[i];
                cv::circle(to_show, {feat.x+templ[0].tl_x, feat.y+templ[0].tl_y}, 3, {0, 0, 255}, -1);
            }
            
            // will be faster if not showing this
            imshow("train", to_show);
            waitKey(1);

            std::cout << "templ_id: " << templ_id << std::endl;
            if(templ_id != -1){
                infos_have_templ.push_back(info);
            }
        }
        detector.writeClasses(prefix+"case1/%s_templ.yaml");
        shapes.save_infos(infos_have_templ, prefix + "case1/test_info.yaml");
        std::cout << "train end" << std::endl << std::endl;
    }else if(mode=="test"){
        std::vector<std::string> ids;
        ids.push_back("test");
        detector.readClasses(ids, prefix+"case1/%s_templ.yaml");

        // angle & scale are saved here, fetched by match id
        auto infos = shape_based_matching::shapeInfo_producer::load_infos(prefix + "case1/test_info.yaml");

        Mat test_img = imread(prefix+"case1/train.png", IMREAD_GRAYSCALE);
        assert(!test_img.empty() && "check your img path");

        //int padding = 250;
        //cv::Mat padded_img = cv::Mat(test_img.rows + 2*padding,
        //                             test_img.cols + 2*padding, test_img.type(), cv::Scalar::all(0));
        //test_img.copyTo(padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));

        //int stride = 16;
        //int n = padded_img.rows/stride;
        //int m = padded_img.cols/stride;
        //Rect roi(0, 0, stride*m , stride*n);
        //Mat img = padded_img(roi).clone();
        Mat img = test_img;
        assert(img.isContinuous());

//        cvtColor(img, img, CV_BGR2GRAY);

        std::cout << "test img size: " << img.rows * img.cols << " (" << img.cols << "x" << img.rows << ")" << std::endl << std::endl;

		// 获取当前时间的时间戳（自epoch以来的毫秒数）
        auto start = std::chrono::high_resolution_clock::now();

        Timer timer;
        auto matches = detector.match(img, 90, ids);
        timer.out("match end");

        // B: one match per spatial cluster
        matches = line2Dup::Detector::clusterMatchesByPosition(matches, 16);
        std::cout << "matches after cluster: " << matches.size() << std::endl;

        // A: local fine-angle refine around each coarse template (±1deg, 0.1deg step)
        std::vector<float> template_angles(infos.size());
        for (size_t i = 0; i < infos.size(); ++i)
            template_angles[i] = infos[i].angle;
        for (auto &m : matches)
        {
            if (m.template_id >= 0 && m.template_id < (int)infos.size())
                m.angle = infos[m.template_id].angle;
        }
        const float train_img_half_width = 520 / 2.0f + 100;
        const float train_img_half_height = 520 / 2.0f + 100;
        detector.refineMatchAngles(img, matches, template_angles,
                                   {train_img_half_width, train_img_half_height},
                                   1.0f, 0.1f);
        timer.out("angle refine");
		
        auto end = std::chrono::high_resolution_clock::now();
		// 以毫秒为单位计算时间差
		auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
		std::cout << "Time difference is " << duration_ms << " seconds." << std::endl;


        if(img.channels() == 1) cvtColor(img, img, CV_GRAY2BGR);

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 1;
        if(top5>matches.size()) top5=matches.size();
        for(size_t i=0; i<top5; i++){
            auto match = matches[i];
            auto templ = detector.getTemplates("test",
                                               match.template_id);

            // 270（new 520） is width of template image
            // 100 is padding when training
            // tl_x/y: template croping topleft corner when training

            float r_scaled = 520/2.0f*infos[match.template_id].scale;

            // center x,y of train_img in test img
            float x =  match.x - templ[0].tl_x + train_img_half_width;
            float y =  match.y - templ[0].tl_y + train_img_half_height;

            cv::Vec3b randColor;
            randColor[0] = rand()%155 + 100;
            randColor[1] = rand()%155 + 100;
            randColor[2] = rand()%155 + 100;
            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(img, {feat.x+match.x, feat.y+match.y}, 3, randColor, -1);
            }

            cv::putText(img, to_string(int(round(match.similarity))),
                        Point(match.x+r_scaled-10, match.y-3), FONT_HERSHEY_PLAIN, 2, randColor);

            cv::RotatedRect rotatedRectangle({x, y}, {2*r_scaled, 2*r_scaled}, -match.angle);

            cv::Point2f vertices[4];
            rotatedRectangle.points(vertices);
            for(int i=0; i<4; i++){
                int next = (i+1==4) ? 0 : (i+1);
                cv::line(img, vertices[i], vertices[next], randColor, 2);
            }

            std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
            std::cout << "coarse angle: " << infos[match.template_id].angle
                      << ", refined angle: " << match.angle << std::endl;
        }
        cv::namedWindow("img", WINDOW_NORMAL);
        imshow("img", img);
        waitKey(0);

        std::cout << "test end" << std::endl << std::endl;
    }
}

void noise_test(string mode = "test"){
    line2Dup::Detector detector(30, {4, 8});

    if(mode == "train"){
        Mat img = imread(prefix+"case2/train.png");
        assert(!img.empty() && "check your img path");
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        shape_based_matching::shapeInfo_producer shapes(img, mask);
        shapes.angle_range = {0, 360};
        shapes.angle_step = 1;
        shapes.produce_infos();
        std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
        string class_id = "test";
        for(auto& info: shapes.infos){
            imshow("train", shapes.src_of(info));
            waitKey(1);

            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
            std::cout << "templ_id: " << templ_id << std::endl;
            if(templ_id != -1){
                infos_have_templ.push_back(info);
            }
        }
        detector.writeClasses(prefix+"case2/%s_templ.yaml");
        shapes.save_infos(infos_have_templ, prefix + "case2/test_info.yaml");
        std::cout << "train end" << std::endl << std::endl;
    }else if(mode=="test"){
        std::vector<std::string> ids;
        ids.push_back("test");
        detector.readClasses(ids, prefix+"case2/%s_templ.yaml");

        Mat test_img = imread(prefix+"case2/test.png");
        assert(!test_img.empty() && "check your img path");

        // cvtColor(test_img, test_img, CV_BGR2GRAY);

        int stride = 16;
        int n = test_img.rows/stride;
        int m = test_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);

        test_img = test_img(roi).clone();

        Timer timer;
        auto matches = detector.match(test_img, 90, ids);
        timer.out();

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 500;
        if(top5>matches.size()) top5=matches.size();

        vector<Rect> boxes;
        vector<float> scores;
        vector<int> idxs;
        for(auto match: matches){
            Rect box;
            box.x = match.x;
            box.y = match.y;

            auto templ = detector.getTemplates("test",
                                               match.template_id);

            box.width = templ[0].width;
            box.height = templ[0].height;
            boxes.push_back(box);
            scores.push_back(match.similarity);
        }
        cv_dnn::NMSBoxes(boxes, scores, 0, 0.5f, idxs);

        for(auto idx: idxs){
            auto match = matches[idx];
            auto templ = detector.getTemplates("test",
                                               match.template_id);

            int x =  templ[0].width + match.x;
            int y = templ[0].height + match.y;
            int r = templ[0].width/2;
            cv::Vec3b randColor;
            randColor[0] = rand()%155 + 100;
            randColor[1] = rand()%155 + 100;
            randColor[2] = rand()%155 + 100;

            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(test_img, {feat.x+match.x, feat.y+match.y}, 2, randColor, -1);
            }

            cv::putText(test_img, to_string(int(round(match.similarity))),
                        Point(match.x+r-10, match.y-3), FONT_HERSHEY_PLAIN, 2, randColor);
            cv::rectangle(test_img, {match.x, match.y}, {x, y}, randColor, 2);

            std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
        }

        imshow("img", test_img);
        waitKey(0);

        std::cout << "test end" << std::endl << std::endl;
    }
}

void MIPP_test(){
    std::cout << "MIPP tests" << std::endl;
    std::cout << "----------" << std::endl << std::endl;

    std::cout << "Instr. type:       " << mipp::InstructionType                  << std::endl;
    std::cout << "Instr. full type:  " << mipp::InstructionFullType              << std::endl;
    std::cout << "Instr. version:    " << mipp::InstructionVersion               << std::endl;
    std::cout << "Instr. size:       " << mipp::RegisterSizeBit       << " bits" << std::endl;
    std::cout << "Instr. lanes:      " << mipp::Lanes                            << std::endl;
    std::cout << "64-bit support:    " << (mipp::Support64Bit    ? "yes" : "no") << std::endl;
    std::cout << "Byte/word support: " << (mipp::SupportByteWord ? "yes" : "no") << std::endl;

#ifndef has_max_int8_t
        std::cout << "in this SIMD, int8 max is not inplemented by MIPP" << std::endl;
#endif

#ifndef has_shuff_int8_t
        std::cout << "in this SIMD, int8 shuff is not inplemented by MIPP" << std::endl;
#endif

    std::cout << "----------" << std::endl << std::endl;
}
void print_ps(__m128 v, const char* name) {
	float arr[4];
	_mm_storeu_ps(arr, v);
	printf("%s: %6.1f %6.1f %6.1f %6.1f\n", name, arr[3], arr[2], arr[1], arr[0]);
}


int main()
{
    //scale_test("test");
    angle_test("test", true); // test or train
	//noise_test("test");
	return 0;


	//// 准备数据
	//__m128 a = _mm_set_ps(8.0f, 7.0f, 6.0f, 5.0f);  // [8,7,6,5]
	//__m128 b = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);  // [4,3,2,1]

	//print_ps(a, "a");
	//print_ps(b, "b");

	//// 示例1: 选择 a[0], a[1], b[0], b[1]
	//// imm8 = 0b 01 00 11 10 = 0x1E
 //   //  00 01 11 10
	//__m128 r1 = _mm_shuffle_ps(a, b, 0x1E);
	//print_ps(r1, "a[1],a[0],b[1],b[0]");

	//// 示例2: 广播 a[0]
	//// imm8 = 0b 00 00 00 00 = 0x00
	//__m128 r2 = _mm_shuffle_ps(a, a, 0x00);
	//print_ps(r2, "广播 a[0]");

	//// 示例3: 交换 a[0] 和 a[1]
	//// imm8 = 0b 00 01 00 01 = 0x11
	//__m128 r3 = _mm_shuffle_ps(a, a, 0x11);
	//print_ps(r3, "交换 a[0],a[1]");

	//return 0;

	//float f = 1.0f;
	//for (int i = 0; i < 10000000; i++) {
	//	f *= 0.999998f; 
	//}
	//std::cout << "Default: " << f << std::endl;

	//_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	//_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
 //   _mm_setcsr(_mm_getcsr() | 0x8040);
	//unsigned int csr = _mm_getcsr();
	//printf("MXCSR: 0x%X\n", csr);

	//f = 1.0f;
	//for (int i = 0; i < 10000000; i++) {
	//	f *= 0.999998f;
	//}
	//std::cout << "With FTZ/DAZ: " << f << std::endl;

	//return 0;


	//Mat src = imread("C:\\Users\\byd\\Desktop\\ipad\\33.png");

 //   //cv::Mat expand_Img;
 //   //cv::copyMakeBorder(src, expand_Img, 250, 250, 250, 250, cv::BORDER_REPLICATE);

	//static const int KERNEL_SIZE = 7;
 //   Mat smoothed;
	//GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);
 //   GaussianBlur(src, smoothed, Size(0, 0), KERNEL_SIZE, 0, BORDER_REPLICATE);

	//int Height = src.rows;
	//int Width = src.cols;
	//unsigned char* Src = src.data;
	//unsigned char* Dest = new unsigned char[Height * Width * 3];
	//int Stride = Width * 3;
	//int Radius = 1;
	//int64 st = cvGetTickCount();
	////for (int i = 0; i < 20; i++) {
	////	GaussBlur_SSE(Src, Dest, Width, Height, Stride, Radius);
	////}
	//double duration = (cv::getTickCount() - st) / cv::getTickFrequency() * 50;
	//printf("%.5f\n", duration);
 //   //GaussBlur(Src, Dest, Width, Height, Stride, Radius);
	//GaussBlur_SSE(Src, Dest, Width, Height, Stride, 1);
 //   GaussBlur_SSE(Src, Dest, Width, Height, Stride, 3);
 //   GaussBlur_SSE(Src, Dest, Width, Height, Stride, 5);
 //   GaussBlur_SSE(Src, Dest, Width, Height, Stride, 7);
	//Mat dst(Height, Width, CV_8UC3, Dest);
	//imshow("origin", src);
	//imshow("result", dst);
	//imwrite("C:\\Users\\byd\\Desktop\\ipad\\res.jpg", dst);
	//waitKey(0);

 //   return 1;




	//// 原始数据
	//float GXYL[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	//float GXYH[8] = { 10, 20, 30, 40, 50, 60, 70, 80 };

	//// 加载到寄存器
	//__m256 low = _mm256_loadu_ps(GXYL);   // {1,2,3,4,5,6,7,8}
	//__m256 high = _mm256_loadu_ps(GXYH);   // {10,20,30,40,50,60,70,80}

	//// 步骤1: 提取低4个元素和高4个元素
	//// low_low   = {1,2,3,4} (低位)
	//// low_high  = {5,6,7,8} (高位)
	//// high_low  = {10,20,30,40}
	//// high_high = {50,60,70,80}
	//__m128 low_low_128 = _mm256_castps256_ps128(low);      // 低128位
	//__m128 low_high_128 = _mm256_extractf128_ps(low, 1);    // 高128位
	//__m128 high_low_128 = _mm256_castps256_ps128(high);
	//__m128 high_high_128 = _mm256_extractf128_ps(high, 1);

	//// 步骤2: 重新组合成两个256位向量
	//// part1 = {1,2,3,4, 10,20,30,40}
	//__m256 part1 = _mm256_set_m128(high_low_128, low_low_128);

	//// part2 = {5,6,7,8, 50,60,70,80}
	//__m256 part2 = _mm256_set_m128(high_high_128, low_high_128);

	//// 结果输出验证
	//float result1[8], result2[8];
	//_mm256_storeu_ps(result1, part1);
	//_mm256_storeu_ps(result2, part2);

	//printf("part1: ");
	//for (int i = 0; i < 8; i++) printf("%.0f ", result1[i]);
	//printf("\npart2: ");
	//for (int i = 0; i < 8; i++) printf("%.0f ", result2[i]);
	//printf("\n");

	//return 0;

	//int16_t a_data[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	//int16_t b_data[8] = { 10, 20, 30, 40, 50, 60, 70, 80 };

	//// 加载到 256 位寄存器
	//__m256i a = _mm256_loadu_si256((__m256i*)a_data);
	//__m256i b = _mm256_loadu_si256((__m256i*)b_data);

	//// 乘加运算
	//__m256i result = _mm256_madd_epi16(a, b);

	//// 提取结果
	//int32_t output[4];
	//_mm256_storeu_si256((__m256i*)output, result);

	//std::cout << "乘加结果:" << std::endl;
	//for (int i = 0; i < 4; i++) {
	//	std::cout << "dst[" << i << "] = " << output[i] << std::endl;
	//}

	//// 手动验证
	//std::cout << "\n手动验证:" << std::endl;
	//std::cout << "1*10 + 2*20 = " << (1 * 10 + 2 * 20) << std::endl;
	//std::cout << "3*30 + 4*40 = " << (3 * 30 + 4 * 40) << std::endl;
	//std::cout << "5*50 + 6*60 = " << (5 * 50 + 6 * 60) << std::endl;
	//std::cout << "7*70 + 8*80 = " << (7 * 70 + 8 * 80) << std::endl;

	//return 0;

	//int16_t a_data[8] = { 100, 20, 500, 700, 1, 2,3, 4 };
	//__m128i a128 = _mm_loadu_si128((__m128i*)a_data);
	////__m256i a256 = _mm256_cvtepi32_epi64(a128);  // 扩展到 256 位（4→2 个元素）
	//// 或者
	//__m256i a256 = _mm256_set_m128i(_mm_setzero_si128(), a128);  // 高位清0
	//// 提取结果
	//uint16_t output[16];
	//_mm256_storeu_si256((__m256i*)output, a256);
	//std::cout << "饱和打包结果（uint16_t）:" << std::endl;
	//for (int i = 0; i < 16; i++) {
	//	std::cout << output[i] << " ";
	//}
	//std::cout << std::endl;
 //   return;

 //   // 准备测试数据 (8个 int32_t)
	//int32_t a_data[8] = { -100, 0, 50000, 70000 };    // 70000 > 65535
	//int32_t b_data[4] = { 100, 200, 300, 65536 };      // 65536 > 65535

	//// 加载到 256 位寄存器
	//__m256i a = _mm256_loadu_si256((__m256i*)a_data);
	//__m256i b = _mm256_loadu_si256((__m256i*)b_data);

	//// 饱和打包为 uint16_t
	//__m256i result = _mm256_packus_epi32(a, b);

 //   __m256i Zero = _mm256_setzero_si256();
 //   __m256i Result1 = _mm256_packus_epi16(result, Zero);

	//// 提取结果
	//uint16_t output[8];
	//_mm256_storeu_si256((__m256i*)output, Result1);

	//// 打印结果
	//std::cout << "饱和打包结果（uint16_t）:" << std::endl;
	//for (int i = 0; i < 16; i++) {
	//	std::cout << output[i] << " ";
	//	if (i == 3) std::cout << "| ";
	//}
	//std::cout << std::endl;

	//return 0;


	//// 准备 16 个 16 位整数
	//int16_t data[16] = {
	//	1, 2, 3, 4, 5, 6, 7, 8,    // 低 128 位
	//	9, 10, 11, 12, 13, 14, 15, 16  // 高 128 位
	//};

	//// 加载到 256 位寄存器
	//__m256i vec256 = _mm256_loadu_si256((__m256i*)data);

	//// 转换为 128 位（丢弃高 128 位）
	//__m128i vec128 = _mm256_castsi256_si128(vec256);

	//// 提取结果
	//int16_t result[8];
	//_mm_storeu_si128((__m128i*)result, vec128);

	//// 输出：只保留了低 8 个元素
	//for (int i = 0; i < 8; i++) {
	//	std::cout << result[i] << " ";
	//}
	//// 输出: 1 2 3 4 5 6 7 8
 //   // 
 //   return 1;



	//// 准备测试数据 (32个 int16_t)
	//int16_t a_data[16] = {
	//	-10, 0, 50, 100, 200, 255, 300, 500,    // 超出范围的值
	//	1000, -50, 128, 256, 30, 40, 50, 60
	//};

	//int16_t b_data[16] = {
	//	10, 20, 30, 40, 50, 60, 70, 80,
	//	90, 100, 110, 120, 130, 140, 150, 160
	//};

	//// 加载到 256 位寄存器
	//__m256i a = _mm256_loadu_si256((__m256i*)a_data);
	//__m256i b = _mm256_loadu_si256((__m256i*)b_data);

	//// 饱和打包为 uint8_t
	//__m256i result = _mm256_packus_epi16(a, b);

	//// 提取结果
	//uint8_t output[32];
	//_mm256_storeu_si256((__m256i*)output, result);

	//// 打印结果
	//std::cout << "饱和打包结果（uint8_t）:" << std::endl;
	//for (int i = 0; i < 32; i++) {
	//	std::cout << (int)output[i] << " ";
	//	if ((i + 1) % 16 == 0) std::cout << std::endl;
	//}

 //   __m128i Ans1 = _mm256_castsi256_si128(result);
	//// 提取结果
	//uint8_t output_Test[16];
	//_mm256_storeu_si256((__m256i*)output_Test, result);
	//// 打印结果
	//std::cout << "裁剪打包结果（uint8_t）:" << std::endl;
	//for (int i = 0; i < 16; i++) {
	//	std::cout << (int)output_Test[i] << " ";
	//	if ((i + 1) % 16 == 0) std::cout << std::endl;
	//}

	//return 0;

	//int f = mipp::N<float>() * 2;
	//std::vector<float> myVectorF(f);
	//myVectorF[0] = 555.44;
	//myVectorF[7] = 555.55;
	////myVectorF[8] = 555.66;
	////myVectorF[12] = 555.77;
	//mipp::Reg<float> r1 = &myVectorF[0 * mipp::N<float>()];
	//mipp::Reg<float> r2 = &myVectorF[1 * mipp::N<float>()];
	//mipp::Reg<float> rout0;
	//rout0 = r1 + r2;//8个float类型的数据正常需要8个时钟周期，这里向量相加只需要1个时钟周期，
	//rout0.store(&myVectorF[(1) * mipp::N<float>()]);//将值保存到myVectorF的8-16位置上

	//int db = mipp::N<double>() * 2;
	//std::vector<double> myVectorD(db);
	//myVectorD[0] = 0.33;
	//myVectorD[3] = 1.44;
	//myVectorD[4] = 2.55;
	//mipp::Reg<double> r3 = &myVectorD[0 * mipp::N<double>()];
	//mipp::Reg<double> r4 = &myVectorD[1 * mipp::N<double>()];
	//mipp::Reg<double> rout1;
	//rout1 = r3 + r4;
	//rout1.store(&myVectorD[(0) * mipp::N<double>()]);

    //return 0;
}