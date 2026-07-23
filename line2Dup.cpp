#include "line2Dup.h"
#include <iostream>
#include <future>
#include "speedUp.h"
#include <pmmintrin.h>
#include <chrono>

using namespace std;
using namespace cv;

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s\n" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

namespace line2Dup
{
/**
 * \brief Get the label [0,8) of the single bit set in quantized.
 */
static inline int getLabel(int quantized)
{
    switch (quantized)
    {
    case 1:
        return 0;
    case 2:
        return 1;
    case 4:
        return 2;
    case 8:
        return 3;
    case 16:
        return 4;
    case 32:
        return 5;
    case 64:
        return 6;
    case 128:
        return 7;
    default:
        CV_Error(Error::StsBadArg, "Invalid value of quantized parameter");
        return -1; //avoid warning
    }
}

void Feature::read(const FileNode &fn)
{
    FileNodeIterator fni = fn.begin();
    fni >> x >> y >> label;
}

void Feature::write(FileStorage &fs) const
{
    fs << "[:" << x << y << label << "]";
}

void Template::read(const FileNode &fn)
{
    width = fn["width"];
    height = fn["height"];
    tl_x = fn["tl_x"];
    tl_y = fn["tl_y"];
    pyramid_level = fn["pyramid_level"];

    FileNode features_fn = fn["features"];
    features.resize(features_fn.size());
    FileNodeIterator it = features_fn.begin(), it_end = features_fn.end();
    for (int i = 0; it != it_end; ++it, ++i)
    {
        features[i].read(*it);
    }
}

void Template::write(FileStorage &fs) const
{
    fs << "width" << width;
    fs << "height" << height;
    fs << "tl_x" << tl_x;
    fs << "tl_y" << tl_y;
    fs << "pyramid_level" << pyramid_level;

    fs << "features"
       << "[";
    for (int i = 0; i < (int)features.size(); ++i)
    {
        features[i].write(fs);
    }
    fs << "]"; // features
}

static Rect cropTemplates(std::vector<Template> &templates)
{
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int max_y = std::numeric_limits<int>::min();

    // First pass: find min/max feature x,y over all pyramid levels and modalities
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];

        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            int x = templ.features[j].x << templ.pyramid_level;
            int y = templ.features[j].y << templ.pyramid_level;
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
    }

    /// @todo Why require even min_x, min_y?
    if (min_x % 2 == 1)
        --min_x;
    if (min_y % 2 == 1)
        --min_y;

    // Second pass: set width/height and shift all feature positions
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];
        templ.width = (max_x - min_x) >> templ.pyramid_level;
        templ.height = (max_y - min_y) >> templ.pyramid_level;
        templ.tl_x = min_x >> templ.pyramid_level;
        templ.tl_y = min_y  >> templ.pyramid_level;

        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            templ.features[j].x -= templ.tl_x;
            templ.features[j].y -= templ.tl_y;
        }
    }

    return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

bool ColorGradientPyramid::selectScatteredFeatures(const std::vector<Candidate> &candidates,
                                                   std::vector<Feature> &features,
                                                   size_t num_features, float distance)
{
    features.clear();
    float distance_sq = distance * distance;
    int i = 0;

    bool first_select = true;

    while(true)
    {
        Candidate c = candidates[i];

        // Add if sufficient distance away from any previously chosen feature
        bool keep = true;
        for (int j = 0; (j < (int)features.size()) && keep; ++j)
        {
            Feature f = features[j];
            keep = (c.f.x - f.x) * (c.f.x - f.x) + (c.f.y - f.y) * (c.f.y - f.y) >= distance_sq;
        }
        if (keep)
            features.push_back(c.f);

        if (++i == (int)candidates.size()){
            bool num_ok = features.size() >= num_features;

            if(first_select){
                if(num_ok){
                    features.clear(); // we don't want too many first time
                    i = 0;
                    distance += 1.0f;
                    distance_sq = distance * distance;
                    continue;
                }else{
                    first_select = false;
                }
            }

            // Start back at beginning, and relax required distance
            i = 0;
            distance -= 1.0f;
            distance_sq = distance * distance;
             if (num_ok || distance < 3){
                 break;
             }
        }
    }
    return true;
}

/****************************************************************************************\
*                                                         Color gradient ColorGradient                                                                        *
\****************************************************************************************/

void hysteresisGradient(Mat &magnitude, Mat &quantized_angle,
                        Mat &angle, float threshold)
{
    // Quantize 360 degree range of orientations into 16 buckets
    // Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
    // for stability of horizontal and vertical features.
    Mat_<unsigned char> quantized_unfiltered;
    angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);

    // Zero out top and bottom rows
    /// @todo is this necessary, or even correct?
    memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
    memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
    // Zero out first and last columns
    for (int r = 0; r < quantized_unfiltered.rows; ++r)
    {
        quantized_unfiltered(r, 0) = 0;
        quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
    }

    // Mask 16 buckets into 8 quantized orientations
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        uchar *quant_r = quantized_unfiltered.ptr<uchar>(r);
        for (int c = 1; c < angle.cols - 1; ++c)
        {
            quant_r[c] &= 7;
        }
    }

    // Filter the raw quantized image. Only accept pixels where the magnitude is above some
    // threshold, and there is local agreement on the quantization.
    quantized_angle = Mat::zeros(angle.size(), CV_8U);
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        float *mag_r = magnitude.ptr<float>(r);

        for (int c = 1; c < angle.cols - 1; ++c)
        {
            if (mag_r[c] > threshold)
            {
                // Compute histogram of quantized bins in 3x3 patch around pixel
                int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                uchar *patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                // Find bin with the most votes from the patch
                int max_votes = 0;
                int index = -1;
                for (int i = 0; i < 8; ++i)
                {
                    if (max_votes < histogram[i])
                    {
                        index = i;
                        max_votes = histogram[i];
                    }
                }

                // Only accept the quantization if majority of pixels in the patch agree
                static const int NEIGHBOR_THRESHOLD = 5;
                if (max_votes >= NEIGHBOR_THRESHOLD)
                    quantized_angle.at<uchar>(r, c) = uchar(1 << index);
            }
        }
    }
}

// 将 16 个 int16_t 转换为 2组 8 个 float(32位)
static void int16_to_float_16x16(__m256i data,__m256& out1, __m256& out2) 
{
	// data 包含 16 个 int16_t: [a15, a14, ..., a0]

	// 步骤1：将 16 位扩展到 32 位（分成低8个和高8个）
	__m256i low_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(data));
	__m256i high_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data, 1));

	// 步骤2：将 32 位整数转换为单精度浮点
	out1 = _mm256_cvtepi32_ps(low_i32);
	out2 = _mm256_cvtepi32_ps(high_i32);
}

inline unsigned char IM_ClampToByte(int Value)
{
	if (Value < 0)
		return 0;
	else if (Value > 255)
		return 255;
	else
		return (unsigned char)Value;
	//return ((Value | ((signed int)(255 - Value) >> 31)) & ~((signed int)Value >> 31));
}

// 修正后的平方和计算（无溢出）
static inline __m256i compute_square_sum_32bit(__m256i gx16, __m256i gy16) 
{
	// 提取低8个值
	__m128i gx_low_i16 = _mm256_castsi256_si128(gx16);
	__m128i gx_high_i16 = _mm256_extracti128_si256(gx16, 1);
	__m128i gy_low_i16 = _mm256_castsi256_si128(gy16);
	__m128i gy_high_i16 = _mm256_extracti128_si256(gy16, 1);

	// 扩展到32位
	__m256i gx_low_i32 = _mm256_cvtepi16_epi32(gx_low_i16);
	__m256i gx_high_i32 = _mm256_cvtepi16_epi32(gx_high_i16);
	__m256i gy_low_i32 = _mm256_cvtepi16_epi32(gy_low_i16);
	__m256i gy_high_i32 = _mm256_cvtepi16_epi32(gy_high_i16);

	// 计算平方和
	__m256i square_low = _mm256_add_epi32(
		_mm256_mullo_epi32(gx_low_i32, gx_low_i32),
		_mm256_mullo_epi32(gy_low_i32, gy_low_i32)
	);
	__m256i square_high = _mm256_add_epi32(
		_mm256_mullo_epi32(gx_high_i32, gx_high_i32),
		_mm256_mullo_epi32(gy_high_i32, gy_high_i32)
	);

	// 重新打包为16个32位值（或保持分开）
	return square_low; // 需要根据需求处理
}

void _Sobel(unsigned char* Src, const int32_t Width, const int32_t Height, const int32_t start_row, const int32_t thread_stride, const int32_t Stride, 
    unsigned char* Dest, unsigned char** thread_local_buffers, float* sobelResMat_C3_32F_Speed, const int32_t Stride2, const int Channel, const int BlockSize, const int Block,
    float* sobleAngle_C3_32F_Speed, const int32_t Stride3, short* sobel_3dx_Speed, int Stride4, short* sobel_3dy_Speed, int Stride5)
{
	unsigned char* First = thread_local_buffers[0];
	unsigned char* Second = thread_local_buffers[1];
	unsigned char* Third = thread_local_buffers[2];
	for (int Y = start_row; Y < start_row + thread_stride; Y++) 
    {
		unsigned char* LinePS = Src + Y * Stride;
		unsigned char* LinePD = Dest + Y * Stride;
        float* LinePD_FloatTest = sobelResMat_C3_32F_Speed + Y * Stride2;
        float* LinePD_AngleSpeed = sobleAngle_C3_32F_Speed + Y * Stride3;
        short* sobel_dx_SpeedPtr = sobel_3dx_Speed + Y * Stride4;
        short* sobel_dy_SpeedPtr = sobel_3dy_Speed + Y * Stride5;
		if (Y != start_row)
        {
			unsigned char* Temp = First;
			First = Second;
			Second = Third;
			Third = Temp;
		}
		if (Y == Height - 1) 
        {
			memcpy(Third, Second, (Width + 2) * Channel);
		}
		else 
        {
			memcpy(Third, Src + (Y + 1) * Stride, Channel);
			memcpy(Third + Channel, Src + (Y + 1) * Stride, Width * Channel);                            //    由于备份了前面一行的数据，这里即使Src和Dest相同也是没有问题的
			memcpy(Third + (Width + 1) * Channel, Src + (Y + 1) * Stride + (Width - 1) * Channel, Channel);
		}
		if (Channel == 1) 
        {
			for (int X = 0; X < Width; X++)
			{
				float GX = First[X] - First[X + 2] + (Second[X] - Second[X + 2]) * 2 + Third[X] - Third[X + 2];
                float GY = First[X] + First[X + 2] + (First[X + 1] - Third[X + 1]) * 2 - Third[X] - Third[X + 2];
				//LinePD[X] = Table[min(GX * GX + GY * GY, 65025)];
                LinePD[X] = sqrtf(GX * GX + GY * GY + 0.0F);
			}
		}
		else
		{
			__m256i Zero = _mm256_setzero_si256();
			for (int X = 0; X < Block * BlockSize; X += BlockSize)
			{
				__m256i FirstP0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(First + X)));
				__m256i FirstP1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(First + X + 3)));
				__m256i FirstP2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(First + X + 6)));

				__m256i SecondP0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(Second + X)));
				__m256i SecondP2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(Second + X + 6)));

				__m256i ThirdP0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(Third + X)));
				__m256i ThirdP1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(Third + X + 3)));
				__m256i ThirdP2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(Third + X + 6)));


				//GX0	GX1	    GX2    GX3    GX4    GX5    GX6    GX7     GX8	 GX9	 GX10    GX11    GX12    GX13    GX14    GX15
				__m256i GX16 = _mm256_abs_epi16(_mm256_adds_epi16(_mm256_adds_epi16(_mm256_subs_epi16(FirstP2, FirstP0), _mm256_slli_epi16(_mm256_subs_epi16(SecondP2, SecondP0), 1)), _mm256_subs_epi16(ThirdP2, ThirdP0)));
				//GY0   GY1     GY2    GY3    GY4    GY5    GY6    GY7     GY8   GY9     GY10    GY11    GY12    GY13    GY14    GY15
				__m256i GY16 = _mm256_abs_epi16(_mm256_subs_epi16(_mm256_adds_epi16(_mm256_adds_epi16(ThirdP0, ThirdP2), _mm256_slli_epi16(_mm256_subs_epi16(ThirdP1, FirstP1), 1)), _mm256_adds_epi16(FirstP0, FirstP2)));
				//GX0　　GY0　　GX1　　GY1　　GX2　　GY2　　GX3　　GY3    GX4    GY4     GX5     GY5      GX6     GY6     GX7     GY7
                //其实应该是:
                //GX0　　GY0　　GX1　　GY1　　GX2　　GY2　　GX3　　GY3    GX8　　GY8　　 GX9　　 GY9　　  GX10　  GY10　　GX11　  GY11
				__m256i GXYL = _mm256_unpacklo_epi16(GX16, GY16);
				//GX8　　GY8　　GX9　　GY9　　GX10　 GY10　 GX11   GY11   GX12   GY12    GX13    GY13     GX14    GY14    GX15    GY15     
                //其实应该是:
				//GX4    GY4    GX5    GY5    GX6    GY6    GX7    GY7    GX12   GY12    GX13    GY13     GX14    GY14    GX15    GY15     
				__m256i GXYH = _mm256_unpackhi_epi16(GX16, GY16);

				__m256 squareSum_Low = _mm256_cvtepi32_ps(_mm256_madd_epi16(GXYL, GXYL));
				__m256 squareSum_Height = _mm256_cvtepi32_ps(_mm256_madd_epi16(GXYH, GXYH));
                //sqrt(GX0^2+GY0^2)   sqrt(GX1^2+GY1^2)  sqrt(GX2^2+GY2^2)    sqrt(GX3^2+GY3^2)    sqrt(GX4^2+GY4^2)    sqrt(GX5^2+GY5^2)    sqrt(GX6^2+GY6^2)    sqrt(GX7^2+GY7^2)
                //其实应该是:
                //sqrt(GX0^2+GY0^2)   sqrt(GX1^2+GY1^2)  sqrt(GX2^2+GY2^2)    sqrt(GX3^2+GY3^2)    sqrt(GX8^2+GY8^2)   sqrt(GX9^2+GY9^2)  sqrt(GX10^2+GY10^2)  sqrt(GX11^2+GY11^2)
				__m256i ResultL = _mm256_cvtps_epi32(_mm256_sqrt_ps(squareSum_Low));
                //sqrt(GX8^2+GY8^2)   sqrt(GX9^2+GY9^2)  sqrt(GX10^2+GY10^2)  sqrt(GX11^2+GY11^2)  sqrt(GX12^2+GY12^2)  sqrt(GX13^2+GY13^2)  sqrt(GX14^2+GY14^2)  sqrt(GX15^2+GY15^2)
                //其实应该是:
                //sqrt(GX4^2+GY4^2)    sqrt(GX5^2+GY5^2)    sqrt(GX6^2+GY6^2)    sqrt(GX7^2+GY7^2) sqrt(GX12^2+GY12^2)  sqrt(GX13^2+GY13^2)  sqrt(GX14^2+GY14^2)  sqrt(GX15^2+GY15^2)
				__m256i ResultH = _mm256_cvtps_epi32(_mm256_sqrt_ps(squareSum_Height));
                //_mm256_packus_epi32 是一个 AVX2 intrinsic 函数，作用是将两组各 8 个有符号 32 位整数（int32_t）饱和打包为一组 16 个无符号 16 位整数（uint16_t）
                //_mm256_packus_epi16 是一个 AVX2 intrinsic 函数，作用是将两组各 16 个有符号 16 位整数（int16_t）饱和打包为一组 32 个无符号 8 位整数（uint8_t）。
			    __m256i Result = _mm256_packus_epi16(_mm256_packus_epi32(ResultL, ResultH), Zero);

                // 源码屏蔽了上面这一行( __m256i Result = _mm256_packus_epi16(_mm256_packus_epi32(ResultL, ResultH), Zero); ),直接调用了下面的两行,会有问题,星号里面的两行后加的.
                // 问题原因在: 32位整数直接写入8位内存
				//__m128i Ans1 = _mm256_castsi256_si128(ResultL);
				//_mm_storeu_si128((__m128i*)(LinePD + X), Ans1);

				//__m128i Ans2 = _mm256_castsi256_si128(ResultH);
				//_mm_storeu_si128((__m128i*)(LinePD + X + 8), Ans2);

                /***************************************************/
                //_mm256_castsi256_si128 是一个 AVX2 intrinsic 函数，作用是将 256 位整数向量寄存器重新解释为 128 位整数向量寄存器，并截取其低 128 位数据。
				__m128i Ans1 = _mm256_castsi256_si128(Result);
				_mm_storeu_si128((__m128i*)(LinePD + X), Ans1);
                //_mm256_extracti128_si256 是一个 AVX2 intrinsic 函数，作用是从 256 位整数向量中提取一个 128 位通道。
				//通道提取：从 256 位寄存器中提取指定的 128 位通道
				//索引选择：imm8 参数指定提取低通道(0)或高通道(1)
				__m128i Ans2 = _mm256_extracti128_si256(Result, 1);
				_mm_storeu_si128((__m128i*)(LinePD + X + 8), Ans2);
                /****************************************************/

				 // 原始数据
				//float GXYL[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
				//float GXYH[8] = { 10, 20, 30, 40, 50, 60, 70, 80 };
				// 步骤1: 提取低4个元素和高4个元素
				// low_low   = {1,2,3,4} (低位)
	            // low_high  = {5,6,7,8} (高位)
	            // high_low  = {10,20,30,40}
	            // high_high = {50,60,70,80}
				__m128 low_low_128 = _mm256_castps256_ps128(squareSum_Low);      // 低128位
				__m128 low_high_128 = _mm256_extractf128_ps(squareSum_Low, 1);    // 高128位
				__m128 high_low_128 = _mm256_castps256_ps128(squareSum_Height);
				__m128 high_high_128 = _mm256_extractf128_ps(squareSum_Height, 1);

				// 步骤2: 重新组合成两个256位向量
				// part1 = {1,2,3,4, 10,20,30,40}
				__m256 part1 = _mm256_set_m128(high_low_128, low_low_128);
				// part2 = {5,6,7,8, 50,60,70,80}
				__m256 part2 = _mm256_set_m128(high_high_128, low_high_128);

				_mm256_storeu_ps(LinePD_FloatTest + X, part1);
				_mm256_storeu_ps(LinePD_FloatTest + X + 8, part2);

                // 存atan2图
                // 16个16位
    //            __m256i GX16_Speed = _mm256_adds_epi16(_mm256_adds_epi16(_mm256_subs_epi16(FirstP2, FirstP0), _mm256_slli_epi16(_mm256_subs_epi16(SecondP2, SecondP0), 1)), _mm256_subs_epi16(ThirdP2, ThirdP0));
    //            __m256i GY16_Speed = _mm256_subs_epi16(_mm256_adds_epi16(_mm256_adds_epi16(ThirdP0, ThirdP2), _mm256_slli_epi16(_mm256_subs_epi16(ThirdP1, FirstP1), 1)), _mm256_adds_epi16(FirstP0, FirstP2));
				//// 8个16位
				//__m256 out1_X, out2_X;
    //            __m256 out1_Y, out2_Y;
				//int16_to_float_16x16(GX16_Speed, out1_X, out2_X);
    //            int16_to_float_16x16(GY16_Speed, out1_Y, out2_Y);
				//// 8个32位
				//__m256 out_atan2_Low = _mm256_atan2_ps(out1_Y, out1_X);
    //            __m256 out_atan2_High = _mm256_atan2_ps(out2_Y, out2_X);
    //            _mm256_storeu_ps(LinePD_AngleSpeed + X, out_atan2_Low);
    //            _mm256_storeu_ps(LinePD_AngleSpeed + X + 8  , out_atan2_High);

				//GX0	GX1	    GX2    GX3    GX4    GX5    GX6    GX7     GX8	 GX9	 GX10    GX11    GX12    GX13    GX14    GX15
				__m256i GX16_N0_abs = _mm256_adds_epi16(_mm256_adds_epi16(_mm256_subs_epi16(FirstP2, FirstP0), _mm256_slli_epi16(_mm256_subs_epi16(SecondP2, SecondP0), 1)), _mm256_subs_epi16(ThirdP2, ThirdP0));
				//GY0   GY1     GY2    GY3    GY4    GY5    GY6    GY7     GY8   GY9     GY10    GY11    GY12    GY13    GY14    GY15
				__m256i GY16_N0_abs = _mm256_subs_epi16(_mm256_adds_epi16(_mm256_adds_epi16(ThirdP0, ThirdP2), _mm256_slli_epi16(_mm256_subs_epi16(ThirdP1, FirstP1), 1)), _mm256_adds_epi16(FirstP0, FirstP2));
                // 16个16位
				_mm256_storeu_si256((__m256i*)(sobel_dx_SpeedPtr + X), GX16_N0_abs);
                _mm256_storeu_si256((__m256i*)(sobel_dy_SpeedPtr + X), GY16_N0_abs);
            }

			for (int X = Block * BlockSize; X < Width * 3; X++)
			{
				int GX = First[X] - First[X + 6] + (Second[X] - Second[X + 6]) * 2 + Third[X] - Third[X + 6];
				int GY = First[X] + First[X + 6] + (First[X + 3] - Third[X + 3]) * 2 - Third[X] - Third[X + 6];
				LinePD[X] = IM_ClampToByte(sqrtf(GX * GX + GY * GY + 0.0F));

                sobel_dx_SpeedPtr[X] = -(First[X] - First[X + 6] + (Second[X] - Second[X + 6]) * 2 + Third[X] - Third[X + 6]);
                sobel_dy_SpeedPtr[X] = -(First[X] + First[X + 6] + (First[X + 3] - Third[X + 3]) * 2 - Third[X] - Third[X + 6]);
			}
		}
	}
}

void Sobel_AVX2(unsigned char* Src, unsigned char* Dest, int Width, int Height, int Stride, float* sobelResMat_C3_32F_Speed, int Stride2,
    float* sobleAngle_C3_32F_Speed, int Stride3, short* sobel_3dx_Speed, int Stride4, short* sobel_3dy_Speed, int Stride5)
{
	//INIT
	int Channel = Stride / Width;
    int BlockSize = 16;
    int Block = (Width * Channel) / BlockSize;

	//Run                                         
	// Height >> 4,右移4位，相当于除以 16（2^4 = 16）,含义：图像高度分成多少个16像素高度的块. 为什么是16？因为，高度优化：16像素块大小通常是SIMD优化的友好单位（如SSE/AVX）
	//std::thread::hardware_concurrency()--返回系统真正支持的硬件并发线程数（CPU核心数或超线程数）,例如：4核8线程的CPU → 返回 8
    const int32_t hw_concur = std::min(100/*Height >> 4*/, static_cast<int32_t>(std::thread::hardware_concurrency()));

	std::vector<std::future<void>> fut(hw_concur);
    std::vector<unsigned char*> allocated_buffers;

	const int thread_stride = (Height - 1) / hw_concur + 1;
    int i = 0, start = 0;

	for (; i < std::min(Height, hw_concur); i++, start += thread_stride)
	{
		// 为每个线程分配独立的缓冲区
		unsigned char* rowCopy = (unsigned char*)_mm_malloc((Width + 2) * 3 * Channel, 32);
		allocated_buffers.push_back(rowCopy);

		unsigned char* first = rowCopy;
		unsigned char* second = rowCopy + (Width + 2) * Channel;
		unsigned char* third = rowCopy + (Width + 2) * 2 * Channel;

		if (start == 0)
		{
			// 线程1：处理顶部
			memcpy(second, Src, Channel);
			memcpy(second + Channel, Src, Width * Channel);
			memcpy(second + (Width + 1) * Channel, Src + (Width - 1) * Channel, Channel);
			memcpy(first, second, (Width + 2) * Channel);

			if (Height > 1) 
            {
				unsigned char* nextRow = Src + Stride;
				memcpy(third, nextRow, Channel);
				memcpy(third + Channel, nextRow, Width * Channel);
				memcpy(third + (Width + 1) * Channel, nextRow + (Width - 1) * Channel, Channel);
			}
			else 
            {
				memcpy(third, second, (Width + 2) * Channel);
			}
		}
		else
		{
			// 线程2：处理中间部分
			int prev_row = start - 1;
			int curr_row = start;
			int next_row = start + 1;

			// First: 上一行
			unsigned char* prevSrc = Src + prev_row * Stride;
			memcpy(first, prevSrc, Channel);
			memcpy(first + Channel, prevSrc, Width * Channel);
			memcpy(first + (Width + 1) * Channel, prevSrc + (Width - 1) * Channel, Channel);

			// Second: 当前行
			unsigned char* currSrc = Src + curr_row * Stride;
			memcpy(second, currSrc, Channel);
			memcpy(second + Channel, currSrc, Width * Channel);
			memcpy(second + (Width + 1) * Channel, currSrc + (Width - 1) * Channel, Channel);

			// Third: 下一行
			if (next_row < Height) 
            {
				unsigned char* nextSrc = Src + next_row * Stride;
				memcpy(third, nextSrc, Channel);
				memcpy(third + Channel, nextSrc, Width * Channel);
				memcpy(third + (Width + 1) * Channel, nextSrc + (Width - 1) * Channel, Channel);
			}
			else 
            {
				memcpy(third, second, (Width + 2) * Channel);
			}
		}
		// 为每个线程创建独立的 buffer 数组（在堆上分配）
		unsigned char** thread_buffers = new unsigned char* [3];
		thread_buffers[0] = first;
		thread_buffers[1] = second;
		thread_buffers[2] = third;

		//fut[i] = std::async(std::launch::async, _Sobel, Src, Width, Height, start, thread_stride, Stride, Dest, thread_buffers, sobelResMat_C3_16S_Speed, Stride2);
		// 传递堆上分配的数组，每个线程独立
		fut[i] = std::async(std::launch::async, [=]() {
			_Sobel(Src, Width, Height, start, thread_stride, Stride, Dest,
				thread_buffers, sobelResMat_C3_32F_Speed, Stride2, Channel, BlockSize, Block, sobleAngle_C3_32F_Speed, Stride3, sobel_3dx_Speed, Stride4, sobel_3dy_Speed, Stride5);
			delete[] thread_buffers;  // 线程结束后释放
            });
	}
    for (int j = 0; j < i; ++j)
    {
		if (fut[j].valid()) 
        {
			fut[j].wait();
		}
    }

	for (auto buf : allocated_buffers) 
    {
		_mm_free(buf);
	}
}

void _phase(float* SrcX, float* SrcY, float* DestAngle, const int32_t Width, const int32_t Height, const int32_t start_row, const int32_t thread_stride,
    const int32_t Stride, const int Channel, const int BlockSize, const int Block, bool angleInDegrees = false)
{
	const float RAD_TO_DEG = 180.0f / 3.14159265358979323846f;
	const __m256 rad_to_deg_vec = _mm256_set1_ps(RAD_TO_DEG);
	const __m256 zero_vec = _mm256_setzero_ps();
	const __m256 three_sixty_vec = _mm256_set1_ps(360.0f);

    for (int Y = start_row; Y < start_row + thread_stride; Y++)
    {
        float* LinePS_X = SrcX + Y * Stride;
        float* LinePS_Y = SrcY + Y * Stride;
        float* LinePD = DestAngle + Y * Stride;

		if (Channel == 1)
		{
			__m256i Zero = _mm256_setzero_si256();
			for (int X = 0; X < Block * BlockSize; X += BlockSize)
			{
				__m256 SrcXP0 = _mm256_loadu_ps(LinePS_X + X);
                __m256 SrcYP0 = _mm256_loadu_ps(LinePS_Y + X);
				// 8个32位
				__m256 out_atan2= _mm256_atan2_ps(SrcYP0, SrcXP0);
				if (angleInDegrees) 
                {
					out_atan2 = _mm256_mul_ps(out_atan2, rad_to_deg_vec);
				}
				// 将负角度转换为 0-360 范围
			   // 如果 out_atan2 < 0，则 out_atan2 += 360
				__m256 mask = _mm256_cmp_ps(out_atan2, zero_vec, _CMP_LT_OQ);  // 小于0的掩码
				__m256 add_360 = _mm256_and_ps(mask, three_sixty_vec);          // 只对负数加360
				out_atan2 = _mm256_add_ps(out_atan2, add_360);

                _mm256_storeu_ps(LinePD + X, out_atan2);
			}
		}
        else
        {
			//__m256i Zero = _mm256_setzero_si256();
   //         for (int X = 0; X < Block * BlockSize; X += BlockSize)
   //         {
			//	__m256i SrcXP0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(LinePS_X + X)));
			//	__m256i SrcYP0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(LinePS_Y + X)));
			//	__m256 out1_X, out2_X;
	  //          __m256 out1_Y, out2_Y;
			//	int16_to_float_16x16(SrcXP0, out1_X, out2_X);
	  //          int16_to_float_16x16(SrcYP0, out1_Y, out2_Y);
			//	// 8个32位
			//	__m256 out_atan2_Low = _mm256_atan2_ps(out1_Y, out1_X);
	  //          __m256 out_atan2_High = _mm256_atan2_ps(out2_Y, out2_X);
   //             _mm256_storeu_ps(DestAngle + X, out_atan2_Low);
	  //          _mm256_storeu_ps(LinePD + X + 8 , out_atan2_High);
   //         }
        }
    }
}

void phaseSelf(float* SrcX, float* SrcY, float* DestAngle, const int32_t Width, const int32_t Height, const int32_t Stride, bool angleInDegrees = false)
{
	//INIT
	int Channel = Stride / Width;
	int BlockSize = 8;
	int Block = (Width * Channel) / BlockSize;

	//Run                                         
	// Height >> 4,右移4位，相当于除以 16（2^4 = 16）,含义：图像高度分成多少个16像素高度的块. 为什么是16？因为，高度优化：16像素块大小通常是SIMD优化的友好单位（如SSE/AVX）
	//std::thread::hardware_concurrency()--返回系统真正支持的硬件并发线程数（CPU核心数或超线程数）,例如：4核8线程的CPU → 返回 8
	const int32_t hw_concur = std::min(100/*Height >> 4*/, static_cast<int32_t>(std::thread::hardware_concurrency()));

	std::vector<std::future<void>> fut(hw_concur);
	std::vector<unsigned char*> allocated_buffers;

	const int thread_stride = (Height - 1) / hw_concur + 1;
	int i = 0, start = 0;

    for (; i < std::min(Height, hw_concur); i++, start += thread_stride)
    {
		fut[i] = std::async(std::launch::async, [=]() {
            _phase(SrcX, SrcY, DestAngle, Width, Height, start, thread_stride, Stride, Channel, BlockSize, Block, angleInDegrees);
			});
    }
}

static void quantizedOrientations(const Mat &src, Mat &magnitude,
                                  Mat &angle, Mat& angle_ori, float threshold)
{
    Mat smoothed = src.clone();
    // Compute horizontal and vertical image derivatives on all color channels separately
    static const int SIGMA = 1;
    // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
    //GaussianBlur(src, smoothed, Size(SIGMA * 8 + 1, SIGMA * 8 + 1), SIGMA, 0, BORDER_REPLICATE);
    // 先把Padding区域去除,纯黑区域且当SIGM=1时,耗时会大大增加
    cv::Rect roiRect(g_paddingPixel, g_paddingPixel, g_imgWidthNoPadding, g_imgHeightNoPadding);
    cv::Mat roiImg = src(roiRect).clone();
	int Height = roiImg.rows;
	int Width = roiImg.cols;
	unsigned char* roiImgPtr = roiImg.data;
	unsigned char* Dest = new unsigned char[Height * Width * 3];
    int Stride = Width * 3;
    GaussBlur_SSE(roiImgPtr, Dest, Width, Height, Stride, SIGMA);
    Mat dst(Height, Width, CV_8UC3, Dest);
    dst.copyTo(smoothed(roiRect));

    if(src.channels() == 1)
    {
        Mat sobel_dx, sobel_dy, sobel_ag;
        Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
        Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
        magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
        phase(sobel_dx, sobel_dy, sobel_ag, true);
        hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
        angle_ori = sobel_ag;

		Mat ptr_SobelResMat_C1_8U_Speed = Mat::zeros(smoothed.size(), CV_8U);
        Mat sobelResMat_C1_32F_Speed = Mat::zeros(smoothed.size(), CV_32F);
		Mat sobleAngle_C1_32F_Speed = cv::Mat::zeros(smoothed.size(), CV_32F);
		Mat sobel_1dx_Speed = Mat::zeros(smoothed.size(), CV_16S);
		Mat sobel_1dy_Speed = Mat::zeros(smoothed.size(), CV_16S);
		Sobel_AVX2(smoothed.ptr(), ptr_SobelResMat_C1_8U_Speed.ptr(), smoothed.cols, smoothed.rows, smoothed.step1(), 
            sobelResMat_C1_32F_Speed.ptr<float>(), sobelResMat_C1_32F_Speed.step1(), sobleAngle_C1_32F_Speed.ptr<float>(), sobleAngle_C1_32F_Speed.step1(),
            sobel_1dx_Speed.ptr<short>(), sobel_1dx_Speed.step1(), sobel_1dy_Speed.ptr<short>(), sobel_1dx_Speed.step1());

		// 转换为浮点用于对比
		Mat sobel_dx_Test_F;
        ptr_SobelResMat_C1_8U_Speed.convertTo(sobel_dx_Test_F, CV_32F);
    }
    else
    {

        magnitude.create(src.size(), CV_32F);

        // Allocate temporary buffers
        Size size = src.size();
        //Mat sobel_3dx;              // per-channel horizontal derivative
        //Mat sobel_3dy;              // per-channel vertical derivative
        //Mat sobel_dx(size, CV_32F); // maximum horizontal derivative
        //Mat sobel_dy(size, CV_32F); // maximum vertical derivative
        //Mat sobel_ag;               // final gradient orientation (unquantized)

        //Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
        //Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);

		/*******************************************************************/

		Mat sobelResMat_C3_8U_Speed = Mat::zeros(smoothed.size(), CV_8UC3);
        Mat sobelResMat_C3_32F_Speed = Mat::zeros(smoothed.size(), CV_32FC3);
        Mat sobleAngle_C3_32F_Speed = cv::Mat::zeros(smoothed.size(), CV_32FC3);
        Mat sobel_3dx_Speed = Mat::zeros(smoothed.size(), CV_16SC3);
        Mat sobel_3dy_Speed = Mat::zeros(smoothed.size(), CV_16SC3);
		//step1(i):每一维元素的通道数,函数默认参数是0,即: step1() == step1(0)
		//step[i]:每一维元素的大小，单位字节
		Sobel_AVX2(smoothed.ptr(), sobelResMat_C3_8U_Speed.ptr(), smoothed.cols, smoothed.rows, smoothed.step1(), 
            sobelResMat_C3_32F_Speed.ptr<float>(), sobelResMat_C3_32F_Speed.step1(), sobleAngle_C3_32F_Speed.ptr<float>(), sobleAngle_C3_32F_Speed.step1(),
            sobel_3dx_Speed.ptr<short>(), sobel_3dx_Speed.step1(), sobel_3dy_Speed.ptr<short>(), sobel_3dy_Speed.step1());

        Mat SobelResMat_Magnitude_Max(size, CV_32F); // maximum horizontal derivative
        Mat magnitude_Test = Mat::zeros(src.size(), CV_32F);
        Mat sobel_dx_Max = Mat::zeros(src.size(), CV_32F);
        Mat sobel_dy_Max = Mat::zeros(src.size(), CV_32F);

        float* ptr_SobelResMat_Magnitude_Max = SobelResMat_Magnitude_Max.ptr<float>();
		float* ptrmg_Test = magnitude_Test.ptr<float>();
        float* ptr_sobel_dx_Max = sobel_dx_Max.ptr<float>();
        float* ptr_sobel_dy_Max = sobel_dy_Max.ptr<float>();

        float* ptr_SobelResMat_C3_32F_Speed = sobelResMat_C3_32F_Speed.ptr<float>();
		short* ptr_sobel_3dx_Speed = sobel_3dx_Speed.ptr<short>();
		short* ptr_sobel_3dy_Speed = sobel_3dy_Speed.ptr<short>();

		const int length_SobelResMat_Magnitude_Max = static_cast<const int>(SobelResMat_Magnitude_Max.step1());
		const int length_magnitude_Test = static_cast<const int>(magnitude_Test.step1());
		const int length_sobel_dx_Max = static_cast<const int>(sobel_dx_Max.step1());
		const int length_sobel_dy_Max = static_cast<const int>(sobel_dy_Max.step1());

		const int length_sobelResMat_C3_32F_Speed = static_cast<const int>(sobelResMat_C3_32F_Speed.step1());
		const int length_sobel_3dx_Max = static_cast<const int>(sobel_3dx_Speed.step1());
		const int length_sobel_3dy_Max = static_cast<const int>(sobel_3dy_Speed.step1());

		for (int r = 0; r < sobelResMat_C3_32F_Speed.rows; ++r)
		{
			int ind = 0;

			for (int i = 0; i < length_sobelResMat_C3_32F_Speed; i += 3)
			{
				// Use the gradient orientation of the channel whose magnitude is largest
				float mag1 = ptr_SobelResMat_C3_32F_Speed[i + 0];
                float mag2 = ptr_SobelResMat_C3_32F_Speed[i + 1];
                float mag3 = ptr_SobelResMat_C3_32F_Speed[i + 2];

				if (mag1 >= mag2 && mag1 >= mag3)
				{
                    ptr_SobelResMat_Magnitude_Max[ind] = ptr_SobelResMat_C3_32F_Speed[i];
					ptrmg_Test[ind] = (float)mag1;
                    ptr_sobel_dx_Max[ind] = (float)ptr_sobel_3dx_Speed[i];
                    ptr_sobel_dy_Max[ind] = (float)ptr_sobel_3dy_Speed[i];
				}
				else if (mag2 >= mag1 && mag2 >= mag3)
				{
                    ptr_SobelResMat_Magnitude_Max[ind] = ptr_SobelResMat_C3_32F_Speed[i + 1];
					ptrmg_Test[ind] = (float)mag2;
					ptr_sobel_dx_Max[ind] = (float)ptr_sobel_3dx_Speed[i + 1];
					ptr_sobel_dy_Max[ind] = (float)ptr_sobel_3dy_Speed[i + 1];
				}
				else
				{
                    ptr_SobelResMat_Magnitude_Max[ind] = ptr_SobelResMat_C3_32F_Speed[i + 2];
					ptrmg_Test[ind] = (float)mag3;
					ptr_sobel_dx_Max[ind] = (float)ptr_sobel_3dx_Speed[i + 2];
					ptr_sobel_dy_Max[ind] = (float)ptr_sobel_3dy_Speed[i + 2];
				}
				++ind;
			}
            ptr_SobelResMat_C3_32F_Speed += length_sobelResMat_C3_32F_Speed;
            ptr_sobel_3dx_Speed += length_sobel_3dx_Max;
            ptr_sobel_3dy_Speed += length_sobel_3dy_Max;
            ptr_SobelResMat_Magnitude_Max += length_SobelResMat_Magnitude_Max;
            ptrmg_Test += length_magnitude_Test;
            ptr_sobel_dx_Max += length_sobel_dx_Max;
            ptr_sobel_dy_Max += length_sobel_dy_Max;
		}

        Mat sobleAngle_C1_32F_Speed = cv::Mat::zeros(smoothed.size(), CV_32F);
        phaseSelf(sobel_dx_Max.ptr<float>(), sobel_dy_Max.ptr<float>(), sobleAngle_C1_32F_Speed.ptr<float>(), smoothed.cols, smoothed.rows, sobel_dx_Max.step1(), true);

        /*******************************************************************/

        //short *ptrx = (short *)sobel_3dx.data;
        //short *ptry = (short *)sobel_3dy.data;
        //float *ptr0x = (float *)sobel_dx.data;
        //float *ptr0y = (float *)sobel_dy.data;
        //float *ptrmg = (float *)magnitude.data;

        //const int length1 = static_cast<const int>(sobel_3dx.step1());
        //const int length2 = static_cast<const int>(sobel_3dy.step1());
        //const int length3 = static_cast<const int>(sobel_dx.step1());
        //const int length4 = static_cast<const int>(sobel_dy.step1());
        //const int length5 = static_cast<const int>(magnitude.step1());
        //const int length0 = sobel_3dy.cols * 3;

        //for (int r = 0; r < sobel_3dy.rows; ++r)
        //{
        //    int ind = 0;

        //    for (int i = 0; i < length0; i += 3)
        //    {
        //        // Use the gradient orientation of the channel whose magnitude is largest
        //        int mag1 = ptrx[i + 0] * ptrx[i + 0] + ptry[i + 0] * ptry[i + 0];
        //        int mag2 = ptrx[i + 1] * ptrx[i + 1] + ptry[i + 1] * ptry[i + 1];
        //        int mag3 = ptrx[i + 2] * ptrx[i + 2] + ptry[i + 2] * ptry[i + 2];

        //        if (mag1 >= mag2 && mag1 >= mag3)
        //        {
        //            ptr0x[ind] = ptrx[i];
        //            ptr0y[ind] = ptry[i];
        //            ptrmg[ind] = (float)mag1;
        //        }
        //        else if (mag2 >= mag1 && mag2 >= mag3)
        //        {
        //            ptr0x[ind] = ptrx[i + 1];
        //            ptr0y[ind] = ptry[i + 1];
        //            ptrmg[ind] = (float)mag2;
        //        }
        //        else
        //        {
        //            ptr0x[ind] = ptrx[i + 2];
        //            ptr0y[ind] = ptry[i + 2];
        //            ptrmg[ind] = (float)mag3;
        //        }
        //        ++ind;
        //    }
        //    ptrx += length1;
        //    ptry += length2;
        //    ptr0x += length3;
        //    ptr0y += length4;
        //    ptrmg += length5;
        //}

        // Calculate the final gradient orientations
        //phase(sobel_dx, sobel_dy, sobel_ag, true);
        //hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
        //angle_ori = sobel_ag;
        magnitude = magnitude_Test.clone();
        hysteresisGradient(magnitude_Test, angle, sobleAngle_C1_32F_Speed, threshold* threshold);
        angle_ori = sobleAngle_C1_32F_Speed;
    }
}

ColorGradientPyramid::ColorGradientPyramid(const Mat &_src, const Mat &_mask,
                                           float _weak_threshold, size_t _num_features,
                                           float _strong_threshold)
    : src(_src),
      mask(_mask),
      pyramid_level(0),
      weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold)
{
    update();
}

void ColorGradientPyramid::update()
{
    quantizedOrientations(src, magnitude, angle, angle_ori, weak_threshold);
}

void ColorGradientPyramid::pyrDown()
{
    // Some parameters need to be adjusted
    num_features /= 2; /// @todo Why not 4?
    ++pyramid_level;

    // Downsample the current inputs
    Size size(src.cols / 2, src.rows / 2);
    Mat next_src;
    cv::pyrDown(src, next_src, size);
    src = next_src;

    if (!mask.empty())
    {
        Mat next_mask;
        resize(mask, next_mask, size, 0.0, 0.0, INTER_NEAREST);
        mask = next_mask;
    }
    g_paddingPixel /= 2;
    g_imgWidthNoPadding /= 2;
    g_imgHeightNoPadding /= 2;
    update();
}

void ColorGradientPyramid::quantize(Mat &dst) const
{
    dst = Mat::zeros(angle.size(), CV_8U);
    angle.copyTo(dst, mask);
}

bool ColorGradientPyramid::extractTemplate(Template &templ) const
{
    // Want features on the border to distinguish from background
    Mat local_mask;
    if (!mask.empty())
    {
        erode(mask, local_mask, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
//        subtract(mask, local_mask, local_mask);
    }

    std::vector<Candidate> candidates;
    bool no_mask = local_mask.empty();
    float threshold_sq = strong_threshold * strong_threshold;

    int nms_kernel_size = 5;
    cv::Mat magnitude_valid = cv::Mat(magnitude.size(), CV_8UC1, cv::Scalar(255));

    for (int r = 0+nms_kernel_size/2; r < magnitude.rows-nms_kernel_size/2; ++r)
    {
        const uchar *mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

        for (int c = 0+nms_kernel_size/2; c < magnitude.cols-nms_kernel_size/2; ++c)
        {
            if (no_mask || mask_r[c])
            {
                float score = 0;
                if(magnitude_valid.at<uchar>(r, c)>0){
                    score = magnitude.at<float>(r, c);
                    bool is_max = true;
                    for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++){
                        for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++){
                            if(r_offset == 0 && c_offset == 0) continue;

                            if(score < magnitude.at<float>(r+r_offset, c+c_offset)){
                                score = 0;
                                is_max = false;
                                break;
                            }
                        }
                        if(!is_max) break;
                    }

                    if(is_max){
                        for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++){
                            for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++){
                                if(r_offset == 0 && c_offset == 0) continue;
                                magnitude_valid.at<uchar>(r+r_offset, c+c_offset) = 0;
                            }
                        }
                    }
                }

                if (score > threshold_sq && angle.at<uchar>(r, c) > 0)
                {
                    candidates.push_back(Candidate(c, r, getLabel(angle.at<uchar>(r, c)), score));
                    candidates.back().f.theta = angle_ori.at<float>(r, c);
                }
            }
        }
    }
    // We require a certain number of features
    if (candidates.size() < num_features){
        if(candidates.size() <= 4) {
            std::cout << "too few features, abort" << std::endl;
            return false;
        }
        std::cout << "have no enough features, exaustive mode" << std::endl;
    }

    // NOTE: Stable sort to agree with old code, which used std::list::sort()
    std::stable_sort(candidates.begin(), candidates.end());

    // Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
    float distance = static_cast<float>(candidates.size() / num_features + 1);

    // selectScatteredFeatures always return true
    if (!selectScatteredFeatures(candidates, templ.features, num_features, distance))
    {
        return false;
    }

    // Size determined externally, needs to match templates for other modalities
    templ.width = -1;
    templ.height = -1;
    templ.pyramid_level = pyramid_level;

    return true;
}

ColorGradient::ColorGradient()
    : weak_threshold(30.0f),
      num_features(63),
      strong_threshold(60.0f)
{
}

ColorGradient::ColorGradient(float _weak_threshold, size_t _num_features, float _strong_threshold)
    : weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold)
{
}

static const char CG_NAME[] = "ColorGradient";

std::string ColorGradient::name() const
{
    return CG_NAME;
}

void ColorGradient::read(const FileNode &fn)
{
    String type = fn["type"];
    CV_Assert(type == CG_NAME);

    weak_threshold = fn["weak_threshold"];
    num_features = int(fn["num_features"]);
    strong_threshold = fn["strong_threshold"];
}

void ColorGradient::write(FileStorage &fs) const
{
    fs << "type" << CG_NAME;
    fs << "weak_threshold" << weak_threshold;
    fs << "num_features" << int(num_features);
    fs << "strong_threshold" << strong_threshold;
}
/****************************************************************************************\
*                                                                 Response maps                                                                                    *
\****************************************************************************************/

// 作用：将 src 和 dst 对应位置的字节进行按位 OR (|) 操作，结果写回 dst
// 即：dst[r][c] |= src[r][c]
static void orUnaligned8u(const uchar *src, const int src_stride,
                          uchar *dst, const int dst_stride,
                          const int width, const int height)
{
    // 主循环：逐行处理
    for (int r = 0; r < height; ++r)
    {
        int c = 0;

        // 逐字节处理，直到 src 指针达到 16 字节对齐
        // 确保后续 SIMD 操作在对齐的内存上进行，提高性能
        while (reinterpret_cast<unsigned long long>(src + c) % 16 != 0) {
            dst[c] |= src[c];
            c++;
        }

        // 使用 MIPP 库（跨平台 SIMD 封装）进行批量处理
        // mipp::N<uint8_t>() 返回 SIMD 寄存器能容纳的 uint8_t 元素个数（如 AVX2 为 32，SSE 为 16）
        // orb 函数执行按位或操作
        // 一次处理一批数据，大幅提升性能
        for (; c <= width-mipp::N<uint8_t>(); c+=mipp::N<uint8_t>()){
            mipp::Reg<uint8_t> src_v((uint8_t*)src + c);
            mipp::Reg<uint8_t> dst_v((uint8_t*)dst + c);
            // mipp::orb(a, b) 是向量化的按位 OR
            mipp::Reg<uint8_t> res_v = mipp::orb(src_v, dst_v);
            // .store() 将结果写回内存
            res_v.store((uint8_t*)dst + c);
        }
        // 尾部剩余元素处理（标量）
        // 处理无法被 SIMD 向量整除的尾部像素（即最后几个不足一个向量宽度的元素）。
        for(; c<width; c++)
            dst[c] |= src[c];

        // Advance to next row
        src += src_stride;
        dst += dst_stride;
    }
}

static void spread(const Mat &src, Mat &dst, int T)
{
    // 确保输入是 CV_8U 单通道
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(T > 0);

    // Allocate and zero-initialize spread (OR'ed) image
    dst = Mat::zeros(src.size(), CV_8UC1);

    // Fill in spread gradient image (section 2.3)
    for (int r = 0; r < T; ++r)
    {
        for (int c = 0; c < T; ++c)
        {
			// Check if the sub-region is valid (not empty)
			if (src.rows - r <= 0 || src.cols - c <= 0) continue;

            const uchar* src_ptr = src.ptr<uchar>(r) + c;
            orUnaligned8u(src_ptr, static_cast<const int>(src.step1()), dst.ptr(),
                          static_cast<const int>(dst.step1()), src.cols - c, src.rows - r);
        }
    }
}

void IM_Spread(unsigned char* Src, unsigned char* Dest, int Width, int Height, int Stride, int T)
{
	//    利用了或运算的行和列分离特性
	memset(Dest, 0, Height * Stride);
    int BlockSize = 16;
    int Block = Width / BlockSize;
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char* LinePS = Src + Y * Stride;
		unsigned char* LinePD = Dest + Y * Stride;
		for (int J = 0; J < ((Y + T > Height) ? Height - Y : T); J++)
		{
			//    高度方向依次向下进行T次或运算
			for (int X = 0; X < Block * BlockSize; X += BlockSize)
			{
				__m128i SrcV = _mm_loadu_si128((__m128i*)(LinePS + X));
				__m128i DstV = _mm_loadu_si128((__m128i*)(LinePD + X));
				_mm_storeu_si128((__m128i*)(LinePD + X), _mm_or_si128(SrcV, DstV));
			}
			for (int X = Block * BlockSize; X < Width; X++)
			{
				LinePD[X] |= LinePS[X];
			}
			LinePS += Stride;        //    源数据向下移动，目标数据不动
		}
	}

	BlockSize = 16, Block = (Width - T) / BlockSize;

	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char* LinePD = Dest + Y * Stride;
		for (int X = 0; X < Block * BlockSize; X += BlockSize)
		{
			__m128i Value = _mm_setzero_si128();
			for (int I = 0; I < T; I++)
			{
				__m128i SrcV = _mm_loadu_si128((__m128i*)(LinePD + X + I));
				Value = _mm_or_si128(Value, SrcV);
			}
			//    这个读取和写入是没有重叠的，所以可以不利用中间内存
			_mm_storeu_si128((__m128i*)(LinePD + X), Value);
		}
		for (int X = Block * BlockSize; X < Width; X++)
		{
			int Value = 0;
			for (int I = 0; I < T && (X + I) < Width; I++)
				{
					Value = Value | LinePD[X + I];
				}
			LinePD[X] = Value;
		}
	}
}

static const unsigned char LUT3 = 3;
// 1,2-->0 3-->LUT3
CV_DECL_ALIGNED(16)
static const unsigned char SIMILARITY_LUT[256] = {0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4};

inline __m128i _mm_extract_low4_epu8(__m128i v)
{
	return _mm_and_si128(v, _mm_set1_epi8(0x0F));
}

inline __m128i _mm_extract_high4_epu8(__m128i v)
{
	v = _mm_and_si128(v, _mm_set1_epi8(0xF0));
	v = _mm_srli_epi16(v, 4);
	return v;
}

// 使用辅助函数的版本
void process_row_simd_v2(const uchar* src_r, uchar* lsb4_r, uchar* msb4_r, int width)
{
	int c = 0;

	// 处理 16 字节对齐的部分
	for (; c <= width - 16; c += 16)
	{
		__m128i src_data = _mm_loadu_si128((__m128i*)(src_r + c));

		__m128i low4 = _mm_extract_low4_epu8(src_data);
		__m128i high4 = _mm_extract_high4_epu8(src_data);

		_mm_storeu_si128((__m128i*)(lsb4_r + c), low4);
		_mm_storeu_si128((__m128i*)(msb4_r + c), high4);
	}

	// 处理剩余像素
	for (; c < width; ++c)
	{
		lsb4_r[c] = src_r[c] & 15;
		msb4_r[c] = (src_r[c] & 240) >> 4;
	}
}

void computeOneResponseMap(const Mat& src, Mat& lsb4, Mat& msb4, std::vector<Mat>& response_maps, int startMapIndex, int thread_stride)
{
		uchar * lsb4_data = lsb4.ptr<uchar>();
		uchar* msb4_data = msb4.ptr<uchar>();

		bool no_max = false;
		bool no_shuff = false;

#ifdef has_max_int8_t
		no_max = false;
#endif

#ifdef has_shuff_int8_t
		no_shuff = false;
#endif
		// LUT is designed for 128 bits SIMD, so quite triky for others

		// For each of the 8 quantized orientations...
		for (int ori = startMapIndex; ori < (startMapIndex + thread_stride) && ori < 8; ++ori)
		{
			uchar* map_data = response_maps[ori].ptr<uchar>();
			const uchar* lut_low = SIMILARITY_LUT + 32 * ori;

			if (mipp::N<uint8_t>() == 1 || no_max || no_shuff)
			{ // no SIMD
				for (int i = 0; i < src.rows * src.cols; ++i)
					map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
			}
			else if (mipp::N<uint8_t>() == 16)
			{ // 128 SIMD, no add base

				const uchar* lut_low = SIMILARITY_LUT + 32 * ori;
				mipp::Reg<uint8_t> lut_low_v((uint8_t*)lut_low);
				mipp::Reg<uint8_t> lut_high_v((uint8_t*)lut_low + 16);

				for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>()) {
					mipp::Reg<uint8_t> low_mask((uint8_t*)lsb4_data + i);
					mipp::Reg<uint8_t> high_mask((uint8_t*)msb4_data + i);

					mipp::Reg<uint8_t> low_res = mipp::shuff(lut_low_v, low_mask);
					mipp::Reg<uint8_t> high_res = mipp::shuff(lut_high_v, high_mask);

					mipp::Reg<uint8_t> result = mipp::max(low_res, high_res);
					result.store((uint8_t*)map_data + i);
				}
			}
			else if (mipp::N<uint8_t>() == 16 || mipp::N<uint8_t>() == 32
					|| mipp::N<uint8_t>() == 64)
			{ //128 256 512 SIMD
				CV_Assert((src.rows * src.cols) % mipp::N<uint8_t>() == 0);

				uint8_t lut_temp[mipp::N<uint8_t>()] = {0};

				for (int slice = 0; slice < mipp::N<uint8_t>() / 16; slice++)
				{
					std::copy_n(lut_low, 16, lut_temp + slice * 16);
				}
				mipp::Reg<uint8_t> lut_low_v(lut_temp);

				uint8_t base_add_array[mipp::N<uint8_t>()] = {0};
				for (uint8_t slice = 0; slice < mipp::N<uint8_t>(); slice += 16)
				{
					std::copy_n(lut_low + 16, 16, lut_temp + slice);
					std::fill_n(base_add_array + slice, 16, slice);
				}
				mipp::Reg<uint8_t> base_add(base_add_array);
				mipp::Reg<uint8_t> lut_high_v(lut_temp);

				for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>())
				{
					mipp::Reg<uint8_t> mask_low_v((uint8_t*)lsb4_data + i);
					mipp::Reg<uint8_t> mask_high_v((uint8_t*)msb4_data + i);

					mask_low_v += base_add;
					mask_high_v += base_add;

					mipp::Reg<uint8_t> shuff_low_result = mipp::shuff(lut_low_v, mask_low_v);
					mipp::Reg<uint8_t> shuff_high_result = mipp::shuff(lut_high_v, mask_high_v);

					mipp::Reg<uint8_t> result = mipp::max(shuff_low_result, shuff_high_result);
					result.store((uint8_t*)map_data + i);
				}
			}
			else
			{
				for (int i = 0; i < src.rows * src.cols; ++i)
					map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
			}
		}
}

void computeResponseMap_async(const Mat& src, Mat& lsb4, Mat& msb4, std::vector<Mat>& response_maps, int mapNum)
{
	//Run                                         
	// Height >> 4,右移4位，相当于除以 16（2^4 = 16）,含义：图像高度分成多少个16像素高度的块. 为什么是16？因为，高度优化：16像素块大小通常是SIMD优化的友好单位（如SSE/AVX）
	//std::thread::hardware_concurrency()--返回系统真正支持的硬件并发线程数（CPU核心数或超线程数）,例如：4核8线程的CPU → 返回 8
	const int32_t hw_concur = std::min(100/*Height >> 4*/, static_cast<int32_t>(std::thread::hardware_concurrency()));

	std::vector<std::future<void>> fut(hw_concur);

	const int thread_stride = (mapNum - 1) / hw_concur + 1;
	int i = 0, start = 0;

	for (; i < std::min(mapNum, hw_concur); i++, start += thread_stride)
	{
		fut[i] = std::async(std::launch::async, [=, &src, &lsb4, &msb4, &response_maps]() {
            computeOneResponseMap(src, lsb4, msb4, response_maps, start, thread_stride);
			});
	}
	for (int j = 0; j < i; ++j)
	{
		if (fut[j].valid())
		{
			fut[j].wait();
		}
	}
}

static void computeResponseMaps(const Mat &src, std::vector<Mat> &response_maps)
{
    CV_Assert((src.rows * src.cols) % 16 == 0);

    // Allocate response maps
    response_maps.resize(8);
    for (int i = 0; i < 8; ++i)
        response_maps[i].create(src.size(), CV_8U);

    Mat lsb4(src.size(), CV_8U);
    Mat msb4(src.size(), CV_8U);

    for (int r = 0; r < src.rows; ++r)
    {
        const uchar *src_r = src.ptr(r);
        uchar *lsb4_r = lsb4.ptr(r);
        uchar *msb4_r = msb4.ptr(r);

        //for (int c = 0; c < src.cols; ++c)
        //{
        //    // Least significant 4 bits of spread image pixel
        //    lsb4_r[c] = src_r[c] & 15;
        //    // Most significant 4 bits, right-shifted to be in [0, 16)
        //    msb4_r[c] = (src_r[c] & 240) >> 4;
        //}

		int c = 0;
         
 #ifdef __AVX2__
          // AVX2: 一次处理32个像素
		for (; c <= src.cols - 32; c += 32)
		{
			__m256i src_data = _mm256_loadu_si256((__m256i*)(src_r + c));

			// 低4位
			__m256i low4 = _mm256_and_si256(src_data, _mm256_set1_epi8(0x0F));
			// 高4位
			__m256i high4 = _mm256_and_si256(src_data, _mm256_set1_epi8(0xF0));
			high4 = _mm256_srli_epi16(high4, 4);

			_mm256_storeu_si256((__m256i*)(lsb4_r + c), low4);
			_mm256_storeu_si256((__m256i*)(msb4_r + c), high4);
		}

		// SSE: 一次处理16个像素
		for (; c <= src.cols - 16; c += 16)
		{
			__m128i src_data = _mm_loadu_si128((__m128i*)(src_r + c));

			__m128i low4 = _mm_and_si128(src_data, _mm_set1_epi8(0x0F));
			__m128i high4 = _mm_and_si128(src_data, _mm_set1_epi8(0xF0));
			high4 = _mm_srli_epi16(high4, 4);

			_mm_storeu_si128((__m128i*)(lsb4_r + c), low4);
			_mm_storeu_si128((__m128i*)(msb4_r + c), high4);
		}
#endif // __AVX2__

		// 剩余像素用标量处理
		for (; c < src.cols; ++c)
		{
			lsb4_r[c] = src_r[c] & 15;
			msb4_r[c] = (src_r[c] & 240) >> 4;
		}
    }

    computeResponseMap_async(src, lsb4, msb4, response_maps, 8);

    /*
    {
        uchar *lsb4_data = lsb4.ptr<uchar>();
        uchar *msb4_data = msb4.ptr<uchar>();

        bool no_max = false;
        bool no_shuff = false;

#ifdef has_max_int8_t
        no_max = false;
#endif

#ifdef has_shuff_int8_t
        no_shuff = false;
#endif
        // LUT is designed for 128 bits SIMD, so quite triky for others

        // For each of the 8 quantized orientations...
        for (int ori = 0; ori < 8; ++ori)
        {
            uchar *map_data = response_maps[ori].ptr<uchar>();
            const uchar *lut_low = SIMILARITY_LUT + 32 * ori;

            if(mipp::N<uint8_t>() == 1 || no_max || no_shuff)
            { // no SIMD
                for (int i = 0; i < src.rows * src.cols; ++i)
                    map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
            }
            else if(mipp::N<uint8_t>() == 16)
            { // 128 SIMD, no add base

                const uchar *lut_low = SIMILARITY_LUT + 32 * ori;
                mipp::Reg<uint8_t> lut_low_v((uint8_t*)lut_low);
                mipp::Reg<uint8_t> lut_high_v((uint8_t*)lut_low + 16);

                for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>()){
                    mipp::Reg<uint8_t> low_mask((uint8_t*)lsb4_data + i);
                    mipp::Reg<uint8_t> high_mask((uint8_t*)msb4_data + i);

                    mipp::Reg<uint8_t> low_res = mipp::shuff(lut_low_v, low_mask);
                    mipp::Reg<uint8_t> high_res = mipp::shuff(lut_high_v, high_mask);

                    mipp::Reg<uint8_t> result = mipp::max(low_res, high_res);
                    result.store((uint8_t*)map_data + i);
                }
            }
            else if(mipp::N<uint8_t>() == 16 || mipp::N<uint8_t>() == 32
                    || mipp::N<uint8_t>() == 64)
            { //128 256 512 SIMD
                CV_Assert((src.rows * src.cols) % mipp::N<uint8_t>() == 0);

                uint8_t lut_temp[mipp::N<uint8_t>()] = {0};

                for(int slice=0; slice<mipp::N<uint8_t>()/16; slice++)
                {
                    std::copy_n(lut_low, 16, lut_temp+slice*16);
                }
                mipp::Reg<uint8_t> lut_low_v(lut_temp);

                uint8_t base_add_array[mipp::N<uint8_t>()] = {0};
                for(uint8_t slice=0; slice<mipp::N<uint8_t>(); slice+=16)
                {
                    std::copy_n(lut_low+16, 16, lut_temp+slice);
                    std::fill_n(base_add_array+slice, 16, slice);
                }
                mipp::Reg<uint8_t> base_add(base_add_array);
                mipp::Reg<uint8_t> lut_high_v(lut_temp);

                for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>())
                {
                    mipp::Reg<uint8_t> mask_low_v((uint8_t*)lsb4_data+i);
                    mipp::Reg<uint8_t> mask_high_v((uint8_t*)msb4_data+i);

                    mask_low_v += base_add;
                    mask_high_v += base_add;

                    mipp::Reg<uint8_t> shuff_low_result = mipp::shuff(lut_low_v, mask_low_v);
                    mipp::Reg<uint8_t> shuff_high_result = mipp::shuff(lut_high_v, mask_high_v);

                    mipp::Reg<uint8_t> result = mipp::max(shuff_low_result, shuff_high_result);
                    result.store((uint8_t*)map_data + i);
                }
            }
            else
            {
                for (int i = 0; i < src.rows * src.cols; ++i)
                    map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
            }
        }


    }*/
}

static void linearize(const Mat &response_map, Mat &linearized, int T)
{
    CV_Assert(response_map.rows % T == 0);
    CV_Assert(response_map.cols % T == 0);

    // linearized has T^2 rows, where each row is a linear memory
    int mem_width = response_map.cols / T;
    int mem_height = response_map.rows / T;
    linearized.create(T * T, mem_width * mem_height, CV_8U);

    // Outer two for loops iterate over top-left T^2 starting pixels
    int index = 0;
    for (int r_start = 0; r_start < T; ++r_start)
    {
        for (int c_start = 0; c_start < T; ++c_start)
        {
            uchar *memory = linearized.ptr(index);
            ++index;

            // Inner two loops copy every T-th pixel into the linear memory
            for (int r = r_start; r < response_map.rows; r += T)
            {
                const uchar *response_data = response_map.ptr(r);
                for (int c = c_start; c < response_map.cols; c += T)
                    *memory++ = response_data[c];
            }
        }
    }
}

static void linearize_mipp(const Mat& response_map, Mat& linearized, int T)
{
	CV_Assert(response_map.rows % T == 0);
	CV_Assert(response_map.cols % T == 0);

	int mem_width = response_map.cols / T;
	int mem_height = response_map.rows / T;
	linearized.create(T * T, mem_width * mem_height, CV_8U);

	const int vec_size = mipp::N<uint8_t>();      // 向量寄存器可容纳的uint8_t数量 (AVX2下为32)
	uchar* dst_data = linearized.ptr<uchar>();
	const uchar* src_data = response_map.ptr<uchar>();
	const int src_stride = response_map.cols;      // 源图像步长（字节）

	// 为 T*T 个输出通道预先计算起始指针
	std::vector<uchar*> dst_ptrs(T * T);
	for (int i = 0; i < T * T; ++i) {
		dst_ptrs[i] = dst_data + i * (mem_width * mem_height);
	}

	// 核心优化：按宏块处理，每个宏块一次处理一个向量的像素
	for (int macro_r = 0; macro_r < mem_height; ++macro_r) {
		for (int macro_c = 0; macro_c < mem_width; ++macro_c) {
			// 处理当前 T x T 宏块
			for (int dj = 0; dj < T; ++dj) {
				const uchar* src_row = src_data + (macro_r * T + dj) * src_stride + macro_c * T;

				// 使用MIPP向量化处理宏块内的一行
				int di = 0;
				// 向量化部分：一次处理 vec_size 个像素
				for (; di <= T - vec_size; di += vec_size) {
					// 关键：使用 gather 从 src_row[di] 开始，后续为步长 1 加载（实际为连续加载）
					// 由于宏块内 src_row[di..di+vec_size-1] 在内存中是连续的，
					// 这里直接使用 load 即可，因为 T x T 宏块内的数据本身就是连续存储的
					mipp::Reg<uint8_t> src_reg = mipp::load<uint8_t>(src_row + di);

					// 将向量中的数据分散写入对应的 T 个输出通道
					for (int i = 0; i < vec_size; ++i) {
						int channel_id = dj * T + (di + i);
						int dst_offset = macro_r * mem_width + macro_c;
						dst_ptrs[channel_id][dst_offset] = src_reg[i];
					}
				}
				// 标量处理尾部（不足 vec_size 的部分）
				for (; di < T; ++di) {
					int channel_id = dj * T + di;
					int dst_offset = macro_r * mem_width + macro_c;
					dst_ptrs[channel_id][dst_offset] = src_row[di];
				}
			}
		}
	}
}
/****************************************************************************************\
*                                                             Linearized similarities                                                                    *
\****************************************************************************************/

static const unsigned char *accessLinearMemory(const std::vector<Mat> &linear_memories,
                                               const Feature &f, int T, int W)
{
    // Retrieve the TxT grid of linear memories associated with the feature label
    const Mat &memory_grid = linear_memories[f.label];
    CV_DbgAssert(memory_grid.rows == T * T);
    CV_DbgAssert(f.x >= 0);
    CV_DbgAssert(f.y >= 0);
    // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
    int grid_x = f.x % T;
    int grid_y = f.y % T;
    int grid_index = grid_y * T + grid_x;
    CV_DbgAssert(grid_index >= 0);
    CV_DbgAssert(grid_index < memory_grid.rows);
    const unsigned char *memory = memory_grid.ptr(grid_index);
    // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
    // input image width decimated by T.
    int lm_x = f.x / T;
    int lm_y = f.y / T;
    int lm_index = lm_y * W + lm_x;
    CV_DbgAssert(lm_index >= 0);
    CV_DbgAssert(lm_index < memory_grid.cols);
    return memory + lm_index;
}

static void similarity(const std::vector<Mat> &linear_memories, const Template &templ,
                       Mat &dst, Size size, int T)
{
    // we only have one modality, so 8192*2, due to mipp, back to 8192
    CV_Assert(templ.features.size() < 8192);

    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    int template_positions = span_y * W + span_x + 1; // why add 1?

    dst = Mat::zeros(H, W, CV_16U);
    short *dst_ptr = dst.ptr<short>();
    mipp::Reg<uint8_t> zero_v(uint8_t(0));

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {

        Feature f = templ.features[i];

        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        int j = 0;

        // *2 to avoid int8 read out of range
        for(; j <= template_positions -mipp::N<int16_t>()*2; j+=mipp::N<int16_t>()){
            mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + j);

            // uchar to short, once for N bytes
            mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

            mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + j);

            mipp::Reg<int16_t> res_v = src16_v + dst_v;
            res_v.store((int16_t*)dst_ptr + j);
        }

        for(; j<template_positions; j++)
            dst_ptr[j] += short(lm_ptr[j]);
    }
}

static void similarityLocal(const std::vector<Mat> &linear_memories, const Template &templ,
                            Mat &dst, Size size, int T, Point center)
{
    CV_Assert(templ.features.size() < 8192);

    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_16U);

    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;
    mipp::Reg<uint8_t> zero_v = uint8_t(0);

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);
        {
            short *dst_ptr = dst.ptr<short>();

            if(mipp::N<uint8_t>() > 32){ //512 bits SIMD
                for (int row = 0; row < 16; row += mipp::N<int16_t>()/16){
                    mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + row*16);

                    // load lm_ptr, 16 bytes once, for half
                    uint8_t local_v[mipp::N<uint8_t>()] = {0};
                    for(int slice=0; slice<mipp::N<uint8_t>()/16/2; slice++){
                        std::copy_n(lm_ptr, 16, &local_v[16*slice]);
                        lm_ptr += W;
                    }
                    mipp::Reg<uint8_t> src8_v(local_v);
                    // uchar to short, once for N bytes
                    mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

                    mipp::Reg<int16_t> res_v = src16_v + dst_v;
                    res_v.store((int16_t*)dst_ptr);

                    dst_ptr += mipp::N<int16_t>();
                }
            }else{ // 256 128 or no SIMD
                for (int row = 0; row < 16; ++row){
                    for(int col=0; col<16; col+=mipp::N<int16_t>()){
                        mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + col);

                        // uchar to short, once for N bytes
                        mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

                        mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + col);
                        mipp::Reg<int16_t> res_v = src16_v + dst_v;
                        res_v.store((int16_t*)dst_ptr + col);
                    }
                    dst_ptr += 16;
                    lm_ptr += W;
                }
            }
        }
    }
}

static void similarity_64(const std::vector<Mat> &linear_memories, const Template &templ,
                          Mat &dst, Size size, int T)
{
    // 63 features or less is a special case because the max similarity per-feature is 4.
    // 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
    // about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas a more
    // general function would use _mm_add_epi16.
    CV_Assert(templ.features.size() < 64);
    /// @todo Handle more than 255/MAX_RESPONSE features!!

    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    // Compute number of contiguous (in memory) pixels to check when sliding feature over
    // image. This allows template to wrap around left/right border incorrectly, so any
    // wrapped template matches must be filtered out!
    int template_positions = span_y * W + span_x + 1; // why add 1?
    //int template_positions = (span_y - 1) * W + span_x; // More correct?

    /// @todo In old code, dst is buffer of size m_U. Could make it something like
    /// (span_x)x(span_y) instead?
    dst = Mat::zeros(H, W, CV_8U);
    uchar *dst_ptr = dst.ptr<uchar>();

    // Compute the similarity measure for this template by accumulating the contribution of
    // each feature
    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        // Add the linear memory at the appropriate offset computed from the location of
        // the feature in the template
        Feature f = templ.features[i];
        // Discard feature if out of bounds
        /// @todo Shouldn't actually see x or y < 0 here?
        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
        int j = 0;

        for(; j <= template_positions -mipp::N<uint8_t>(); j+=mipp::N<uint8_t>()){
            mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + j);
            mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + j);

            mipp::Reg<uint8_t> res_v = src_v + dst_v;
            res_v.store((uint8_t*)dst_ptr + j);
        }

        for(; j<template_positions; j++)
            dst_ptr[j] += lm_ptr[j];
    }
}

static void similarityLocal_64(const std::vector<Mat> &linear_memories, const Template &templ,
                               Mat &dst, Size size, int T, Point center)
{
    // Similar to whole-image similarity() above. This version takes a position 'center'
    // and computes the energy in the 16x16 patch centered on it.
    CV_Assert(templ.features.size() < 64);

    // Compute the similarity map in a 16x16 patch around center
    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_8U);

    // Offset each feature point by the requested center. Further adjust to (-8,-8) from the
    // center to get the top-left corner of the 16x16 patch.
    // NOTE: We make the offsets multiples of T to agree with results of the original code.
    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        {
            uchar *dst_ptr = dst.ptr<uchar>();

            if(mipp::N<uint8_t>() > 16){ // 256 or 512 bits SIMD
                for (int row = 0; row < 16; row += mipp::N<uint8_t>()/16){
                    mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr);

                    // load lm_ptr, 16 bytes once
                    uint8_t local_v[mipp::N<uint8_t>()];
                    for(int slice=0; slice<mipp::N<uint8_t>()/16; slice++){
                        std::copy_n(lm_ptr, 16, &local_v[16*slice]);
                        lm_ptr += W;
                    }
                    mipp::Reg<uint8_t> src_v(local_v);

                    mipp::Reg<uint8_t> res_v = src_v + dst_v;
                    res_v.store((uint8_t*)dst_ptr);

                    dst_ptr += mipp::N<uint8_t>();
                }
            }else{ // 128 or no SIMD
                for (int row = 0; row < 16; ++row){
                    for(int col=0; col<16; col+=mipp::N<uint8_t>()){
                        mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + col);
                        mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + col);
                        mipp::Reg<uint8_t> res_v = src_v + dst_v;
                        res_v.store((uint8_t*)dst_ptr + col);
                    }
                    dst_ptr += 16;
                    lm_ptr += W;
                }
            }
        }
    }
}

/****************************************************************************************\
*                                                             High-level Detector API                                                                    *
\****************************************************************************************/

Detector::Detector()
{
    this->modality = makePtr<ColorGradient>();
    pyramid_levels = 2;
    T_at_level.push_back(4);
    T_at_level.push_back(8);
}

Detector::Detector(std::vector<int> T)
{
    this->modality = makePtr<ColorGradient>();
    pyramid_levels = T.size();
    T_at_level = T;
}

Detector::Detector(int num_features, std::vector<int> T, float weak_thresh, float strong_threash)
{
    this->modality = makePtr<ColorGradient>(weak_thresh, num_features, strong_threash);
    pyramid_levels = T.size();
    T_at_level = T;
}

std::vector<Match> Detector::match(Mat source, float threshold,
                                   const std::vector<std::string> &class_ids, const Mat mask) const
{
    Timer timer;
    std::vector<Match> matches;

    // Initialize each ColorGradient with our sources
    std::vector<Ptr<ColorGradientPyramid>> quantizers;
    CV_Assert(mask.empty() || mask.size() == source.size());
    quantizers.push_back(modality->process(source, mask));

    // pyramid level -> ColorGradient -> quantization
    LinearMemoryPyramid lm_pyramid(pyramid_levels,
                                   std::vector<LinearMemories>(1, LinearMemories(8)));

    // For each pyramid level, precompute linear memories for each ColorGradient
    std::vector<Size> sizes;
    for (int l = 0; l < pyramid_levels; ++l)
    {
        int T = T_at_level[l];
        std::vector<LinearMemories> &lm_level = lm_pyramid[l];

        if (l > 0)
        {
            for (int i = 0; i < (int)quantizers.size(); ++i)
                quantizers[i]->pyrDown();
        }

        Mat quantized, spread_quantized;
        std::vector<Mat> response_maps;
        for (int i = 0; i < (int)quantizers.size(); ++i)
        {
            quantizers[i]->quantize(quantized);
            //spread(quantized, spread_quantized, T);
            spread_quantized = Mat::zeros(quantized.size(), CV_8UC1);
            IM_Spread(quantized.ptr(), spread_quantized.ptr(), quantized.cols, quantized.rows, static_cast<const int>(quantized.step1()), T);
            computeResponseMaps(spread_quantized, response_maps);

            LinearMemories &memories = lm_level[i];
            for (int j = 0; j < 8; ++j)
                linearize(response_maps[j], memories[j], T);
                //linearize_mipp(response_maps[j], memories[j], T);
        }

        sizes.push_back(quantized.size());
    }

    timer.out("construct response map");

    if (class_ids.empty())
    {
        // Match all templates
        TemplatesMap::const_iterator it = class_templates.begin(), itend = class_templates.end();
        for (; it != itend; ++it)
            matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
    }
    else
    {
        // Match only templates for the requested class IDs
        for (int i = 0; i < (int)class_ids.size(); ++i)
        {
            TemplatesMap::const_iterator it = class_templates.find(class_ids[i]);
            if (it != class_templates.end())
                matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
        }
    }

    // Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
    std::sort(matches.begin(), matches.end());
    std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
    matches.erase(new_end, matches.end());

    timer.out("templ match");

    return matches;
}

// Used to filter out weak matches
struct MatchPredicate
{
    MatchPredicate(float _threshold) : threshold(_threshold) {}
    bool operator()(const Match &m) { return m.similarity < threshold; }
    float threshold;
};

void Detector::matchClass(const LinearMemoryPyramid &lm_pyramid,
                          const std::vector<Size> &sizes,
                          float threshold, std::vector<Match> &matches,
                          const std::string &class_id,
                          const std::vector<TemplatePyramid> &template_pyramids) const
{
#pragma omp declare reduction \
    (omp_insert: std::vector<Match>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

#pragma omp parallel for reduction(omp_insert:matches)
    for (size_t template_id = 0; template_id < template_pyramids.size(); ++template_id)
    {
        const TemplatePyramid &tp = template_pyramids[template_id];
        // First match over the whole image at the lowest pyramid level
        /// @todo Factor this out into separate function
        const std::vector<LinearMemories> &lowest_lm = lm_pyramid.back();

        std::vector<Match> candidates;
        {
            // Compute similarity maps for each ColorGradient at lowest pyramid level
            Mat similarities;
            int lowest_start = static_cast<int>(tp.size() - 1);
            int lowest_T = T_at_level.back();
            int num_features = 0;

            {
                const Template &templ = tp[lowest_start];
                num_features += static_cast<int>(templ.features.size());

                if (templ.features.size() < 64){
                    similarity_64(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
                    similarities.convertTo(similarities, CV_16U);
                }else if (templ.features.size() < 8192){
                    similarity(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
                }else{
                    CV_Error(Error::StsBadArg, "feature size too large");
                }
            }

            // Find initial matches
            for (int r = 0; r < similarities.rows; ++r)
            {
                ushort *row = similarities.ptr<ushort>(r);
                for (int c = 0; c < similarities.cols; ++c)
                {
                    int raw_score = row[c];
                    float score = (raw_score * 100.f) / (4 * num_features);

                    if (score > threshold)
                    {
                        int offset = lowest_T / 2 + (lowest_T % 2 - 1);
                        int x = c * lowest_T + offset;
                        int y = r * lowest_T + offset;
                        candidates.push_back(Match(x, y, score, class_id, static_cast<int>(template_id)));
                    }
                }
            }
        }


        // Locally refine each match by marching up the pyramid
        for (int l = pyramid_levels - 2; l >= 0; --l)
        {
            const std::vector<LinearMemories> &lms = lm_pyramid[l];
            int T = T_at_level[l];
            int start = static_cast<int>(l);
            Size size = sizes[l];
            int border = 8 * T;
            int offset = T / 2 + (T % 2 - 1);
            int max_x = size.width - tp[start].width - border;
            int max_y = size.height - tp[start].height - border;

            Mat similarities2;
            for (int m = 0; m < (int)candidates.size(); ++m)
            {
                Match &match2 = candidates[m];
                int x = match2.x * 2 + 1; /// @todo Support other pyramid distance
                int y = match2.y * 2 + 1;

                // Require 8 (reduced) row/cols to the up/left
                x = std::max(x, border);
                y = std::max(y, border);

                // Require 8 (reduced) row/cols to the down/left, plus the template size
                x = std::min(x, max_x);
                y = std::min(y, max_y);

                // Compute local similarity maps for each ColorGradient
                int numFeatures = 0;

                {
                    const Template &templ = tp[start];
                    numFeatures += static_cast<int>(templ.features.size());

                    if (templ.features.size() < 64){
                        similarityLocal_64(lms[0], templ, similarities2, size, T, Point(x, y));
                        similarities2.convertTo(similarities2, CV_16U);
                    }else if (templ.features.size() < 8192){
                        similarityLocal(lms[0], templ, similarities2, size, T, Point(x, y));
                    }else{
                        CV_Error(Error::StsBadArg, "feature size too large");
                    }
                }

                // Find best local adjustment
                float best_score = 0;
                int best_r = -1, best_c = -1;
                for (int r = 0; r < similarities2.rows; ++r)
                {
                    ushort *row = similarities2.ptr<ushort>(r);
                    for (int c = 0; c < similarities2.cols; ++c)
                    {
                        int score_int = row[c];
                        float score = (score_int * 100.f) / (4 * numFeatures);

                        if (score > best_score)
                        {
                            best_score = score;
                            best_r = r;
                            best_c = c;
                        }
                    }
                }
                // Update current match
                match2.similarity = best_score;
                match2.x = (x / T - 8 + best_c) * T + offset;
                match2.y = (y / T - 8 + best_r) * T + offset;
            }

            // Filter out any matches that drop below the similarity threshold
            std::vector<Match>::iterator new_end = std::remove_if(candidates.begin(), candidates.end(),
                                                                  MatchPredicate(threshold));
            candidates.erase(new_end, candidates.end());
        }

        matches.insert(matches.end(), candidates.begin(), candidates.end());
    }
}

int Detector::addTemplate(const Mat source, const std::string &class_id,
                          const Mat &object_mask, int num_features)
{
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());

    TemplatePyramid tp;
    tp.resize(pyramid_levels);

    {
        // Extract a template at each pyramid level
        Ptr<ColorGradientPyramid> qp = modality->process(source, object_mask);

        if(num_features > 0)
        qp->num_features = num_features;

        for (int l = 0; l < pyramid_levels; ++l)
        {
            /// @todo Could do mask subsampling here instead of in pyrDown()
            if (l > 0)
                qp->pyrDown();

            bool success = qp->extractTemplate(tp[l]);
            if (!success)
                return -1;
        }
    }

    //    Rect bb =
    cropTemplates(tp);

    /// @todo Can probably avoid a copy of tp here with swap
    template_pyramids.push_back(tp);
    return template_id;
}

static cv::Point2f rotate2d(const cv::Point2f inPoint, const double angRad)
{
    cv::Point2f outPoint;
    //CW rotation
    outPoint.x = std::cos(angRad)*inPoint.x - std::sin(angRad)*inPoint.y;
    outPoint.y = std::sin(angRad)*inPoint.x + std::cos(angRad)*inPoint.y;
    return outPoint;
}

static cv::Point2f rotatePoint(const cv::Point2f inPoint, const cv::Point2f center, const double angRad)
{
    return rotate2d(inPoint - center, angRad) + center;
}

int Detector::addTemplate_rotate(const string &class_id, int zero_id,
                                 float theta, cv::Point2f center)
{
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());

    const auto& to_rotate_tp = template_pyramids[zero_id];

    TemplatePyramid tp;
    tp.resize(pyramid_levels);

    for (int l = 0; l < pyramid_levels; ++l)
    {
        if(l>0) center /= 2;

        for(auto& f: to_rotate_tp[l].features){
            Point2f p;
            p.x = f.x + to_rotate_tp[l].tl_x;
            p.y = f.y + to_rotate_tp[l].tl_y;
            Point2f p_rot = rotatePoint(p, center, -theta/180*CV_PI);

            Feature f_new;
            f_new.x = int(p_rot.x + 0.5f);
            f_new.y = int(p_rot.y + 0.5f);

            f_new.theta = f.theta - theta;
            while(f_new.theta > 360) f_new.theta -= 360;
            while(f_new.theta < 0) f_new.theta += 360;

            f_new.label = int(f_new.theta * 16 / 360 + 0.5f);
            f_new.label &= 7;


            tp[l].features.push_back(f_new);
        }
        tp[l].pyramid_level = l;
    }

    cropTemplates(tp);

    template_pyramids.push_back(tp);
    return template_id;
}
const std::vector<Template> &Detector::getTemplates(const std::string &class_id, int template_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    CV_Assert(i != class_templates.end());
    CV_Assert(i->second.size() > size_t(template_id));
    return i->second[template_id];
}

int Detector::numTemplates() const
{
    int ret = 0;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
        ret += static_cast<int>(i->second.size());
    return ret;
}

int Detector::numTemplates(const std::string &class_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    if (i == class_templates.end())
        return 0;
    return static_cast<int>(i->second.size());
}

std::vector<std::string> Detector::classIds() const
{
    std::vector<std::string> ids;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
    {
        ids.push_back(i->first);
    }

    return ids;
}

void Detector::read(const FileNode &fn)
{
    class_templates.clear();
    pyramid_levels = fn["pyramid_levels"];
    fn["T"] >> T_at_level;

    modality = makePtr<ColorGradient>();
}

void Detector::write(FileStorage &fs) const
{
    fs << "pyramid_levels" << pyramid_levels;
    fs << "T" << T_at_level;

    modality->write(fs);
}

std::string Detector::readClass(const FileNode &fn, const std::string &class_id_override)
{
    // Detector should not already have this class
    String class_id;
    if (class_id_override.empty())
    {
        String class_id_tmp = fn["class_id"];
        CV_Assert(class_templates.find(class_id_tmp) == class_templates.end());
        class_id = class_id_tmp;
    }
    else
    {
        class_id = class_id_override;
    }

    TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
    std::vector<TemplatePyramid> &tps = v.second;
    int expected_id = 0;

    FileNode tps_fn = fn["template_pyramids"];
    tps.resize(tps_fn.size());
    FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
    for (; tps_it != tps_it_end; ++tps_it, ++expected_id)
    {
        int template_id = (*tps_it)["template_id"];
        CV_Assert(template_id == expected_id);
        FileNode templates_fn = (*tps_it)["templates"];
        tps[template_id].resize(templates_fn.size());

        FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
        int idx = 0;
        for (; templ_it != templ_it_end; ++templ_it)
        {
            tps[template_id][idx++].read(*templ_it);
        }
    }

    class_templates.insert(v);
    return class_id;
}

void Detector::writeClass(const std::string &class_id, FileStorage &fs) const
{
    TemplatesMap::const_iterator it = class_templates.find(class_id);
    CV_Assert(it != class_templates.end());
    const std::vector<TemplatePyramid> &tps = it->second;

    fs << "class_id" << it->first;
    fs << "pyramid_levels" << pyramid_levels;
    fs << "template_pyramids"
       << "[";
    for (size_t i = 0; i < tps.size(); ++i)
    {
        const TemplatePyramid &tp = tps[i];
        fs << "{";
        fs << "template_id" << int(i); //TODO is this cast correct? won't be good if rolls over...
        fs << "templates"
           << "[";
        for (size_t j = 0; j < tp.size(); ++j)
        {
            fs << "{";
            tp[j].write(fs);
            fs << "}"; // current template
        }
        fs << "]"; // templates
        fs << "}"; // current pyramid
    }
    fs << "]"; // pyramids
}

void Detector::readClasses(const std::vector<std::string> &class_ids,
                           const std::string &format)
{
    for (size_t i = 0; i < class_ids.size(); ++i)
    {
        const String &class_id = class_ids[i];
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::READ);
        readClass(fs.root());
    }
}

void Detector::writeClasses(const std::string &format) const
{
    TemplatesMap::const_iterator it = class_templates.begin(), it_end = class_templates.end();
    for (; it != it_end; ++it)
    {
        const String &class_id = it->first;
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::WRITE);
        writeClass(class_id, fs);
    }
}

} // namespace line2Dup
