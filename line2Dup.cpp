#include "line2Dup.h"
#include <iostream>
#include <memory>
#include <cmath>
#include <limits>
#include "speedUp.h"
#include <pmmintrin.h>
#include <chrono>
#include <omp.h>

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_BitScanForward)

static inline int ctz_u16(uint16_t mask) {
	unsigned long index;
	if (_BitScanForward(&index, (unsigned long)mask)) {
		return (int)index;
	}
	return 0; // 当 mask == 0 时，返回 0（与 __builtin_ctz 未定义行为不同，这里安全）
}

static inline int ctz_u32(uint32_t mask) {
	unsigned long index;
	if (_BitScanForward(&index, (unsigned long)mask)) {
		return (int)index;
	}
	return 0;
}
#else
static inline int ctz_u16(uint16_t mask) {
	return __builtin_ctz((unsigned)mask);
}

static inline int ctz_u32(uint32_t mask) {
	return __builtin_ctz(mask);
}
#endif

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
    // Older yaml only stored x/y/label; theta defaults to 0 after load.
    if (fni != fn.end())
        fni >> theta;
    else
        theta = 0.0f;
}

void Feature::write(FileStorage &fs) const
{
    fs << "[:" << x << y << label << theta << "]";
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

void hysteresisGradientSpeedUp(Mat& magnitude, Mat& quantized_angle,
	Mat& angle, Mat& angleAuant, float threshold)
{
	// Quantize 360 degree range of orientations into 16 buckets
	// Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
	// for stability of horizontal and vertical features.
	Mat_<unsigned char> quantized_unfiltered;
	//angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);

	// Zero out top and bottom rows
	/// @todo is this necessary, or even correct?
	memset(angleAuant.ptr(), 0, angleAuant.cols);
	memset(angleAuant.ptr(angleAuant.rows - 1), 0, angleAuant.cols);
	// Zero out first and last columns
	for (int r = 0; r < angleAuant.rows; ++r)
	{
        angleAuant.ptr<uchar>(r)[0] = 0;
        angleAuant.ptr<uchar>(r)[angleAuant.cols - 1] = 0;
	}

	//// Mask 16 buckets into 8 quantized orientations
	//for (int r = 1; r < angle.rows - 1; ++r)
	//{
	//	uchar* quant_r = angleAuant.ptr<uchar>(r);
	//	for (int c = 1; c < angle.cols - 1; ++c)
	//	{
	//		quant_r[c] &= 7;
	//	}
	//}

	//// Filter the raw quantized image. Only accept pixels where the magnitude is above some
	//// threshold, and there is local agreement on the quantization.
	//quantized_angle = Mat::zeros(angle.size(), CV_8U);
	//for (int r = 1; r < angle.rows - 1; ++r)
	//{
	//	float* mag_r = magnitude.ptr<float>(r);

	//	for (int c = 1; c < angle.cols - 1; ++c)
	//	{
	//		if (mag_r[c] > threshold)
	//		{
	//			// Compute histogram of quantized bins in 3x3 patch around pixel
	//			int histogram[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
 //               
	//			uchar* patch3x3_row = &angleAuant.ptr<uchar>(r - 1)[c - 1];
	//			histogram[patch3x3_row[0]]++;
	//			histogram[patch3x3_row[1]]++;
	//			histogram[patch3x3_row[2]]++;

	//			patch3x3_row += angleAuant.step1();
	//			histogram[patch3x3_row[0]]++;
	//			histogram[patch3x3_row[1]]++;
	//			histogram[patch3x3_row[2]]++;

	//			patch3x3_row += angleAuant.step1();
	//			histogram[patch3x3_row[0]]++;
	//			histogram[patch3x3_row[1]]++;
	//			histogram[patch3x3_row[2]]++;

	//			// Find bin with the most votes from the patch
	//			int max_votes = 0;
	//			int index = -1;
	//			for (int i = 0; i < 8; ++i)
	//			{
	//				if (max_votes < histogram[i])
	//				{
	//					index = i;
	//					max_votes = histogram[i];
	//				}
	//			}

	//			// Only accept the quantization if majority of pixels in the patch agree
	//			static const int NEIGHBOR_THRESHOLD = 5;
	//			if (max_votes >= NEIGHBOR_THRESHOLD)
	//				quantized_angle.at<uchar>(r, c) = uchar(1 << index);
	//		}
	//	}
	//}
	quantized_angle.create(magnitude.size(), CV_8U);

	const int rows = magnitude.rows;
	const int cols = magnitude.cols;
    memset(quantized_angle.ptr<uchar>(0), 0, cols);
    memset(quantized_angle.ptr<uchar>(rows - 1), 0, cols);

#pragma omp parallel for schedule(static)
	for (int r = 1; r < rows - 1; ++r) 
    {
		uchar* q_row0 = angleAuant.ptr<uchar>(r - 1);
		uchar* q_row1 = angleAuant.ptr<uchar>(r);
		uchar* q_row2 = angleAuant.ptr<uchar>(r + 1);
		float* mag_row = magnitude.ptr<float>(r);
		uchar* out_row = quantized_angle.ptr<uchar>(r);

		// 初始化直方图（9个值）
		int h[8] = { 0 };
		for (int cc = 0; cc < 3; ++cc) 
        { 
            ++h[q_row0[cc]];
            ++h[q_row1[cc]];
            ++h[q_row2[cc]];
        }

        out_row[0] = 0;
        out_row[cols - 1] = 0;
		// 滑动窗口
		for (int c = 1; c < cols - 1; ++c) 
        {
            out_row[c] = 0;
			// 当前窗口中心为 (r,c)，直方图 h 对应列 [c-1, c+1]
			if (mag_row[c] > threshold) 
            {
				// 找最大值（展开比较）
				int max_votes = h[0], idx = 0;
				for (int i = 1; i < 8; ++i) 
                {
                    if (h[i] > max_votes) 
                    { 
                        max_votes = h[i]; idx = i; 
                    }
                }
				if (max_votes >= 5) out_row[c] = (uchar)(1 << idx);
			}

			// 滑动窗口：移除左侧列 (c-1)，加入右侧列 (c+2)
			if (c - 1 >= 0) 
            {
				int left = c - 1; // 其实 c-1 是即将被移除的列，但我们现在窗口是 [c-1, c+1]，下一步窗口是 [c, c+2]
				// 移除列 c-1（如果是第一次，c=1，移除列0）
				--h[q_row0[left]];
				--h[q_row1[left]];
				--h[q_row2[left]];
			}
			int right = c + 2;
			if (right < cols) 
            {
				++h[q_row0[right]];
				++h[q_row1[right]];
				++h[q_row2[right]];
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

static inline __m128i quantizePhase8(__m256 gx, __m256 gy, float* angle_dst)
{
    const __m256 rad_to_deg = _mm256_set1_ps(
        180.0f / 3.14159265358979323846f);
    const __m256 zero = _mm256_setzero_ps();
    __m256 angle_v = _mm256_mul_ps(_mm256_atan2_ps(gy, gx), rad_to_deg);
    const __m256 negative = _mm256_cmp_ps(angle_v, zero, _CMP_LT_OQ);
    angle_v = _mm256_add_ps(
        angle_v, _mm256_and_ps(negative, _mm256_set1_ps(360.0f)));
    if (angle_dst)
        _mm256_storeu_ps(angle_dst, angle_v);

    const __m256 scaled = _mm256_mul_ps(
        angle_v, _mm256_set1_ps(16.0f / 360.0f));
    // Make round-to-nearest explicit instead of depending on MXCSR state.
    const __m256 rounded = _mm256_round_ps(
        scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i bins = _mm256_cvttps_epi32(rounded);
    bins = _mm256_and_si256(bins, _mm256_set1_epi32(7));

    const __m128i lo = _mm256_castsi256_si128(bins);
    const __m128i hi = _mm256_extracti128_si256(bins, 1);
    return _mm_packus_epi16(
        _mm_packus_epi32(lo, hi), _mm_setzero_si128());
}

static inline unsigned char quantizePhaseScalar(
    int gx, int gy, float* angle_dst)
{
    float angle_value = atan2f(static_cast<float>(gy), static_cast<float>(gx)) *
        (180.0f / 3.14159265358979323846f);
    if (angle_value < 0.0f)
        angle_value += 360.0f;
    if (angle_dst)
        *angle_dst = angle_value;

    const __m128 scaled = _mm_set_ss(angle_value * (16.0f / 360.0f));
    const __m128 rounded = _mm_round_ss(
        _mm_setzero_ps(), scaled,
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return static_cast<unsigned char>(_mm_cvttss_si32(rounded) & 7);
}

static void SobelPhaseMatch_AVX2(
    const unsigned char* src, int width, int height, int stride,
    float* magnitude, int magnitude_stride,
    unsigned char* angle_quant, int angle_quant_stride,
    float* angle, int angle_stride)
{
#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y)
    {
        const unsigned char* prev =
            src + std::max(0, y - 1) * stride;
        const unsigned char* curr = src + y * stride;
        const unsigned char* next =
            src + std::min(height - 1, y + 1) * stride;
        float* magnitude_row = magnitude + y * magnitude_stride;
        unsigned char* quant_row = angle_quant + y * angle_quant_stride;
        float* angle_row = angle ? angle + y * angle_stride : nullptr;

        int x = 0;
        for (; x < width && x < 1; ++x)
        {
            const int left = 0;
            const int right = std::min(1, width - 1);
            const int gx = prev[right] - prev[left] +
                2 * (curr[right] - curr[left]) +
                next[right] - next[left];
            const int gy = next[left] + next[right] +
                2 * (next[x] - prev[x]) - prev[left] - prev[right];
            magnitude_row[x] = static_cast<float>(gx * gx + gy * gy);
            quant_row[x] = quantizePhaseScalar(
                gx, gy, angle_row ? angle_row + x : nullptr);
        }

        for (; x + 15 < width - 1; x += 16)
        {
            const __m256i p0 = _mm256_cvtepu8_epi16(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(prev + x - 1)));
            const __m256i p1 = _mm256_cvtepu8_epi16(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(prev + x)));
            const __m256i p2 = _mm256_cvtepu8_epi16(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(prev + x + 1)));
            const __m256i c0 = _mm256_cvtepu8_epi16(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(curr + x - 1)));
            const __m256i c2 = _mm256_cvtepu8_epi16(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(curr + x + 1)));
            const __m256i n0 = _mm256_cvtepu8_epi16(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(next + x - 1)));
            const __m256i n1 = _mm256_cvtepu8_epi16(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(next + x)));
            const __m256i n2 = _mm256_cvtepu8_epi16(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(next + x + 1)));

            const __m256i gx16 = _mm256_add_epi16(
                _mm256_add_epi16(
                    _mm256_sub_epi16(p2, p0),
                    _mm256_slli_epi16(_mm256_sub_epi16(c2, c0), 1)),
                _mm256_sub_epi16(n2, n0));
            const __m256i gy16 = _mm256_sub_epi16(
                _mm256_add_epi16(
                    _mm256_add_epi16(n0, n2),
                    _mm256_slli_epi16(_mm256_sub_epi16(n1, p1), 1)),
                _mm256_add_epi16(p0, p2));

            const __m256i gx_lo_i32 = _mm256_cvtepi16_epi32(
                _mm256_castsi256_si128(gx16));
            const __m256i gx_hi_i32 = _mm256_cvtepi16_epi32(
                _mm256_extracti128_si256(gx16, 1));
            const __m256i gy_lo_i32 = _mm256_cvtepi16_epi32(
                _mm256_castsi256_si128(gy16));
            const __m256i gy_hi_i32 = _mm256_cvtepi16_epi32(
                _mm256_extracti128_si256(gy16, 1));

            const __m256i mag_lo_i32 = _mm256_add_epi32(
                _mm256_mullo_epi32(gx_lo_i32, gx_lo_i32),
                _mm256_mullo_epi32(gy_lo_i32, gy_lo_i32));
            const __m256i mag_hi_i32 = _mm256_add_epi32(
                _mm256_mullo_epi32(gx_hi_i32, gx_hi_i32),
                _mm256_mullo_epi32(gy_hi_i32, gy_hi_i32));
            _mm256_storeu_ps(
                magnitude_row + x, _mm256_cvtepi32_ps(mag_lo_i32));
            _mm256_storeu_ps(
                magnitude_row + x + 8, _mm256_cvtepi32_ps(mag_hi_i32));

            const __m256 gx_lo = _mm256_cvtepi32_ps(gx_lo_i32);
            const __m256 gx_hi = _mm256_cvtepi32_ps(gx_hi_i32);
            const __m256 gy_lo = _mm256_cvtepi32_ps(gy_lo_i32);
            const __m256 gy_hi = _mm256_cvtepi32_ps(gy_hi_i32);
            const __m128i quant_lo = quantizePhase8(
                gx_lo, gy_lo, angle_row ? angle_row + x : nullptr);
            const __m128i quant_hi = quantizePhase8(
                gx_hi, gy_hi, angle_row ? angle_row + x + 8 : nullptr);
            _mm_storel_epi64(
                reinterpret_cast<__m128i*>(quant_row + x), quant_lo);
            _mm_storel_epi64(
                reinterpret_cast<__m128i*>(quant_row + x + 8), quant_hi);
        }

        for (; x < width; ++x)
        {
            const int left = x - 1;
            const int right = std::min(x + 1, width - 1);
            const int gx = prev[right] - prev[left] +
                2 * (curr[right] - curr[left]) +
                next[right] - next[left];
            const int gy = next[left] + next[right] +
                2 * (next[x] - prev[x]) - prev[left] - prev[right];
            magnitude_row[x] = static_cast<float>(gx * gx + gy * gy);
            quant_row[x] = quantizePhaseScalar(
                gx, gy, angle_row ? angle_row + x : nullptr);
        }
    }
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
    float* sobleAngle_C3_32F_Speed, const int32_t Stride3, short* sobel_3dx_Speed, int Stride4, short* sobel_3dy_Speed, int Stride5, 
    float* sobel_3dx_Float_Speed, int Stride6, float* sobel_3dy_Float_Speed, int Stride7)
{
	unsigned char* First = thread_local_buffers[0];
	unsigned char* Second = thread_local_buffers[1];
	unsigned char* Third = thread_local_buffers[2];
	for (int Y = start_row; Y < start_row + thread_stride && Y < Height; Y++) 
    {
		unsigned char* LinePS = Src + Y * Stride;
		unsigned char* LinePD = Dest ? Dest + Y * Stride : nullptr;
        float* LinePD_FloatTest = sobelResMat_C3_32F_Speed + Y * Stride2;
        float* LinePD_AngleSpeed = sobleAngle_C3_32F_Speed ? sobleAngle_C3_32F_Speed + Y * Stride3 : nullptr;
        short* sobel_dx_SpeedPtr = sobel_3dx_Speed ? sobel_3dx_Speed + Y * Stride4 : nullptr;
        short* sobel_dy_SpeedPtr = sobel_3dy_Speed ? sobel_3dy_Speed + Y * Stride5 : nullptr;
        float* sobel_dx_Float_SpeedPtr = sobel_3dx_Float_Speed + Y * Stride6;
        float* sobel_dy_Float_SpeedPtr = sobel_3dy_Float_Speed + Y * Stride7;
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
			for (int X = 0; X < Block * BlockSize; X += BlockSize)
			{
				// 单通道：邻域偏移为 1/2（对应三通道的 3/6）
				__m256i FirstP0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(First + X)));
				__m256i FirstP1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(First + X + 1)));
				__m256i FirstP2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(First + X + 2)));

				__m256i SecondP0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(Second + X)));
				__m256i SecondP2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(Second + X + 2)));

				__m256i ThirdP0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(Third + X)));
				__m256i ThirdP1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(Third + X + 1)));
				__m256i ThirdP2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(Third + X + 2)));

				// Signed gradients are reused for magnitude and phase output.
				__m256i GX16_N0_abs = _mm256_adds_epi16(_mm256_adds_epi16(_mm256_subs_epi16(FirstP2, FirstP0), _mm256_slli_epi16(_mm256_subs_epi16(SecondP2, SecondP0), 1)), _mm256_subs_epi16(ThirdP2, ThirdP0));
				__m256i GY16_N0_abs = _mm256_subs_epi16(_mm256_adds_epi16(_mm256_adds_epi16(ThirdP0, ThirdP2), _mm256_slli_epi16(_mm256_subs_epi16(ThirdP1, FirstP1), 1)), _mm256_adds_epi16(FirstP0, FirstP2));
				__m256i GX16 = _mm256_abs_epi16(GX16_N0_abs);
				__m256i GY16 = _mm256_abs_epi16(GY16_N0_abs);

				__m256i GXYL = _mm256_unpacklo_epi16(GX16, GY16);
				__m256i GXYH = _mm256_unpackhi_epi16(GX16, GY16);

				__m256 squareSum_Low = _mm256_cvtepi32_ps(_mm256_madd_epi16(GXYL, GXYL));
				__m256 squareSum_Height = _mm256_cvtepi32_ps(_mm256_madd_epi16(GXYH, GXYH));

				// The 8-bit magnitude output is optional. The matching path does
				// not consume it, so avoid the two sqrt operations and stores.
				if (LinePD)
				{
					const __m256i Zero = _mm256_setzero_si256();
					__m256i ResultL = _mm256_cvtps_epi32(_mm256_sqrt_ps(squareSum_Low));
					__m256i ResultH = _mm256_cvtps_epi32(_mm256_sqrt_ps(squareSum_Height));
					__m256i Result = _mm256_packus_epi16(_mm256_packus_epi32(ResultL, ResultH), Zero);

					__m128i Ans1 = _mm256_castsi256_si128(Result);
					_mm_storel_epi64((__m128i*)(LinePD + X), Ans1);
					__m128i Ans2 = _mm256_extracti128_si256(Result, 1);
					_mm_storel_epi64((__m128i*)(LinePD + X + 8), Ans2);
				}

				// 按像素顺序写回 GX^2+GY^2（修正 AVX 128-bit lane 交错）
				__m128 low_low_128 = _mm256_castps256_ps128(squareSum_Low);
				__m128 low_high_128 = _mm256_extractf128_ps(squareSum_Low, 1);
				__m128 high_low_128 = _mm256_castps256_ps128(squareSum_Height);
				__m128 high_high_128 = _mm256_extractf128_ps(squareSum_Height, 1);
				__m256 part1 = _mm256_set_m128(high_low_128, low_low_128);
				__m256 part2 = _mm256_set_m128(high_high_128, low_high_128);
				_mm256_storeu_ps(LinePD_FloatTest + X, part1);
				_mm256_storeu_ps(LinePD_FloatTest + X + 8, part2);

				if (sobel_dx_SpeedPtr)
					_mm256_storeu_si256((__m256i*)(sobel_dx_SpeedPtr + X), GX16_N0_abs);
				if (sobel_dy_SpeedPtr)
					_mm256_storeu_si256((__m256i*)(sobel_dy_SpeedPtr + X), GY16_N0_abs);

                // ***增加float类型的输出
				// 提取低8个 int16
				__m128i gx_low_i16 = _mm256_castsi256_si128(GX16_N0_abs);
				__m128i gx_high_i16 = _mm256_extracti128_si256(GX16_N0_abs, 1);
				__m128i gy_low_i16 = _mm256_castsi256_si128(GY16_N0_abs);
				__m128i gy_high_i16 = _mm256_extracti128_si256(GY16_N0_abs, 1);
				// 扩展为 int32
				__m256i gx_low_i32 = _mm256_cvtepi16_epi32(gx_low_i16);
				__m256i gx_high_i32 = _mm256_cvtepi16_epi32(gx_high_i16);
				__m256i gy_low_i32 = _mm256_cvtepi16_epi32(gy_low_i16);
				__m256i gy_high_i32 = _mm256_cvtepi16_epi32(gy_high_i16);
				// 转换为 float
				__m256 gx_low_ps = _mm256_cvtepi32_ps(gx_low_i32);
				__m256 gx_high_ps = _mm256_cvtepi32_ps(gx_high_i32);
				__m256 gy_low_ps = _mm256_cvtepi32_ps(gy_low_i32);
				__m256 gy_high_ps = _mm256_cvtepi32_ps(gy_high_i32);
				// 存储
				_mm256_storeu_ps(sobel_dx_Float_SpeedPtr + X, gx_low_ps);
				_mm256_storeu_ps(sobel_dx_Float_SpeedPtr + X + 8, gx_high_ps);
				_mm256_storeu_ps(sobel_dy_Float_SpeedPtr + X, gy_low_ps);
				_mm256_storeu_ps(sobel_dy_Float_SpeedPtr + X + 8, gy_high_ps);
			}

			for (int X = Block * BlockSize; X < Width; X++)
			{
				int GX = First[X + 2] - First[X] + (Second[X + 2] - Second[X]) * 2 + Third[X + 2] - Third[X];
				int GY = Third[X] + Third[X + 2] + (Third[X + 1] - First[X + 1]) * 2 - First[X] - First[X + 2];
				if (LinePD)
					LinePD[X] = IM_ClampToByte((int)(sqrtf(GX * GX + GY * GY + 0.0F) + 0.5f));
				LinePD_FloatTest[X] = (float)(GX * GX + GY * GY);
				if (sobel_dx_SpeedPtr)
					sobel_dx_SpeedPtr[X] = (short)GX;
				if (sobel_dy_SpeedPtr)
					sobel_dy_SpeedPtr[X] = (short)GY;
                sobel_dx_Float_SpeedPtr[X] = (float)GX;
                sobel_dy_Float_SpeedPtr[X] = (float)GY;
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
    float* sobleAngle_C3_32F_Speed, int Stride3, short* sobel_3dx_Speed, int Stride4, short* sobel_3dy_Speed, int Stride5, 
    float* sobel_3dx_Float_Speed, int Stride6, float* sobel_3dy_Float_Speed, int Stride7,
    unsigned char* external_row_buffers = nullptr, int external_buffer_stride = 0)
{
	//INIT
	int Channel = Stride / Width;
    int BlockSize = 16;
    int Block = (Width * Channel) / BlockSize;

    const int thread_count = std::max(1, std::min(
		std::max(1, Height >> 4),
		omp_get_max_threads()));
    const int required_buffer_stride = (Width + 2) * 3 * Channel;

#pragma omp parallel num_threads(thread_count)
	{
        const int tid = omp_get_thread_num();
        const int start = Height * tid / thread_count;
        const int end = Height * (tid + 1) / thread_count;
        const int thread_stride = end - start;
        unsigned char* rowCopy = external_row_buffers
            ? external_row_buffers + tid * external_buffer_stride
            : static_cast<unsigned char*>(
                _mm_malloc(required_buffer_stride, 32));
        if (!rowCopy)
            CV_Error(Error::StsNoMem, "Sobel row buffer allocation failed");

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
        unsigned char* thread_buffers[3] = {first, second, third};
        _Sobel(Src, Width, Height, start, thread_stride, Stride, Dest,
            thread_buffers, sobelResMat_C3_32F_Speed, Stride2, Channel,
            BlockSize, Block, sobleAngle_C3_32F_Speed, Stride3,
            sobel_3dx_Speed, Stride4, sobel_3dy_Speed, Stride5,
            sobel_3dx_Float_Speed, Stride6,
            sobel_3dy_Float_Speed, Stride7);

        if (!external_row_buffers)
            _mm_free(rowCopy);
	}
}

void _phase(float* SrcX, float* SrcY, float* DestAngle, unsigned char* DestAngleQuant, const int32_t Width, const int32_t Height, const int32_t start_row, const int32_t thread_stride,
    const int32_t Stride, const int Channel, const int BlockSize, const int Block, bool angleInDegrees = false)
{
	const float RAD_TO_DEG = 180.0f / 3.14159265358979323846f;
	const __m256 rad_to_deg_vec = _mm256_set1_ps(RAD_TO_DEG);
	const __m256 zero_vec = _mm256_setzero_ps();
	const __m256 three_sixty_vec = _mm256_set1_ps(360.0f);
    const int end_row = std::min(start_row + thread_stride, Height);

#pragma omp parallel for schedule(static)
    for (int Y = start_row; Y < end_row; Y++)
    {
        float* LinePS_X = SrcX + Y * Stride;
        float* LinePS_Y = SrcY + Y * Stride;
        float* LinePD = DestAngle ? DestAngle + Y * Stride : nullptr;

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
                if (LinePD)
                    _mm256_storeu_ps(LinePD + X, out_atan2);

				// 量化, 乘 16/360
				const __m256 scale = _mm256_set1_ps(16.0f / 360.0f);
			    // 对 out_atan2 量化
				__m256 scaled = _mm256_mul_ps(out_atan2, scale);
				// Explicit nearest rounding; do not depend on the caller's MXCSR mode.
				__m256 rounded = _mm256_round_ps(
                    scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
				__m256i int32 = _mm256_cvttps_epi32(rounded);
				// 限制到 0~15（按位与 15，因为 16 -> 0）
				__m256i mask2 = _mm256_set1_epi32(15);
				int32 = _mm256_and_si256(int32, mask2);
				// 提取低 128 位（前 4 个 int32）和高 128 位（后 4 个 int32）
				__m128i lo32 = _mm256_castsi256_si128(int32);
				__m128i hi32 = _mm256_extracti128_si256(int32, 1);
				// 打包为 8 个 int16 (低半部分 lo32, 高半部分 hi32)
				__m128i int16 = _mm_packus_epi32(lo32, hi32);   // 注意：这是 SSE2 指令，返回 8 个 uint16_t
				// 压缩为 8 个 uint8_t (取 int16 的低 8 位)
				__m128i uint8 = _mm_packus_epi16(int16, _mm_setzero_si128());
                // 映射到 0~7
				__m128i mask7 = _mm_set1_epi8(7);
				__m128i uint8_7 = _mm_and_si128(uint8, mask7);
				// int32 -> uint8 produces 8 valid values; store exactly 8 bytes.
				_mm_storel_epi64((__m128i*)(DestAngleQuant + Y * Stride + X), uint8_7);
			}

			// Handle widths that are not divisible by the AVX block size.
			for (int X = Block * BlockSize; X < Width; ++X)
			{
				float value = atan2f(LinePS_Y[X], LinePS_X[X]);
				if (angleInDegrees)
					value *= RAD_TO_DEG;
				if (value < 0.0f)
					value += 360.0f;
				if (LinePD)
					LinePD[X] = value;

                const __m128 rounded = _mm_round_ss(
                    _mm_setzero_ps(),
                    _mm_set_ss(value * (16.0f / 360.0f)),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
				const int bucket = _mm_cvttss_si32(rounded);
				DestAngleQuant[Y * Stride + X] =
					static_cast<unsigned char>(bucket & 7);
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

void phaseSelf(float* SrcX, float* SrcY, float* DestAngle, unsigned char* DestAngleQuant, const int32_t Width, const int32_t Height, const int32_t Stride, bool angleInDegrees = false)
{
	//INIT
	int Channel = Stride / Width;
	int BlockSize = 8;
	int Block = (Width * Channel) / BlockSize;
    _phase(SrcX, SrcY, DestAngle, DestAngleQuant, Width, Height, 0, Height,
           Stride, Channel, BlockSize, Block, angleInDegrees);
}

static void quantizedOrientations(const Mat &src, Mat &magnitude,
                                  Mat &angle, Mat& angle_ori, float threshold,
                                  bool keep_angle_ori,
                                  ColorGradientPyramid::ProcessingWorkspace& workspace)
{
    Mat& smoothed = workspace.smoothed;
    // Compute horizontal and vertical image derivatives on all color channels separately
    static const int SIGMA = 1;
    // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
    //GaussianBlur(src, smoothed, Size(SIGMA * 8 + 1, SIGMA * 8 + 1), SIGMA, 0, BORDER_REPLICATE);
    // 先把Padding区域去除,纯黑区域且当SIGM=1时,耗时会大大增加
	// GaussBlur_SSE 仅支持 3 通道；单通道走 GaussBlurGray_SSE
	int Height = src.rows;
	int Width = src.cols;
    int Stride = Width * src.channels();
    smoothed.create(src.size(), src.type());
	if (src.channels() == 1)
	{
        const int padding = 3 * SIGMA;
        workspace.gauss_float_buffer.create(
            Height + 2 * padding, Width + 2 * padding, CV_32F);
		GaussBlurGray_SSE(
            src.data, smoothed.data, Width, Height, Stride, SIGMA,
            workspace.gauss_float_buffer.ptr<float>());
	}
	else
	{
        CV_Assert(src.channels() == 3);
		GaussBlur_SSE(src.data, smoothed.data, Width, Height, Stride, SIGMA);
	}

    if(src.channels() == 1)
    {
        //Mat sobel_dx, sobel_dy, sobel_ag;
        //Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
        //Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
        //magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
        //phase(sobel_dx, sobel_dy, sobel_ag, true);
        //hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
        //angle_ori = sobel_ag;

        magnitude.create(smoothed.size(), CV_32F);
        workspace.angle_quant.create(smoothed.size(), CV_8U);
        if (keep_angle_ori)
        {
            angle_ori.create(smoothed.size(), CV_32F);
        }
        else
        {
            angle_ori.release();
        }

        SobelPhaseMatch_AVX2(
            smoothed.ptr<uchar>(), smoothed.cols, smoothed.rows,
            static_cast<int>(smoothed.step1()),
            magnitude.ptr<float>(), static_cast<int>(magnitude.step1()),
            workspace.angle_quant.ptr<uchar>(),
            static_cast<int>(workspace.angle_quant.step1()),
            keep_angle_ori ? angle_ori.ptr<float>() : nullptr,
            keep_angle_ori ? static_cast<int>(angle_ori.step1()) : 0);
		hysteresisGradientSpeedUp(
            magnitude, angle, angle_ori, workspace.angle_quant, threshold * threshold);
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
        Mat sobel_3dx_Float_Speed = Mat::zeros(smoothed.size(), CV_32FC3);
        Mat sobel_3dy_Float_Speed = Mat::zeros(smoothed.size(), CV_32FC3);
        const int sobel_threads = std::max(
            1, std::min(std::max(1, smoothed.rows >> 4), omp_get_max_threads()));
        const int sobel_buffer_stride = (smoothed.cols + 2) * 9;
        workspace.sobel_row_buffers.create(
            sobel_threads, sobel_buffer_stride, CV_8U);
        const int sobel_buffer_step =
            static_cast<int>(workspace.sobel_row_buffers.step1());
		//step1(i):每一维元素的通道数,函数默认参数是0,即: step1() == step1(0)
		//step[i]:每一维元素的大小，单位字节
		Sobel_AVX2(smoothed.ptr(), sobelResMat_C3_8U_Speed.ptr(), smoothed.cols, smoothed.rows, smoothed.step1(), 
            sobelResMat_C3_32F_Speed.ptr<float>(), sobelResMat_C3_32F_Speed.step1(), sobleAngle_C3_32F_Speed.ptr<float>(), sobleAngle_C3_32F_Speed.step1(),
            sobel_3dx_Speed.ptr<short>(), sobel_3dx_Speed.step1(), sobel_3dy_Speed.ptr<short>(), sobel_3dy_Speed.step1(), 
            sobel_3dx_Float_Speed.ptr<float>(), sobel_3dx_Float_Speed.step1(), sobel_3dy_Float_Speed.ptr<float>(), sobel_3dy_Float_Speed.step1(),
            workspace.sobel_row_buffers.ptr<uchar>(), sobel_buffer_step);

        magnitude.setTo(Scalar(0));
        workspace.sobel_dx_float.create(src.size(), CV_32F);
        workspace.sobel_dy_float.create(src.size(), CV_32F);

		float* ptrmg_Test = magnitude.ptr<float>();
        float* ptr_sobel_dx_Max = workspace.sobel_dx_float.ptr<float>();
        float* ptr_sobel_dy_Max = workspace.sobel_dy_float.ptr<float>();

        float* ptr_SobelResMat_C3_32F_Speed = sobelResMat_C3_32F_Speed.ptr<float>();
		short* ptr_sobel_3dx_Speed = sobel_3dx_Speed.ptr<short>();
		short* ptr_sobel_3dy_Speed = sobel_3dy_Speed.ptr<short>();

		const int length_magnitude_Test = static_cast<const int>(magnitude.step1());
		const int length_sobel_dx_Max = static_cast<const int>(workspace.sobel_dx_float.step1());
		const int length_sobel_dy_Max = static_cast<const int>(workspace.sobel_dy_float.step1());

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
					ptrmg_Test[ind] = (float)mag1;
                    ptr_sobel_dx_Max[ind] = (float)ptr_sobel_3dx_Speed[i];
                    ptr_sobel_dy_Max[ind] = (float)ptr_sobel_3dy_Speed[i];
				}
				else if (mag2 >= mag1 && mag2 >= mag3)
				{
					ptrmg_Test[ind] = (float)mag2;
					ptr_sobel_dx_Max[ind] = (float)ptr_sobel_3dx_Speed[i + 1];
					ptr_sobel_dy_Max[ind] = (float)ptr_sobel_3dy_Speed[i + 1];
				}
				else
				{
					ptrmg_Test[ind] = (float)mag3;
					ptr_sobel_dx_Max[ind] = (float)ptr_sobel_3dx_Speed[i + 2];
					ptr_sobel_dy_Max[ind] = (float)ptr_sobel_3dy_Speed[i + 2];
				}
				++ind;
			}
            ptr_SobelResMat_C3_32F_Speed += length_sobelResMat_C3_32F_Speed;
            ptr_sobel_3dx_Speed += length_sobel_3dx_Max;
            ptr_sobel_3dy_Speed += length_sobel_3dy_Max;
            ptrmg_Test += length_magnitude_Test;
            ptr_sobel_dx_Max += length_sobel_dx_Max;
            ptr_sobel_dy_Max += length_sobel_dy_Max;
		}
        workspace.angle_quant.create(smoothed.size(), CV_8U);
        workspace.angle_quant.setTo(Scalar(0));
        if (keep_angle_ori)
        {
            angle_ori.create(smoothed.size(), CV_32F);
            angle_ori.setTo(Scalar(0));
        }
        else
        {
            angle_ori.release();
        }
        phaseSelf(workspace.sobel_dx_float.ptr<float>(),
            workspace.sobel_dy_float.ptr<float>(),
            keep_angle_ori ? angle_ori.ptr<float>() : nullptr,
            workspace.angle_quant.ptr<uchar>(), smoothed.cols, smoothed.rows,
            workspace.sobel_dx_float.step1(), true);

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
        //hysteresisGradient(magnitude_Test, angle, sobleAngle_C1_32F_Speed, threshold* threshold);
        hysteresisGradientSpeedUp(
            magnitude, angle, angle_ori, workspace.angle_quant, threshold * threshold);
    }
}

ColorGradientPyramid::ColorGradientPyramid(const Mat &_src, const Mat &_mask,
                                           float _weak_threshold, size_t _num_features,
                                           float _strong_threshold, bool _match_only)
    : src(_src),
      mask(_mask),
      pyramid_level(0),
      weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold),
      match_only(_match_only)
{
    update();
}

void ColorGradientPyramid::reset(const Mat &_src, const Mat &_mask,
                                 float _weak_threshold, size_t _num_features,
                                 float _strong_threshold, bool _match_only)
{
    src = _src;
    mask = _mask;
    pyramid_level = 0;
    weak_threshold = _weak_threshold;
    num_features = _num_features;
    strong_threshold = _strong_threshold;
    match_only = _match_only;
    update();
}

//match() 后续只使用 angle，不使用持久化的 magnitude 和 angle_ori；
//后两者主要供 extractTemplate() 使用。增加 processForMatch()，
// 只保留量化角度，可减少连续角度图写入和内存占用
void ColorGradientPyramid::update()
{
    if (workspaces.size() <= static_cast<size_t>(pyramid_level))
        workspaces.resize(pyramid_level + 1);

    ProcessingWorkspace& workspace = workspaces[pyramid_level];
    quantizedOrientations(src, workspace.magnitude, workspace.angle,
                          workspace.angle_ori, weak_threshold, !match_only,
                          workspace);
    magnitude = workspace.magnitude;
    angle = workspace.angle;
    angle_ori = workspace.angle_ori;
}

void ColorGradientPyramid::pyrDown()
{
    // Some parameters need to be adjusted
    num_features /= 2; /// @todo Why not 4?
    // Downsample the current inputs
    Size size(src.cols / 2, src.rows / 2);
    const int next_level = pyramid_level + 1;
    if (workspaces.size() <= static_cast<size_t>(next_level))
        workspaces.resize(next_level + 1);
    ProcessingWorkspace& next_workspace = workspaces[next_level];

    cv::pyrDown(src, next_workspace.pyramid_src, size);
    src = next_workspace.pyramid_src;

    if (!mask.empty())
    {
        resize(mask, next_workspace.pyramid_mask, size, 0.0, 0.0, INTER_NEAREST);
        mask = next_workspace.pyramid_mask;
    }

    pyramid_level = next_level;
    update();
}

void ColorGradientPyramid::quantize(Mat &dst) const
{
    if (mask.empty())
    {
        // Matching only reads this image, so an empty mask needs no copy or clear.
        dst = angle;
        return;
    }

    dst.create(angle.size(), CV_8U);
    dst.setTo(Scalar(0));
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
    CV_Assert(T > 0);
    const int vertical_vector_end = (Width / 32) * 32;
    const int horizontal_vector_end =
        (std::max(0, Width - T + 1) / 32) * 32;

#pragma omp parallel
    {
        // Vertical OR: accumulate in registers and write each output byte once.
#pragma omp for schedule(static)
        for (int y = 0; y < Height; ++y)
        {
            const int count = std::min(T, Height - y);
            unsigned char* dst_row = Dest + y * Stride;
            int x = 0;
            for (; x < vertical_vector_end; x += 32)
            {
                __m256i value = _mm256_setzero_si256();
                for (int j = 0; j < count; ++j)
                {
                    value = _mm256_or_si256(
                        value,
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                            Src + (y + j) * Stride + x)));
                }
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(dst_row + x), value);
            }
            for (; x < Width; ++x)
            {
                unsigned char value = 0;
                for (int j = 0; j < count; ++j)
                    value |= Src[(y + j) * Stride + x];
                dst_row[x] = value;
            }
        }

        // The implicit barrier above is required before horizontal spreading.
#pragma omp for schedule(static)
        for (int y = 0; y < Height; ++y)
        {
            unsigned char* row = Dest + y * Stride;
            int x = 0;
            for (; x < horizontal_vector_end; x += 32)
            {
                __m256i value =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + x));
                if (T == 4)
                {
                    value = _mm256_or_si256(value, _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(row + x + 1)));
                    value = _mm256_or_si256(value, _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(row + x + 2)));
                    value = _mm256_or_si256(value, _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(row + x + 3)));
                }
                else if (T == 8)
                {
                    for (int i = 1; i < 8; ++i)
                    {
                        value = _mm256_or_si256(value, _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(row + x + i)));
                    }
                }
                else
                {
                    for (int i = 1; i < T; ++i)
                    {
                        value = _mm256_or_si256(value, _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(row + x + i)));
                    }
                }
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(row + x), value);
            }
            for (; x < Width; ++x)
            {
                unsigned char value = 0;
                const int count = std::min(T, Width - x);
                for (int i = 0; i < count; ++i)
                    value |= row[x + i];
                row[x] = value;
            }
        }
    }
}

static const unsigned char LUT3 = 3;
// 1,2-->0 3-->LUT3
CV_DECL_ALIGNED(16)
static const unsigned char SIMILARITY_LUT[256] = {0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4};

static void computeResponseMaps(const Mat &src, std::vector<Mat> &response_maps)
{
    CV_Assert(src.type() == CV_8UC1);

    // Allocate response maps
    response_maps.resize(8);
    for (int i = 0; i < 8; ++i)
        response_maps[i].create(src.size(), CV_8U);

#if defined(__AVX2__) || defined(_M_AVX2)
    // Expand each 16-byte table once for AVX2's two independent 128-bit lanes.
    __m256i lut_low_vectors[8];
    __m256i lut_high_vectors[8];
    for (int ori = 0; ori < 8; ++ori)
    {
        const uchar* lut = SIMILARITY_LUT + 32 * ori;
        lut_low_vectors[ori] = _mm256_broadcastsi128_si256(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut)));
        lut_high_vectors[ori] = _mm256_broadcastsi128_si256(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut + 16)));
    }
#endif

    // Process all eight response maps while each source block is in a register.
    // This removes the lsb4/msb4 intermediate images and their memory traffic.
#pragma omp parallel for schedule(static)
    for (int r = 0; r < src.rows; ++r)
    {
        const uchar* src_r = src.ptr<uchar>(r);
        uchar* map_rows[8];
        for (int ori = 0; ori < 8; ++ori)
            map_rows[ori] = response_maps[ori].ptr<uchar>(r);

        int c = 0;

#if defined(__AVX2__) || defined(_M_AVX2)
        const __m256i low_mask = _mm256_set1_epi8(0x0F);
        for (; c <= src.cols - 32; c += 32)
        {
            const __m256i src_v =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_r + c));
            const __m256i low_index = _mm256_and_si256(src_v, low_mask);
            const __m256i high_index = _mm256_and_si256(
                _mm256_srli_epi16(src_v, 4), low_mask);

            for (int ori = 0; ori < 8; ++ori)
            {
                const __m256i low_response =
                    _mm256_shuffle_epi8(lut_low_vectors[ori], low_index);
                const __m256i high_response =
                    _mm256_shuffle_epi8(lut_high_vectors[ori], high_index);
                const __m256i response =
                    _mm256_max_epu8(low_response, high_response);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(map_rows[ori] + c), response);
            }
        }
#endif

        // Handles arbitrary image widths and builds without AVX2.
        for (; c < src.cols; ++c)
        {
            const int low_index = src_r[c] & 0x0F;
            const int high_index = src_r[c] >> 4;
            for (int ori = 0; ori < 8; ++ori)
            {
                const uchar* lut = SIMILARITY_LUT + 32 * ori;
                map_rows[ori][c] =
                    std::max(lut[low_index], lut[high_index + 16]);
            }
        }
    }

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

static void linearizeResponseMaps(const std::vector<Mat> &response_maps,
                                  std::vector<Mat> &linearized_maps, int T)
{
    CV_Assert(response_maps.size() == 8);
    CV_Assert(!response_maps.empty());
    CV_Assert(response_maps[0].rows % T == 0);
    CV_Assert(response_maps[0].cols % T == 0);

    // linearized has T^2 rows, where each row is a linear memory
    const int mem_width = response_maps[0].cols / T;
    const int mem_height = response_maps[0].rows / T;
    linearized_maps.resize(8);
    for (int ori = 0; ori < 8; ++ori)
        linearized_maps[ori].create(T * T, mem_width * mem_height, CV_8U);

    // Parallelize all orientation/grid combinations in one OpenMP region.
#pragma omp parallel for schedule(static)
    for (int task = 0; task < 8 * T * T; ++task)
    {
        const int ori = task / (T * T);
        const int index = task % (T * T);
        const int r_start = index / T;
        const int c_start = index % T;
        uchar *memory = linearized_maps[ori].ptr<uchar>(index);

        // Inner two loops copy every T-th pixel into the linear memory
        for (int r = r_start; r < response_maps[ori].rows; r += T)
        {
            const uchar *response_data = response_maps[ori].ptr<uchar>(r);
            for (int c = c_start; c < response_maps[ori].cols; c += T)
                *memory++ = response_data[c];
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

            if(mipp::N<uint8_t>() > 32)
            { //512 bits SIMD
                for (int row = 0; row < 16; row += mipp::N<int16_t>()/16)
                {
                    mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + row*16);

                    // load lm_ptr, 16 bytes once, for half
                    uint8_t local_v[mipp::N<uint8_t>()] = {0};
                    for(int slice=0; slice<mipp::N<uint8_t>()/16/2; slice++)
                    {
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
                for (int row = 0; row < 16; ++row)
                {
                    for(int col=0; col < 16; col+=mipp::N<int16_t>())
                    {
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
    // Reusable matching buffers are detector-owned; serialize concurrent calls
    // so they cannot overwrite one another.
    std::lock_guard<std::mutex> match_lock(match_mutex);

    Timer timer;
    std::vector<Match> matches;

    // Initialize each ColorGradient with our sources
    std::vector<Ptr<ColorGradientPyramid>> quantizers;
    CV_Assert(mask.empty() || mask.size() == source.size());
    if (match_quantizer.empty())
    {
        match_quantizer = modality->processForMatch(source, mask);
    }
    else
    {
        match_quantizer->reset(source, mask, modality->weak_threshold,
                               modality->num_features, modality->strong_threshold,
                               true);
    }
    quantizers.push_back(match_quantizer);

    // pyramid level -> ColorGradient -> quantization
    match_lm_pyramid.resize(pyramid_levels);
    match_spread_quantized.resize(pyramid_levels);
    match_response_maps.resize(pyramid_levels);
    for (int l = 0; l < pyramid_levels; ++l)
    {
        match_lm_pyramid[l].resize(1);
        match_lm_pyramid[l][0].resize(8);
    }
    LinearMemoryPyramid &lm_pyramid = match_lm_pyramid;

    // For each pyramid level, precompute linear memories for each ColorGradient
    std::vector<Size> &sizes = match_sizes;
    sizes.clear();
    sizes.reserve(pyramid_levels);
    for (int l = 0; l < pyramid_levels; ++l)
    {
        int T = T_at_level[l];
        std::vector<LinearMemories> &lm_level = lm_pyramid[l];

        if (l > 0)
        {
            for (int i = 0; i < (int)quantizers.size(); ++i)
                quantizers[i]->pyrDown();
        }

        Mat quantized;
        Mat &spread_quantized = match_spread_quantized[l];
        std::vector<Mat> &response_maps = match_response_maps[l];
        for (int i = 0; i < (int)quantizers.size(); ++i)
        {
            quantizers[i]->quantize(quantized);
            //spread(quantized, spread_quantized, T);
            spread_quantized.create(quantized.size(), CV_8UC1);
            IM_Spread(quantized.ptr(), spread_quantized.ptr(), quantized.cols, quantized.rows, static_cast<const int>(quantized.step1()), T);
            computeResponseMaps(spread_quantized, response_maps);

            LinearMemories &memories = lm_level[i];
            linearizeResponseMaps(response_maps, memories, T);
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

std::vector<Match> Detector::clusterMatchesByPosition(const std::vector<Match> &matches,
                                                      int max_dist_px)
{
    std::vector<Match> sorted = matches;
    std::sort(sorted.begin(), sorted.end());

    const int max_dist_sq = max_dist_px * max_dist_px;
    std::vector<char> suppressed(sorted.size(), 0);
    std::vector<Match> kept;
    kept.reserve(sorted.size());

    for (size_t i = 0; i < sorted.size(); ++i)
    {
        if (suppressed[i])
            continue;
        kept.push_back(sorted[i]);
        for (size_t j = i + 1; j < sorted.size(); ++j)
        {
            if (suppressed[j])
                continue;
            if (sorted[i].class_id != sorted[j].class_id)
                continue;
            const int dx = sorted[i].x - sorted[j].x;
            const int dy = sorted[i].y - sorted[j].y;
            if (dx * dx + dy * dy <= max_dist_sq)
                suppressed[j] = 1;
        }
    }
    return kept;
}

static cv::Point2f rotate2d(const cv::Point2f inPoint, const double angRad);
static cv::Point2f rotatePoint(const cv::Point2f inPoint, const cv::Point2f center, const double angRad);

static std::vector<Template> makeRotatedTemplatePyramid(
    const std::vector<Template> &to_rotate_tp, float theta, cv::Point2f center,
    bool keep_template_origin)
{
    std::vector<Template> tp;
    tp.resize(to_rotate_tp.size());
    cv::Point2f c = center;

    for (int l = 0; l < (int)to_rotate_tp.size(); ++l)
    {
        if (l > 0)
            c *= 0.5f;

        const int src_tl_x = to_rotate_tp[l].tl_x;
        const int src_tl_y = to_rotate_tp[l].tl_y;
        // Angle refine must keep the same template origin as the coarse match.
        // Re-running cropTemplates() shifts tl_x/tl_y, while match.x/y still refer
        // to the old origin, which misaligns features and collapses the score.
        if (keep_template_origin)
        {
            tp[l].tl_x = src_tl_x;
            tp[l].tl_y = src_tl_y;
            tp[l].width = to_rotate_tp[l].width;
            tp[l].height = to_rotate_tp[l].height;
        }

        for (const auto &f : to_rotate_tp[l].features)
        {
            Point2f p(f.x + src_tl_x, f.y + src_tl_y);
            Point2f p_rot = rotatePoint(p, c, -theta / 180.0 * CV_PI);

            Feature f_new;
            if (keep_template_origin)
            {
                f_new.x = int(p_rot.x + 0.5f) - src_tl_x;
                f_new.y = int(p_rot.y + 0.5f) - src_tl_y;
            }
            else
            {
                f_new.x = int(p_rot.x + 0.5f);
                f_new.y = int(p_rot.y + 0.5f);
            }
            // Matching uses label only. Feature.theta is often 0 after readClasses()
            // (older yaml omitted it). Recomputing label from a zero theta would
            // force every feature into the same orientation bin and collapse the
            // score. Recover an approximate orientation from label when needed.
            // label = round(theta * 16/360) & 7  =>  bin width 22.5deg.
            const float src_theta =
                (f.theta != 0.0f) ? f.theta : (f.label * 22.5f);
            f_new.theta = src_theta - theta;
            while (f_new.theta > 360)
                f_new.theta -= 360;
            while (f_new.theta < 0)
                f_new.theta += 360;
            f_new.label = int(f_new.theta * 16 / 360 + 0.5f);
            f_new.label &= 7;
            tp[l].features.push_back(f_new);
        }
        tp[l].pyramid_level = l;
    }

    if (!keep_template_origin)
        cropTemplates(tp);
    return tp;
}

// Continuous metrics for sub-degree angle refine. LINE-MOD similarity saturates at
// 100% (each feature contributes at most 4), so many nearby angles share the same
// discrete score; edge magnitude + orientation residual still vary smoothly.
static float featureThetaDeg(const Feature &f)
{
    return (f.theta != 0.0f) ? f.theta : (f.label * 22.5f);
}

static float circularAbsDiffDeg(float a, float b)
{
    float d = std::fmod(a - b, 360.0f);
    if (d < 0.0f)
        d += 360.0f;
    if (d > 180.0f)
        d = 360.0f - d;
    return d;
}

static void sampleGradBilinear(const Mat &mag, const Mat &angle_deg, float x, float y,
                               float &out_mag, float &out_angle)
{
    const int w = mag.cols;
    const int h = mag.rows;
    if (x < 0.0f || y < 0.0f || x >= w - 1.0f || y >= h - 1.0f)
    {
        const int xi = std::min(std::max(0, (int)std::floor(x + 0.5f)), w - 1);
        const int yi = std::min(std::max(0, (int)std::floor(y + 0.5f)), h - 1);
        out_mag = mag.at<float>(yi, xi);
        out_angle = angle_deg.at<float>(yi, xi);
        return;
    }
    const int x0 = (int)std::floor(x);
    const int y0 = (int)std::floor(y);
    const float dx = x - x0;
    const float dy = y - y0;
    const float m00 = mag.at<float>(y0, x0);
    const float m10 = mag.at<float>(y0, x0 + 1);
    const float m01 = mag.at<float>(y0 + 1, x0);
    const float m11 = mag.at<float>(y0 + 1, x0 + 1);
    out_mag = (1 - dx) * (1 - dy) * m00 + dx * (1 - dy) * m10 + (1 - dx) * dy * m01 +
              dx * dy * m11;

    // Interpolate orientation via unit vectors to avoid 360deg wrap artifacts.
    auto unit = [](float deg, float &cx, float &sx) {
        const float rad = deg * (float)CV_PI / 180.0f;
        cx = std::cos(rad);
        sx = std::sin(rad);
    };
    float c00, s00, c10, s10, c01, s01, c11, s11;
    unit(angle_deg.at<float>(y0, x0), c00, s00);
    unit(angle_deg.at<float>(y0, x0 + 1), c10, s10);
    unit(angle_deg.at<float>(y0 + 1, x0), c01, s01);
    unit(angle_deg.at<float>(y0 + 1, x0 + 1), c11, s11);
    const float cx = (1 - dx) * (1 - dy) * c00 + dx * (1 - dy) * c10 + (1 - dx) * dy * c01 +
                     dx * dy * c11;
    const float sx = (1 - dx) * (1 - dy) * s00 + dx * (1 - dy) * s10 + (1 - dx) * dy * s01 +
                     dx * dy * s11;
    float deg = std::atan2(sx, cx) * 180.0f / (float)CV_PI;
    if (deg < 0.0f)
        deg += 360.0f;
    out_angle = deg;
}

// Higher is better. Combines edge strength and orientation agreement at float
// feature positions (no integer rounding of the rotated contour).
static float continuousAngleScore(const Template &base, float delta_deg, Point2f center,
                                  int match_x, int match_y, const Mat &mag,
                                  const Mat &angle_deg)
{
    if (base.features.empty())
        return -1e30f;

    const float ang_rad = -delta_deg / 180.0f * (float)CV_PI;
    const float cos_a = std::cos(ang_rad);
    const float sin_a = std::sin(ang_rad);
    const int tl_x = base.tl_x;
    const int tl_y = base.tl_y;

    double mag_sum = 0.0;
    double ori_sum = 0.0;
    int count = 0;
    for (const auto &f : base.features)
    {
        const float abs_x = (float)f.x + (float)tl_x;
        const float abs_y = (float)f.y + (float)tl_y;
        const float dx = abs_x - center.x;
        const float dy = abs_y - center.y;
        const float rot_abs_x = center.x + cos_a * dx - sin_a * dy;
        const float rot_abs_y = center.y + sin_a * dx + cos_a * dy;
        const float fx = rot_abs_x - (float)tl_x;
        const float fy = rot_abs_y - (float)tl_y;
        const float img_x = (float)match_x + fx;
        const float img_y = (float)match_y + fy;

        float m = 0.0f, img_ang = 0.0f;
        sampleGradBilinear(mag, angle_deg, img_x, img_y, m, img_ang);

        float feat_ang = featureThetaDeg(f) - delta_deg;
        while (feat_ang >= 360.0f)
            feat_ang -= 360.0f;
        while (feat_ang < 0.0f)
            feat_ang += 360.0f;

        mag_sum += m;
        ori_sum += circularAbsDiffDeg(img_ang, feat_ang);
        ++count;
    }
    if (count == 0)
        return -1e30f;
    const double mean_mag = mag_sum / count;
    const double mean_ori = ori_sum / count;
    // Prefer strong edges; penalize orientation mismatch (degrees).
    return (float)(mean_mag / (1.0 + mean_ori / 45.0));
}

static float parabolicPeak(float x0, float y0, float x1, float y1, float x2, float y2)
{
    // Peak of parabola through (x0,y0),(x1,y1),(x2,y2); assumes x1 mid, equal step.
    const float denom = (y0 - 2.0f * y1 + y2);
    if (std::abs(denom) < 1e-12f)
        return x1;
    const float step = x1 - x0;
    const float delta = 0.5f * (y0 - y2) / denom;
    if (delta < -1.0f || delta > 1.0f)
        return x1;
    return x1 + delta * step;
}

void Detector::refineMatchAngles(const Mat &source, std::vector<Match> &matches,
                                 const std::vector<float> &template_angles,
                                 Point2f rotate_center, float half_range,
                                 float angle_step, const Mat &mask) const
{
    if (matches.empty())
        return;
    CV_Assert(half_range >= 0.0f);
    CV_Assert(angle_step > 0.0f);

    std::lock_guard<std::mutex> match_lock(match_mutex);

    // Rebuild response maps when cache is empty or image size changed.
    const bool need_rebuild =
        match_sizes.empty() || match_sizes[0] != source.size() || match_lm_pyramid.empty();
    if (need_rebuild)
    {
        std::vector<Ptr<ColorGradientPyramid>> quantizers;
        CV_Assert(mask.empty() || mask.size() == source.size());
        if (match_quantizer.empty())
            match_quantizer = modality->processForMatch(source, mask);
        else
            match_quantizer->reset(source, mask, modality->weak_threshold,
                                   modality->num_features, modality->strong_threshold, true);
        quantizers.push_back(match_quantizer);

        match_lm_pyramid.resize(pyramid_levels);
        match_spread_quantized.resize(pyramid_levels);
        match_response_maps.resize(pyramid_levels);
        match_sizes.clear();
        match_sizes.reserve(pyramid_levels);
        for (int l = 0; l < pyramid_levels; ++l)
        {
            match_lm_pyramid[l].resize(1);
            match_lm_pyramid[l][0].resize(8);
            int T = T_at_level[l];
            if (l > 0)
            {
                for (int i = 0; i < (int)quantizers.size(); ++i)
                    quantizers[i]->pyrDown();
            }
            Mat quantized;
            Mat &spread_quantized = match_spread_quantized[l];
            std::vector<Mat> &response_maps = match_response_maps[l];
            for (int i = 0; i < (int)quantizers.size(); ++i)
            {
                quantizers[i]->quantize(quantized);
                spread_quantized.create(quantized.size(), CV_8UC1);
                IM_Spread(quantized.ptr(), spread_quantized.ptr(), quantized.cols, quantized.rows,
                          static_cast<const int>(quantized.step1()), T);
                computeResponseMaps(spread_quantized, response_maps);
                linearizeResponseMaps(response_maps, match_lm_pyramid[l][i], T);
            }
            match_sizes.push_back(quantized.size());
        }
    }

    // Continuous gradient field for stage-2 angle scoring.
    Mat gray;
    if (source.channels() == 1)
        gray = source;
    else
        cvtColor(source, gray, COLOR_BGR2GRAY);
    Mat blurred, gx, gy, mag_f, angle_deg;
    GaussianBlur(gray, blurred, Size(5, 5), 1.0, 0.0, BORDER_REPLICATE);
    Sobel(blurred, gx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
    Sobel(blurred, gy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
    magnitude(gx, gy, mag_f);
    phase(gx, gy, angle_deg, true);

    const int T = T_at_level[0];
    const Size size = match_sizes[0];
    const std::vector<LinearMemories> &lms = match_lm_pyramid[0];
    const int border = 8 * T;
    const int offset = T / 2 + (T % 2 - 1);

    for (auto &match : matches)
    {
        TemplatesMap::const_iterator it = class_templates.find(match.class_id);
        if (it == class_templates.end())
            continue;
        if (match.template_id < 0 ||
            match.template_id >= (int)it->second.size() ||
            match.template_id >= (int)template_angles.size())
            continue;

        const TemplatePyramid &base_tp = it->second[match.template_id];
        const float base_angle = template_angles[match.template_id];
        if (match.angle == 0.0f)
            match.angle = base_angle;

        struct Cand
        {
            float angle;
            float delta;
            double raw;
            float score;
            int x, y;
            float cont;
        };
        std::vector<Cand> cands;
        cands.reserve((size_t)(2.0f * half_range / angle_step) + 4);

        double best_raw = -1.0;

        for (float delta = -half_range; delta <= half_range + 1e-4f; delta += angle_step)
        {
            const float candidate_angle = base_angle + delta;
            TemplatePyramid tp =
                (std::abs(delta) < 1e-6f)
                    ? base_tp
                    : makeRotatedTemplatePyramid(base_tp, delta, rotate_center, true);

            const int max_x = size.width - tp[0].width - border;
            const int max_y = size.height - tp[0].height - border;
            int x = match.x;
            int y = match.y;
            x = std::max(x, border);
            y = std::max(y, border);
            if (max_x >= border)
                x = std::min(x, max_x);
            if (max_y >= border)
                y = std::min(y, max_y);

            const Template &templ = tp[0];
            Mat similarities2;
            if (templ.features.size() < 64)
                similarityLocal_64(lms[0], templ, similarities2, size, T, Point(x, y));
            else if (templ.features.size() < 8192)
                similarityLocal(lms[0], templ, similarities2, size, T, Point(x, y));
            else
                continue;

            double best_raw_score = 0.0;
            Point best_position;
            minMaxLoc(similarities2, nullptr, &best_raw_score, nullptr, &best_position);
            const int numFeatures = static_cast<int>(templ.features.size());
            if (numFeatures <= 0)
                continue;
            const float score =
                static_cast<float>(best_raw_score * 100.0 / (4 * numFeatures));
            const int refined_x = (x / T - 8 + best_position.x) * T + offset;
            const int refined_y = (y / T - 8 + best_position.y) * T + offset;

            const float cont = continuousAngleScore(base_tp[0], delta, rotate_center,
                                                    refined_x, refined_y, mag_f, angle_deg);
            cands.push_back({candidate_angle, delta, best_raw_score, score, refined_x,
                             refined_y, cont});
            if (best_raw_score > best_raw)
                best_raw = best_raw_score;
        }

        if (cands.empty())
            continue;

        // Among discrete-score plateau (typical when many angles hit ~100%), pick by
        // continuous metric. Allow 1 raw unit slack (~one feature slightly worse).
        const double raw_tol = 1.0;
        float best_cont = -1e30f;
        size_t best_i = 0;
        for (size_t i = 0; i < cands.size(); ++i)
        {
            if (cands[i].raw + 1e-6 < best_raw - raw_tol)
                continue;
            if (cands[i].cont > best_cont)
            {
                best_cont = cands[i].cont;
                best_i = i;
            }
        }

        // Finer continuous-only search around the plateau winner.
        const float fine_half = angle_step;
        const float fine_step = std::max(0.02f, angle_step * 0.2f);
        const float center_delta = cands[best_i].delta;
        const int fix_x = cands[best_i].x;
        const int fix_y = cands[best_i].y;

        float fine_best_delta = center_delta;
        float fine_best_cont = best_cont;
        float prev_delta = 0, prev_cont = 0, cur_delta = 0, cur_cont = 0;
        float next_delta = 0, next_cont = 0;
        bool have_triplet = false;
        std::vector<std::pair<float, float>> fine_samples;
        for (float d = center_delta - fine_half; d <= center_delta + fine_half + 1e-5f;
             d += fine_step)
        {
            const float cont =
                continuousAngleScore(base_tp[0], d, rotate_center, fix_x, fix_y, mag_f,
                                     angle_deg);
            fine_samples.emplace_back(d, cont);
            if (cont > fine_best_cont)
            {
                fine_best_cont = cont;
                fine_best_delta = d;
            }
        }
        for (size_t i = 1; i + 1 < fine_samples.size(); ++i)
        {
            if (std::abs(fine_samples[i].first - fine_best_delta) < 1e-5f)
            {
                prev_delta = fine_samples[i - 1].first;
                prev_cont = fine_samples[i - 1].second;
                cur_delta = fine_samples[i].first;
                cur_cont = fine_samples[i].second;
                next_delta = fine_samples[i + 1].first;
                next_cont = fine_samples[i + 1].second;
                have_triplet = true;
                break;
            }
        }
        float refined_delta = fine_best_delta;
        if (have_triplet)
            refined_delta =
                parabolicPeak(prev_delta, prev_cont, cur_delta, cur_cont, next_delta,
                              next_cont);

        match.similarity = cands[best_i].score;
        match.angle = base_angle + refined_delta;
        match.x = fix_x;
        match.y = fix_y;
    }
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
	//int max_threads = omp_get_max_threads();
    //int num_procs = omp_get_num_procs();
    //printf("OpenMP: max_threads=%d, num_procs=%d\n", max_threads, num_procs);
//#pragma omp declare reduction \
//    (omp_insert: std::vector<Match>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//
//#pragma omp parallel for reduction(omp_insert:matches)
	int num_threads = omp_get_max_threads();
	if (num_threads <= 0) 
        num_threads = 1;
    //每个线程一个局部匹配容器
	std::vector<std::vector<Match>> thread_local_matches(num_threads);

#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (long long template_id = 0; template_id < template_pyramids.size(); ++template_id)
    {
        int tid = omp_get_thread_num(); // 获取当前线程 ID
        std::vector<Match>& local_vec = thread_local_matches[tid];

        const TemplatePyramid& tp = template_pyramids[template_id];

        // First match over the whole image at the lowest pyramid level
        /// @todo Factor this out into separate function
        const std::vector<LinearMemories>& lowest_lm = lm_pyramid.back();

        std::vector<Match> candidates;
        {
            // Compute similarity maps for each ColorGradient at lowest pyramid level
            Mat similarities;
            int lowest_start = static_cast<int>(tp.size() - 1);
            int lowest_T = T_at_level.back();
            int num_features = 0;

            {
				const Template& templ = tp[lowest_start];
				num_features += static_cast<int>(templ.features.size());

				if (templ.features.size() < 64) {
					similarity_64(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
				}
				else if (templ.features.size() < 8192) {
					similarity(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
				}
				else {
					CV_Error(Error::StsBadArg, "feature size too large");
				}
            }

            // Compare integer scores first. For templates below 64 features the
            // similarity map stays CV_8U, avoiding a full-image CV_16U conversion.
            const int offset = lowest_T / 2 + (lowest_T % 2 - 1);
            const int threshold_scaled =
                static_cast<int>(threshold * (4 * num_features) / 100.0f);
            const float score_scale = 100.0f / (4 * num_features);

            if (similarities.type() == CV_8U)
            {
                // Unsigned byte comparison: xor with 0x80 and use signed cmpgt.
                const __m256i sign_bit = _mm256_set1_epi8(static_cast<char>(0x80));
                const __m256i threshold_v = _mm256_set1_epi8(
                    static_cast<char>(static_cast<unsigned char>(threshold_scaled) ^ 0x80));

                for (int r = 0; r < similarities.rows; ++r)
                {
                    const uchar* row = similarities.ptr<uchar>(r);
                    int c = 0;
                    for (; c <= similarities.cols - 32; c += 32)
                    {
                        const __m256i raw_v =
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + c));
                        const __m256i cmp_v =
                            _mm256_cmpgt_epi8(_mm256_xor_si256(raw_v, sign_bit), threshold_v);
                        uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(cmp_v));

                        while (mask != 0)
                        {
                            const int bit = ctz_u32(mask);
                            const int idx = c + bit;
                            candidates.emplace_back(
                                idx * lowest_T + offset,
                                r * lowest_T + offset,
                                row[idx] * score_scale,
                                class_id,
                                static_cast<int>(template_id));
                            mask &= mask - 1;
                        }
                    }
                    for (; c < similarities.cols; ++c)
                    {
                        if (row[c] > threshold_scaled)
                        {
                            candidates.emplace_back(
                                c * lowest_T + offset,
                                r * lowest_T + offset,
                                row[c] * score_scale,
                                class_id,
                                static_cast<int>(template_id));
                        }
                    }
                }
            }
            else
            {
                CV_Assert(similarities.type() == CV_16U);
#if defined(__AVX2__) || defined(_M_AVX2)
                const __m256i sign_bit16 = _mm256_set1_epi16(
                    static_cast<short>(0x8000));
                const __m256i threshold_v16 = _mm256_set1_epi16(
                    static_cast<short>(
                        static_cast<unsigned short>(threshold_scaled) ^ 0x8000));
#endif
                for (int r = 0; r < similarities.rows; ++r)
                {
                    const ushort* row = similarities.ptr<ushort>(r);
                    int c = 0;
#if defined(__AVX2__) || defined(_M_AVX2)
                    for (; c <= similarities.cols - 16; c += 16)
                    {
                        const __m256i raw_v = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(row + c));
                        const __m256i cmp_v = _mm256_cmpgt_epi16(
                            _mm256_xor_si256(raw_v, sign_bit16), threshold_v16);
                        uint32_t lane_mask =
                            static_cast<uint32_t>(_mm256_movemask_epi8(cmp_v));

                        while (lane_mask != 0)
                        {
                            const int lane = ctz_u32(lane_mask) / 2;
                            const int idx = c + lane;
                            candidates.emplace_back(
                                idx * lowest_T + offset,
                                r * lowest_T + offset,
                                row[idx] * score_scale,
                                class_id,
                                static_cast<int>(template_id));
                            lane_mask &= ~(3u << (lane * 2));
                        }
                    }
#endif
                    for (; c < similarities.cols; ++c)
                    {
                        if (row[c] > threshold_scaled)
                        {
                            candidates.emplace_back(
                                c * lowest_T + offset,
                                r * lowest_T + offset,
                                row[c] * score_scale,
                                class_id,
                                static_cast<int>(template_id));
                        }
                    }
                }
            }
            /*const int offset = lowest_T / 2 + (lowest_T % 2 - 1);
            const int T = lowest_T;
            int threshold_scaled = (int)(threshold * 4 * num_features / 100.0 + 0.5);
            const __m256i v_threshold = _mm256_set1_epi16((short)threshold_scaled);
            for (int r = 0; r < similarities.rows; ++r)
            {
                const uint16_t* row = similarities.ptr<uint16_t>(r);
                int c = 0;
                const int cols = similarities.cols;
                for (; c <= cols - 16; c += 16)
                {
                    __m256i v_raw = _mm256_loadu_si256((__m256i*)(row + c));

                    __m256i v_cmp = _mm256_cmpgt_epi16(v_raw, v_threshold);

                    int mask32 = _mm256_movemask_epi8(v_cmp);
                    uint16_t mask = mask32 & 0xFFFF;

                    while (mask)
                    {
                        int bit = ctz_u16(mask);
                        int idx = c + bit;
                        int x = idx * T + offset;
                        int y = r * T + offset;
                        float score = (row[idx] * 100.f) / (4 * num_features);
                        candidates.emplace_back(Match(x, y, score, class_id, static_cast<int>(template_id)));
                        mask &= mask - 1;
                    }
                }
                for (; c < cols; ++c)
                {
                    int raw_score = row[c];
                    if (raw_score > threshold_scaled)
                    {
                        int x = c * T + offset;
                        int y = r * T + offset;
                        float score = (raw_score * 100.f) / (4 * num_features);
                        candidates.emplace_back(x, y, score, class_id, template_id);
                    }
                }
            }*/
        }


        // Locally refine each match by marching up the pyramid
        for (int l = pyramid_levels - 2; l >= 0; --l)
        {
            const std::vector<LinearMemories>& lms = lm_pyramid[l];
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
                Match& match2 = candidates[m];
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
                    const Template& templ = tp[start];
                    numFeatures += static_cast<int>(templ.features.size());

                    if (templ.features.size() < 64) {
                        similarityLocal_64(lms[0], templ, similarities2, size, T, Point(x, y));
                    }
                    else if (templ.features.size() < 8192) {
                        similarityLocal(lms[0], templ, similarities2, size, T, Point(x, y));
                    }
                    else {
                        CV_Error(Error::StsBadArg, "feature size too large");
                    }
                }

                // Find best local adjustment
                double best_raw_score = 0.0;
                Point best_position;
                minMaxLoc(similarities2, nullptr, &best_raw_score, nullptr, &best_position);
                const float best_score =
                    static_cast<float>(best_raw_score * 100.0 / (4 * numFeatures));
                const int best_r = best_position.y;
                const int best_c = best_position.x;
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

        //matches.insert(matches.end(), candidates.begin(), candidates.end());
        //将候选移动到线程局部容器(避免复制)
        if (!candidates.empty())
        {
            local_vec.insert(local_vec.end(),
                std::make_move_iterator(candidates.begin()),
                std::make_move_iterator(candidates.end()));
        }
    }
    //合并所有线程的结果到外部matches
	size_t total = 0;
	for (const auto& vec : thread_local_matches) 
        total += vec.size();
	matches.reserve(matches.size() + total); // Preserve matches from earlier classes.

	for (auto& vec : thread_local_matches)
	{
		if (!vec.empty())
		{
			matches.insert(matches.end(),
				std::make_move_iterator(vec.begin()),
				std::make_move_iterator(vec.end()));
		}
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
    CV_Assert(zero_id >= 0 && zero_id < (int)template_pyramids.size());

    template_pyramids.push_back(
        makeRotatedTemplatePyramid(template_pyramids[zero_id], theta, center, false));
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
    std::lock_guard<std::mutex> match_lock(match_mutex);
    class_templates.clear();
    match_quantizer.release();
    match_lm_pyramid.clear();
    match_sizes.clear();
    match_spread_quantized.clear();
    match_response_maps.clear();
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
