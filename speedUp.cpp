#include <stdio.h>
#include "speedUp.h"
#include <opencv2/opencv.hpp>
#include <future>

using namespace std;
using namespace cv;

void CalcGaussCof(float Radius, float& B0, float& B1, float& B2, float& B3)
{
	float Q, B;
	if (Radius >= 2.5)
		Q = (double)(0.98711 * Radius - 0.96330);                            //    对应论文公式11b
	else if ((Radius >= 0.5) && (Radius < 2.5))
		Q = (double)(3.97156 - 4.14554 * sqrt(1 - 0.26891 * Radius));
	else
		Q = (double)0.1147705018520355224609375;

	B = 1.57825 + 2.44413 * Q + 1.4281 * Q * Q + 0.422205 * Q * Q * Q;        //    对应论文公式8c
	B1 = 2.44413 * Q + 2.85619 * Q * Q + 1.26661 * Q * Q * Q;
	B2 = -1.4281 * Q * Q - 1.26661 * Q * Q * Q;
	B3 = 0.422205 * Q * Q * Q;

	B0 = 1.0 - (B1 + B2 + B3) / B;
	B1 = B1 / B;
	B2 = B2 / B;
	B3 = B3 / B;
}

void ConvertBGR8U2BGRAF(unsigned char* Src, float* Dest, int Width, int Height, int Stride)
{
	//#pragma omp parallel for
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char* LinePS = Src + Y * Stride;
		float* LinePD = Dest + Y * Width * 3;
		for (int X = 0; X < Width; X++, LinePS += 3, LinePD += 3)
		{
			LinePD[0] = LinePS[0];    LinePD[1] = LinePS[1];    LinePD[2] = LinePS[2];
		}
	}
}

void ConvertBGR8U2BGRAF_SSE_test(unsigned char* Src, float* Dest, int Width, int Height, int Stride, int DstRowStride, int DstColOffset, int start_row, int thread_stride)
{
	const int BlockSize = 4;
	int Block = (Width - 2) / BlockSize;
	__m128i Mask = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
	__m128i Zero = _mm_setzero_si128();
	for (int Y = start_row; Y < (start_row + thread_stride) && Y < Height; Y++)
	{
		unsigned char* LinePS = Src + Y * Stride;
		float* LinePD = Dest + Y * DstRowStride + DstColOffset;
		int X = 0;
		for (; X < Block * BlockSize; X += BlockSize, LinePS += BlockSize * 3, LinePD += BlockSize * 4)
		{
			__m128i SrcV = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)LinePS), Mask);
			__m128i Src16L = _mm_unpacklo_epi8(SrcV, Zero);
			__m128i Src16H = _mm_unpackhi_epi8(SrcV, Zero);
			_mm_store_ps(LinePD + 0, _mm_cvtepi32_ps(_mm_unpacklo_epi16(Src16L, Zero)));
			_mm_store_ps(LinePD + 4, _mm_cvtepi32_ps(_mm_unpackhi_epi16(Src16L, Zero)));
			_mm_store_ps(LinePD + 8, _mm_cvtepi32_ps(_mm_unpacklo_epi16(Src16H, Zero)));
			_mm_store_ps(LinePD + 12, _mm_cvtepi32_ps(_mm_unpackhi_epi16(Src16H, Zero)));
		}
		for (; X < Width; X++, LinePS += 3, LinePD += 4)
		{
			LinePD[0] = LinePS[0];
			LinePD[1] = LinePS[1];
			LinePD[2] = LinePS[2];
			LinePD[3] = 0;
		}
	}
}

void ConvertBGR8U2BGRAF_async(unsigned char* Src, float* Dest, int Width, int Height, int Stride, int DstRowStride, int DstColOffset)
{
	//Run                                         
	// Height >> 4,右移4位，相当于除以 16（2^4 = 16）,含义：图像高度分成多少个16像素高度的块. 为什么是16？因为，高度优化：16像素块大小通常是SIMD优化的友好单位（如SSE/AVX）
	//std::thread::hardware_concurrency()--返回系统真正支持的硬件并发线程数（CPU核心数或超线程数）,例如：4核8线程的CPU → 返回 8
	const int32_t hw_concur = std::min(100/*Height >> 4*/, static_cast<int32_t>(std::thread::hardware_concurrency()));

	std::vector<std::future<void>> fut(hw_concur);

	const int thread_stride = (Height - 1) / hw_concur + 1;
	int i = 0, start = 0;

	for (; i < std::min(Height, hw_concur); i++, start += thread_stride)
	{
		fut[i] = std::async(std::launch::async, [=]() {
			ConvertBGR8U2BGRAF_SSE_test(Src, Dest, Width, Height, Stride, DstRowStride, DstColOffset, start, thread_stride);
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

static inline unsigned char ClampToByteF(float v)
{
	int iv = (int)(v + 0.5f);
	if (iv < 0) return 0;
	if (iv > 255) return 255;
	return (unsigned char)iv;
}

void ConvertFloat2Gray8U_SSE_test(float* Src, unsigned char* Dest, int Width, int Height, int Stride, int extWidth, int start_row, int thread_stride)
{
	const int BlockSize = 4;
	int Block = Width / BlockSize;
	const __m128 RoundConst = _mm_set1_ps(0.5f);
	const __m128i Zero = _mm_setzero_si128();

	for (int Y = start_row; Y < (start_row + thread_stride) && Y < Height; Y++)
	{
		float* LinePS = Src + Y * extWidth;
		unsigned char* LinePD = Dest + Y * Stride;
		int X = 0;
		for (; X < Block * BlockSize; X += BlockSize, LinePS += BlockSize, LinePD += BlockSize)
		{
			__m128 v = _mm_loadu_ps(LinePS);
			__m128i i32 = _mm_cvtps_epi32(_mm_add_ps(v, RoundConst));
			__m128i i16 = _mm_packs_epi32(i32, Zero);
			__m128i i8 = _mm_packus_epi16(i16, Zero);
			*(int*)LinePD = _mm_cvtsi128_si32(i8);
		}
		for (; X < Width; X++, LinePS++, LinePD++)
		{
			*LinePD = ClampToByteF(*LinePS);
		}
	}
}

static void ConvertFloat2Gray8U_async(float* Src, unsigned char* Dest, int Width, int Height, int Stride, int extWidth)
{
	const int32_t hw_concur = std::min(100, static_cast<int32_t>(std::thread::hardware_concurrency()));
	std::vector<std::future<void>> fut(hw_concur);
	const int thread_stride = (Height - 1) / hw_concur + 1;
	int i = 0, start = 0;
	for (; i < std::min(Height, hw_concur); i++, start += thread_stride)
	{
		fut[i] = std::async(std::launch::async, [=]() {
			ConvertFloat2Gray8U_SSE_test(Src, Dest, Width, Height, Stride, extWidth, start, thread_stride);
			});
	}
	for (int j = 0; j < i; ++j)
	{
		if (fut[j].valid())
			fut[j].wait();
	}
}

void ConvertBGR8U2BGRAF_SSE(unsigned char* Src, float* Dest, int Width, int Height, int Stride, int DstRowStride, int DstColOffset)
{
	const int BlockSize = 4;
	int Block = (Width - 2) / BlockSize;
	__m128i Mask = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
	__m128i Zero = _mm_setzero_si128();
	for (int Y = 0; Y < Height; Y++) 
	{
		unsigned char* LinePS = Src + Y * Stride;
		float* LinePD = Dest + Y * DstRowStride + DstColOffset;
		int X = 0;
		for (; X < Block * BlockSize; X += BlockSize, LinePS += BlockSize * 3, LinePD += BlockSize * 4) 
		{
			__m128i SrcV = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)LinePS), Mask);
			__m128i Src16L = _mm_unpacklo_epi8(SrcV, Zero);
			__m128i Src16H = _mm_unpackhi_epi8(SrcV, Zero);
			_mm_store_ps(LinePD + 0, _mm_cvtepi32_ps(_mm_unpacklo_epi16(Src16L, Zero)));
			_mm_store_ps(LinePD + 4, _mm_cvtepi32_ps(_mm_unpackhi_epi16(Src16L, Zero)));
			_mm_store_ps(LinePD + 8, _mm_cvtepi32_ps(_mm_unpacklo_epi16(Src16H, Zero)));
			_mm_store_ps(LinePD + 12, _mm_cvtepi32_ps(_mm_unpackhi_epi16(Src16H, Zero)));
		}
		for (; X < Width; X++, LinePS += 3, LinePD += 4) 
		{
			LinePD[0] = LinePS[0];    
			LinePD[1] = LinePS[1];    
			LinePD[2] = LinePS[2];    
			LinePD[3] = 0;
		}
	}
}

//  定义函数指针类型
typedef void (*ProcessFunc)(float*, int, int, float, float, float, float, int, int);
void GaussBlur_asyncProcess(float* Data, int Width, int Height, float B0, float B1, float B2, float B3, int strideAll, ProcessFunc func)   // 新增参数
{
	const int32_t hw_concur = std::min(100/*Height >> 4*/, static_cast<int32_t>(std::thread::hardware_concurrency()));

	std::vector<std::future<void>> fut(hw_concur);

	const int thread_stride = (strideAll - 1) / hw_concur + 1;
	int i = 0, start = 0;

	for (; i < std::min(strideAll, hw_concur); i++, start += thread_stride)
	{
		fut[i] = std::async(std::launch::async, [=]() {
			func(Data, Width, Height, B0, B1, B2, B3, start, thread_stride);
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

void GaussBlur_asyncLeftToRight(float* Data, int Width, int Height, float B0, float B1, float B2, float B3)
{
	//Run                                         
	// Height >> 4,右移4位，相当于除以 16（2^4 = 16）,含义：图像高度分成多少个16像素高度的块. 为什么是16？因为，高度优化：16像素块大小通常是SIMD优化的友好单位（如SSE/AVX）
	//std::thread::hardware_concurrency()--返回系统真正支持的硬件并发线程数（CPU核心数或超线程数）,例如：4核8线程的CPU → 返回 8
	const int32_t hw_concur = std::min(100/*Height >> 4*/, static_cast<int32_t>(std::thread::hardware_concurrency()));

	std::vector<std::future<void>> fut(hw_concur);

	const int thread_stride = (Height - 1) / hw_concur + 1;
	int i = 0, start = 0;

	for (; i < std::min(Height, hw_concur); i++, start += thread_stride)
	{
		fut[i] = std::async(std::launch::async, [=]() {
			GaussBlurFromLeftToRight_test(Data, Width, Height, B0, B1, B2, B3, start, thread_stride);
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

void GaussBlurFromLeftToRight_test(float* Data, int Width, int Height, float B0, float B1, float B2, float B3, int start_row, int thread_stride)
{
	const __m128 CofB0 = _mm_set_ps(0, B0, B0, B0);
	const __m128 CofB1 = _mm_set_ps(0, B1, B1, B1);
	const __m128 CofB2 = _mm_set_ps(0, B2, B2, B2);
	const __m128 CofB3 = _mm_set_ps(0, B3, B3, B3);
	for (int Y = start_row; Y < (start_row + thread_stride) && Y < Height; Y++)
	{
		float* LinePD = Data + Y * Width * 4;
		__m128 V1 = _mm_set_ps(LinePD[3], LinePD[2], LinePD[1], LinePD[0]);
		__m128 V2 = V1;
		__m128 V3 = V1;
		for (int X = 0; X < Width; X++, LinePD += 4)
		{
			__m128 V0 = _mm_load_ps(LinePD);
			__m128 V01 = _mm_add_ps(_mm_mul_ps(CofB0, V0), _mm_mul_ps(CofB1, V1));
			__m128 V23 = _mm_add_ps(_mm_mul_ps(CofB2, V2), _mm_mul_ps(CofB3, V3));
			__m128 V = _mm_add_ps(V01, V23);
			V3 = V2;
			V2 = V1;
			V1 = V;
			_mm_store_ps(LinePD, V);
		}
	}
}

void GaussBlurFromLeftToRight(float* Data, int Width, int Height, float B0, float B1, float B2, float B3)
{
	//#pragma omp parallel for
	for (int Y = 0; Y < Height; Y++)
	{
		float* LinePD = Data + Y * Width * 3;
		//w[n-1], w[n-2], w[n-3]
		float BS1 = LinePD[0], BS2 = LinePD[0], BS3 = LinePD[0]; //边缘处使用重复像素的方案
		float GS1 = LinePD[1], GS2 = LinePD[1], GS3 = LinePD[1];
		float RS1 = LinePD[2], RS2 = LinePD[2], RS3 = LinePD[2];
		for (int X = 0; X < Width; X++, LinePD += 3)
		{
			LinePD[0] = LinePD[0] * B0 + BS1 * B1 + BS2 * B2 + BS3 * B3;
			LinePD[1] = LinePD[1] * B0 + GS1 * B1 + GS2 * B2 + GS3 * B3;         // 进行顺向迭代
			LinePD[2] = LinePD[2] * B0 + RS1 * B1 + RS2 * B2 + RS3 * B3;
			BS3 = BS2, BS2 = BS1, BS1 = LinePD[0];
			GS3 = GS2, GS2 = GS1, GS1 = LinePD[1];
			RS3 = RS2, RS2 = RS1, RS1 = LinePD[2];
		}
	}
}

void GaussBlurFromLeftToRight_SSE(float* Data, int Width, int Height, float B0, float B1, float B2, float B3)
{
	const __m128 CofB0 = _mm_set_ps(0, B0, B0, B0);
	const __m128 CofB1 = _mm_set_ps(0, B1, B1, B1);
	const __m128 CofB2 = _mm_set_ps(0, B2, B2, B2);
	const __m128 CofB3 = _mm_set_ps(0, B3, B3, B3);
	for (int Y = 0; Y < Height; Y++)
	{
		float* LinePD = Data + Y * Width * 4;
		__m128 V1 = _mm_set_ps(LinePD[3], LinePD[2], LinePD[1], LinePD[0]);
		__m128 V2 = V1;
		__m128 V3 = V1;
		for (int X = 0; X < Width; X++, LinePD += 4)
		{
			__m128 V0 = _mm_load_ps(LinePD);
			__m128 V01 = _mm_add_ps(_mm_mul_ps(CofB0, V0), _mm_mul_ps(CofB1, V1));
			__m128 V23 = _mm_add_ps(_mm_mul_ps(CofB2, V2), _mm_mul_ps(CofB3, V3));
			__m128 V = _mm_add_ps(V01, V23);
			V3 = V2;
			V2 = V1;
			V1 = V;
			_mm_store_ps(LinePD, V);
		}
	}
}

void GaussBlur_asyncRightToLeft(float* Data, int Width, int Height, float B0, float B1, float B2, float B3)
{
	//Run                                         
	// Height >> 4,右移4位，相当于除以 16（2^4 = 16）,含义：图像高度分成多少个16像素高度的块. 为什么是16？因为，高度优化：16像素块大小通常是SIMD优化的友好单位（如SSE/AVX）
	//std::thread::hardware_concurrency()--返回系统真正支持的硬件并发线程数（CPU核心数或超线程数）,例如：4核8线程的CPU → 返回 8
	const int32_t hw_concur = std::min(100/*Height >> 4*/, static_cast<int32_t>(std::thread::hardware_concurrency()));

	std::vector<std::future<void>> fut(hw_concur);

	const int thread_stride = (Height - 1) / hw_concur + 1;
	int i = 0, start = 0;

	for (; i < std::min(Height, hw_concur); i++, start += thread_stride)
	{
		fut[i] = std::async(std::launch::async, [=]() {
			GaussBlurFromRightToLeft_test(Data, Width, Height, B0, B1, B2, B3, start, thread_stride);
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

void GaussBlurFromRightToLeft_test(float* Data, int Width, int Height, float B0, float B1, float B2, float B3, int start_row, int thread_stride)
{
	const __m128 CofB0 = _mm_set_ps(0, B0, B0, B0);
	const __m128 CofB1 = _mm_set_ps(0, B1, B1, B1);
	const __m128 CofB2 = _mm_set_ps(0, B2, B2, B2);
	const __m128 CofB3 = _mm_set_ps(0, B3, B3, B3);
	for (int Y = start_row; Y < (start_row + thread_stride) && Y < Height; Y++)
	{
		float* LinePD = Data + Y * Width * 4 + (Width - 1) * 4;

		__m128 last_pixel = _mm_load_ps(LinePD);
		__m128 V1 = last_pixel;
		__m128 V2 = last_pixel;
		__m128 V3 = last_pixel;

		for (int X = Width - 1; X >= 0; X--, LinePD -= 4)
		{
			// w[n] = B0*x[n] + B1*w[n+1] + B2*w[n+2] + B3*w[n+3]
			__m128 V0 = _mm_load_ps(LinePD);
			__m128 V = _mm_add_ps(
				_mm_add_ps(_mm_mul_ps(CofB0, V0), _mm_mul_ps(CofB1, V1)),
				_mm_add_ps(_mm_mul_ps(CofB2, V2), _mm_mul_ps(CofB3, V3))
			);
			V3 = V2;
			V2 = V1;
			V1 = V;
			_mm_store_ps(LinePD, V);
		}
	}
}

void GaussBlurFromRightToLeft(float* Data, int Width, int Height, float B0, float B1, float B2, float B3) {
	for (int Y = 0; Y < Height; Y++) {
		//w[n+1], w[n+2], w[n+3]
		float* LinePD = Data + Y * Width * 3 + (Width * 3);
		float BS1 = LinePD[0], BS2 = LinePD[0], BS3 = LinePD[0]; //边缘处使用重复像素的方案
		float GS1 = LinePD[1], GS2 = LinePD[1], GS3 = LinePD[1];
		float RS1 = LinePD[2], RS2 = LinePD[2], RS3 = LinePD[2];
		for (int X = Width - 1; X >= 0; X--, LinePD -= 3)
		{
			LinePD[0] = LinePD[0] * B0 + BS3 * B1 + BS2 * B2 + BS1 * B3;
			LinePD[1] = LinePD[1] * B0 + GS3 * B1 + GS2 * B2 + GS1 * B3;         // 进行反向迭代
			LinePD[2] = LinePD[2] * B0 + RS3 * B1 + RS2 * B2 + RS1 * B3;
			BS1 = BS2, BS2 = BS3, BS3 = LinePD[0];
			GS1 = GS2, GS2 = GS3, GS3 = LinePD[1];
			RS1 = RS2, RS2 = RS3, RS3 = LinePD[2];
		}
	}
}

void GaussBlurFromRightToLeft_SSE(float* Data, int Width, int Height, float B0, float B1, float B2, float B3) {
	const __m128 CofB0 = _mm_set_ps(0, B0, B0, B0);
	const __m128 CofB1 = _mm_set_ps(0, B1, B1, B1);
	const __m128 CofB2 = _mm_set_ps(0, B2, B2, B2);
	const __m128 CofB3 = _mm_set_ps(0, B3, B3, B3);
	for (int Y = 0; Y < Height; Y++)
	{
		float* LinePD = Data + Y * Width * 4 + (Width-1) * 4;
		
		__m128 last_pixel = _mm_load_ps(LinePD);
		__m128 V1 = last_pixel;
		__m128 V2 = last_pixel;
		__m128 V3 = last_pixel;

		for (int X = Width - 1; X >= 0; X--, LinePD -= 4)
		{
			// w[n] = B0*x[n] + B1*w[n+1] + B2*w[n+2] + B3*w[n+3]
			__m128 V0 = _mm_load_ps(LinePD);
			__m128 V = _mm_add_ps(
				_mm_add_ps(_mm_mul_ps(CofB0, V0), _mm_mul_ps(CofB1, V1)),
				_mm_add_ps(_mm_mul_ps(CofB2, V2), _mm_mul_ps(CofB3, V3))
			);
			V3 = V2;
			V2 = V1;
			V1 = V;
			_mm_store_ps(LinePD, V);
		}
	}
}

void GaussBlur_asyncTopToBottom(float* Data, int Width, int Height, float B0, float B1, float B2, float B3)
{
	//Run                                         
	// Height >> 4,右移4位，相当于除以 16（2^4 = 16）,含义：图像高度分成多少个16像素高度的块. 为什么是16？因为，高度优化：16像素块大小通常是SIMD优化的友好单位（如SSE/AVX）
	//std::thread::hardware_concurrency()--返回系统真正支持的硬件并发线程数（CPU核心数或超线程数）,例如：4核8线程的CPU → 返回 8
	const int32_t hw_concur = std::min(Width >> 4, static_cast<int32_t>(std::thread::hardware_concurrency()));

	std::vector<std::future<void>> fut(hw_concur);

	const int thread_stride = (Width - 1) / hw_concur + 1;
	int i = 0, start = 0;

	for (; i < std::min(Width, hw_concur); i++, start += thread_stride)
	{
		fut[i] = std::async(std::launch::async, [=]() {
			GaussBlurFromTopToBottom_test(Data, Width, Height, B0, B1, B2, B3, start, thread_stride);
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

void GaussBlurFromTopToBottom_test(float* Data, int Width, int Height, float B0, float B1, float B2, float B3, int start_col, int thread_stride)
{
	const  __m128 CofB0 = _mm_set_ps(0, B0, B0, B0);
	const  __m128 CofB1 = _mm_set_ps(0, B1, B1, B1);
	const  __m128 CofB2 = _mm_set_ps(0, B2, B2, B2);
	const  __m128 CofB3 = _mm_set_ps(0, B3, B3, B3);
	for (int Y = 0; Y < Height - 4; Y++)
	{
		int base = (Y + 3) * Width * 4;
		float* LinePS0 = Data + base;
		float* LinePS1 = Data + base - Width * 4;
		float* LinePS2 = Data + base - 2 * Width * 4;
		float* LinePS3 = Data + base - 3 * Width * 4;
		for (int X = start_col * 4; X < (start_col + thread_stride) * 4 && X < Width * 4; X += 4) {
			__m128 V0 = _mm_load_ps(LinePS0 + X);
			__m128 V1 = _mm_load_ps(LinePS1 + X);
			__m128 V2 = _mm_load_ps(LinePS2 + X);
			__m128 V3 = _mm_load_ps(LinePS3 + X);

			__m128 V = _mm_add_ps(
				_mm_add_ps(_mm_mul_ps(CofB0, V0), _mm_mul_ps(CofB1, V1)),
				_mm_add_ps(_mm_mul_ps(CofB2, V2), _mm_mul_ps(CofB3, V3))
			);
			_mm_store_ps(LinePS0 + X, V);
		}
	}
}

//w[n] w[n-1], w[n-2], w[n-3]
void GaussBlurFromTopToBottom(float* Data, int Width, int Height, float B0, float B1, float B2, float B3)
{
	for (int Y = 0; Y < Height; Y++)
	{
		float* LinePD3 = Data + (Y + 0) * Width * 3;
		float* LinePD2 = Data + (Y + 1) * Width * 3;
		float* LinePD1 = Data + (Y + 2) * Width * 3;
		float* LinePD0 = Data + (Y + 3) * Width * 3;
		for (int X = 0; X < Width; X++, LinePD0 += 3, LinePD1 += 3, LinePD2 += 3, LinePD3 += 3)
		{
			LinePD0[0] = LinePD0[0] * B0 + LinePD1[0] * B1 + LinePD2[0] * B2 + LinePD3[0] * B3;
			LinePD0[1] = LinePD0[1] * B0 + LinePD1[1] * B1 + LinePD2[1] * B2 + LinePD3[1] * B3;
			LinePD0[2] = LinePD0[2] * B0 + LinePD1[2] * B1 + LinePD2[2] * B2 + LinePD3[2] * B3;
		}
	}
}

void GaussBlurFromTopToBottom_SSE(float* Data, int Width, int Height, float B0, float B1, float B2, float B3) {
	const  __m128 CofB0 = _mm_set_ps(0, B0, B0, B0);
	const  __m128 CofB1 = _mm_set_ps(0, B1, B1, B1);
	const  __m128 CofB2 = _mm_set_ps(0, B2, B2, B2);
	const  __m128 CofB3 = _mm_set_ps(0, B3, B3, B3);
	for (int Y = 0; Y < Height - 4; Y++)
	{
		int base = (Y + 3) * Width * 4;
		float* LinePS0 = Data + base;
		float* LinePS1 = Data + base - Width * 4;
		float* LinePS2 = Data + base - 2 * Width * 4;
		float* LinePS3 = Data + base - 3 * Width * 4;
		for (int X = 0; X < Width * 4; X += 4) {
			__m128 V0 = _mm_load_ps(LinePS0 + X);
			__m128 V1 = _mm_load_ps(LinePS1 + X);
			__m128 V2 = _mm_load_ps(LinePS2 + X);
			__m128 V3 = _mm_load_ps(LinePS3 + X);

			__m128 V = _mm_add_ps(
				_mm_add_ps(_mm_mul_ps(CofB0, V0), _mm_mul_ps(CofB1, V1)),
				_mm_add_ps(_mm_mul_ps(CofB2, V2), _mm_mul_ps(CofB3, V3))
			);
			_mm_store_ps(LinePS0 + X, V);
		}
	}
}

void GaussBlur_asyncBottomToTop(float* Data, int Width, int Height, float B0, float B1, float B2, float B3)
{
	//Run                                         
	// Height >> 4,右移4位，相当于除以 16（2^4 = 16）,含义：图像高度分成多少个16像素高度的块. 为什么是16？因为，高度优化：16像素块大小通常是SIMD优化的友好单位（如SSE/AVX）
	//std::thread::hardware_concurrency()--返回系统真正支持的硬件并发线程数（CPU核心数或超线程数）,例如：4核8线程的CPU → 返回 8
	const int32_t hw_concur = std::min(Width >> 4, static_cast<int32_t>(std::thread::hardware_concurrency()));

	std::vector<std::future<void>> fut(hw_concur);

	const int thread_stride = (Width - 1) / hw_concur + 1;
	int i = 0, start = 0;

	for (; i < std::min(Width, hw_concur); i++, start += thread_stride)
	{
		fut[i] = std::async(std::launch::async, [=]() {
			GaussBlurFromBottomToTop_test(Data, Width, Height, B0, B1, B2, B3, start, thread_stride);
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

void GaussBlurFromBottomToTop_test(float* Data, int Width, int Height, float B0, float B1, float B2, float B3, int start_col, int thread_stride)
{
	const  __m128 CofB0 = _mm_set_ps(0, B0, B0, B0);
	const  __m128 CofB1 = _mm_set_ps(0, B1, B1, B1);
	const  __m128 CofB2 = _mm_set_ps(0, B2, B2, B2);
	const  __m128 CofB3 = _mm_set_ps(0, B3, B3, B3);
	for (int Y = Height - 1 - 3; Y >= 0; Y--)
	{
		int base = Y * Width * 4;
		float* LinePS0 = Data + base;          // w[n]
		float* LinePS1 = Data + base + Width * 4; // w[n+1]
		float* LinePS2 = Data + base + 2 * Width * 4; // w[n+2]
		float* LinePS3 = Data + base + 3 * Width * 4; // w[n+3]

		//float* LinePS3 = Data + (Y + 3) * Width * 4;
		//float* LinePS2 = Data + (Y + 2) * Width * 4;
		//float* LinePS1 = Data + (Y + 1) * Width * 4;
		//float* LinePS0 = Data + (Y + 0) * Width * 4;

		for (int X = start_col * 4; X < (start_col + thread_stride) * 4 && X < Width * 4; X += 4)
		{
			__m128 V0 = _mm_load_ps(LinePS0 + X);
			__m128 V1 = _mm_load_ps(LinePS1 + X);
			__m128 V2 = _mm_load_ps(LinePS2 + X);
			__m128 V3 = _mm_load_ps(LinePS3 + X);

			__m128 V01 = _mm_add_ps(_mm_mul_ps(CofB0, V0), _mm_mul_ps(CofB1, V1));
			__m128 V23 = _mm_add_ps(_mm_mul_ps(CofB2, V2), _mm_mul_ps(CofB3, V3));
			_mm_store_ps(LinePS0 + X, _mm_add_ps(V01, V23));
		}
	}
}

//w[n] w[n+1], w[n+2], w[n+3]
void GaussBlurFromBottomToTop(float* Data, int Width, int Height, float B0, float B1, float B2, float B3) {
	for (int Y = Height - 1; Y >= 0; Y--) {
		float* LinePD3 = Data + (Y + 3) * Width * 3;
		float* LinePD2 = Data + (Y + 2) * Width * 3;
		float* LinePD1 = Data + (Y + 1) * Width * 3;
		float* LinePD0 = Data + (Y + 0) * Width * 3;
		for (int X = 0; X < Width; X++, LinePD0 += 3, LinePD1 += 3, LinePD2 += 3, LinePD3 += 3) {
			LinePD0[0] = LinePD0[0] * B0 + LinePD1[0] * B1 + LinePD2[0] * B2 + LinePD3[0] * B3;
			LinePD0[1] = LinePD0[1] * B0 + LinePD1[1] * B1 + LinePD2[1] * B2 + LinePD3[1] * B3;
			LinePD0[2] = LinePD0[2] * B0 + LinePD1[2] * B1 + LinePD2[2] * B2 + LinePD3[2] * B3;
		}
	}
}

void GaussBlurFromBottomToTop_SSE(float* Data, int Width, int Height, float B0, float B1, float B2, float B3) {
	const  __m128 CofB0 = _mm_set_ps(0, B0, B0, B0);
	const  __m128 CofB1 = _mm_set_ps(0, B1, B1, B1);
	const  __m128 CofB2 = _mm_set_ps(0, B2, B2, B2);
	const  __m128 CofB3 = _mm_set_ps(0, B3, B3, B3);
	for (int Y = Height - 1 - 3; Y >= 0; Y--) 
	{
		int base = Y * Width * 4;
		float* LinePS0 = Data + base;          // w[n]
		float* LinePS1 = Data + base + Width * 4; // w[n+1]
		float* LinePS2 = Data + base + 2 * Width * 4; // w[n+2]
		float* LinePS3 = Data + base + 3 * Width * 4; // w[n+3]

		//float* LinePS3 = Data + (Y + 3) * Width * 4;
		//float* LinePS2 = Data + (Y + 2) * Width * 4;
		//float* LinePS1 = Data + (Y + 1) * Width * 4;
		//float* LinePS0 = Data + (Y + 0) * Width * 4;

		for (int X = 0; X < Width * 4; X += 4) 
		{
			__m128 V0 = _mm_load_ps(LinePS0 + X);
			__m128 V1 = _mm_load_ps(LinePS1 + X);
			__m128 V2 = _mm_load_ps(LinePS2 + X);
			__m128 V3 = _mm_load_ps(LinePS3 + X);

			__m128 V01 = _mm_add_ps(_mm_mul_ps(CofB0, V0), _mm_mul_ps(CofB1, V1));
			__m128 V23 = _mm_add_ps(_mm_mul_ps(CofB2, V2), _mm_mul_ps(CofB3, V3));
			_mm_store_ps(LinePS0 + X, _mm_add_ps(V01, V23));
		}
	}
}

void ConvertBGRAF2BGR8U(float* Src, unsigned char* Dest, int Width, int Height, int Stride)
{
	//#pragma omp parallel for
	for (int Y = 0; Y < Height; Y++)
	{
		float* LinePS = Src + Y * Width * 3;
		unsigned char* LinePD = Dest + Y * Stride;
		for (int X = 0; X < Width; X++, LinePS += 3, LinePD += 3)
		{
			LinePD[0] = LinePS[0];    LinePD[1] = LinePS[1];    LinePD[2] = LinePS[2];
		}
	}
}

void ConvertBGRAF2BGR8U_2(float* Src, unsigned char* Dest, int Width, int Height, int Stride) {
	for (int Y = 0; Y < Height; Y++) {
		float* LinePS = Src + Y * Width * 4;
		unsigned char* LinePD = Dest + Y * Stride;
		for (int X = 0; X < Width; X++, LinePS += 4, LinePD += 3) {
			LinePD[0] = LinePS[0]; 
			LinePD[1] = LinePS[1];    
			LinePD[2] = LinePS[2];
		}
	}
}
void ConvertBGRAF2BGR8U_SSE(float* Src, unsigned char* Dest, int Width, int Height, int Stride) {
	const int BlockSize = 4;
	int Block = (Width - 2) / BlockSize;
	//__m128i Mask = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15);
	__m128i MaskB = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
	__m128i MaskG = _mm_setr_epi8(1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
	__m128i MaskR = _mm_setr_epi8(2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
	__m128i Zero = _mm_setzero_si128();
	for (int Y = 0; Y < Height; Y++) {
		float* LinePS = Src + Y * Width * 4;
		unsigned char* LinePD = Dest + Y * Stride;
		int X = 0;
		for (; X < Block * BlockSize; X += BlockSize, LinePS += BlockSize * 4, LinePD += BlockSize * 3) {
			__m128i SrcV = _mm_loadu_si128((const __m128i*)LinePS);
			__m128i B = _mm_shuffle_epi8(SrcV, MaskB);
			__m128i G = _mm_shuffle_epi8(SrcV, MaskG);
			__m128i R = _mm_shuffle_epi8(SrcV, MaskR);
			__m128i Ans1 = Zero, Ans2 = Zero, Ans3 = Zero;
			Ans1 = _mm_or_si128(Ans1, _mm_shuffle_epi8(B, _mm_setr_epi8(0, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));
			Ans1 = _mm_or_si128(Ans1, _mm_shuffle_epi8(G, _mm_setr_epi8(-1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));
			Ans1 = _mm_or_si128(Ans1, _mm_shuffle_epi8(R, _mm_setr_epi8(-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));

			Ans2 = _mm_or_si128(Ans2, _mm_shuffle_epi8(B, _mm_setr_epi8(-1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));
			Ans2 = _mm_or_si128(Ans2, _mm_shuffle_epi8(G, _mm_setr_epi8(1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));
			Ans2 = _mm_or_si128(Ans2, _mm_shuffle_epi8(R, _mm_setr_epi8(-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));

			Ans3 = _mm_or_si128(Ans3, _mm_shuffle_epi8(B, _mm_setr_epi8(-1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));
			Ans3 = _mm_or_si128(Ans3, _mm_shuffle_epi8(G, _mm_setr_epi8(-1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));
			Ans3 = _mm_or_si128(Ans3, _mm_shuffle_epi8(R, _mm_setr_epi8(2, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)));

			_mm_storeu_si128((__m128i*)(LinePD + 0), Ans1);
			_mm_storeu_si128((__m128i*)(LinePD + 4), Ans2);
			_mm_storeu_si128((__m128i*)(LinePD + 8), Ans3);
		}
		for (; X < Width; X++, LinePS += 4, LinePD += 3) {
			LinePD[0] = LinePS[0]; LinePD[1] = LinePS[1]; LinePD[2] = LinePS[2];
		}
	}
}

void ConvertBGRAF2BGR8U_async(float* Src, unsigned char* Dest, int Width, int Height, int Stride, int extWidth, int extHeight)
{
	//Run                                         
	// Height >> 4,右移4位，相当于除以 16（2^4 = 16）,含义：图像高度分成多少个16像素高度的块. 为什么是16？因为，高度优化：16像素块大小通常是SIMD优化的友好单位（如SSE/AVX）
	//std::thread::hardware_concurrency()--返回系统真正支持的硬件并发线程数（CPU核心数或超线程数）,例如：4核8线程的CPU → 返回 8
	const int32_t hw_concur = std::min(100/*Height >> 4*/, static_cast<int32_t>(std::thread::hardware_concurrency()));

	std::vector<std::future<void>> fut(hw_concur);

	const int thread_stride = (Height - 1) / hw_concur + 1;
	int i = 0, start = 0;

	for (; i < std::min(Height, hw_concur); i++, start += thread_stride)
	{
		fut[i] = std::async(std::launch::async, [=]() {
			ConvertBGRAF2BGR8U_SSE_Float_Correct_test(Src, Dest, Width, Height, Stride, extWidth, extHeight, start, thread_stride);
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

void ConvertBGRAF2BGR8U_SSE_Float_Correct_test(float* Src, unsigned char* Dest, int Width, int Height, int Stride, int extWidth, int extHeight, int start_row, int thread_stride)
{
	const int BlockSize = 4;
	int Block = Width / BlockSize;
	const __m128 RoundConst = _mm_set1_ps(0.5f);
	const __m128i Zero = _mm_setzero_si128();

	const __m128i PackMask = _mm_setr_epi8(
		0, 4, 8,        // 第0组：B0, G0, R0
		1, 5, 9,        // 第1组：B1, G1, R1
		2, 6, 10,       // 第2组：B2, G2, R2
		3, 7, 11,       // 第3组：B3, G3, R3
		-1, -1, -1, -1
	);

	for (int Y = start_row; Y < (start_row + thread_stride) && Y < Height; Y++) {
		float* LinePS = Src + Y * extWidth * 4;
		unsigned char* LinePD = Dest + Y * Stride;
		int X = 0;

		for (; X < Block * BlockSize; X += BlockSize) {
			__m128 v0 = _mm_load_ps(LinePS + 0);
			__m128 v1 = _mm_load_ps(LinePS + 4);
			__m128 v2 = _mm_load_ps(LinePS + 8);
			__m128 v3 = _mm_load_ps(LinePS + 12);

			__m128 t0 = _mm_unpacklo_ps(v0, v2);
			__m128 t1 = _mm_unpackhi_ps(v0, v2);
			__m128 t2 = _mm_unpacklo_ps(v1, v3);
			__m128 t3 = _mm_unpackhi_ps(v1, v3);

			__m128 B_float = _mm_unpacklo_ps(t0, t2);
			__m128 G_float = _mm_unpackhi_ps(t0, t2);
			__m128 R_float = _mm_unpacklo_ps(t1, t3);

			// float → int32 → int8
			__m128i B_int = _mm_cvtps_epi32(_mm_add_ps(B_float, RoundConst));
			__m128i G_int = _mm_cvtps_epi32(_mm_add_ps(G_float, RoundConst));
			__m128i R_int = _mm_cvtps_epi32(_mm_add_ps(R_float, RoundConst));

			__m128i B_16 = _mm_packs_epi32(B_int, Zero);
			__m128i G_16 = _mm_packs_epi32(G_int, Zero);
			__m128i R_16 = _mm_packs_epi32(R_int, Zero);

			__m128i B_8 = _mm_packus_epi16(B_16, Zero);
			__m128i G_8 = _mm_packus_epi16(G_16, Zero);
			__m128i R_8 = _mm_packus_epi16(R_16, Zero);


			__m128i BG = _mm_unpacklo_epi8(B_8, G_8);
			// BG = [B0, G0, B1, G1, B2, G2, B3, G3, 0, 0, 0, 0, 0, 0, 0, 0]


			__m128i BGR = _mm_setzero_si128();


			__m128i R_expanded = _mm_shuffle_epi8(R_8, _mm_setr_epi8(
				0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, -1, -1, -1, -1
			));
			// R_expanded = [R0, 0, 0, R1, 0, 0, R2, 0, 0, R3, 0, 0, 0, 0, 0, 0]



			int b_vals = _mm_cvtsi128_si32(B_8);
			int g_vals = _mm_cvtsi128_si32(G_8);
			int r_vals = _mm_cvtsi128_si32(R_8);

			LinePD[0] = (b_vals >> 0) & 0xFF;
			LinePD[1] = (g_vals >> 0) & 0xFF;
			LinePD[2] = (r_vals >> 0) & 0xFF;
			LinePD[3] = (b_vals >> 8) & 0xFF;
			LinePD[4] = (g_vals >> 8) & 0xFF;
			LinePD[5] = (r_vals >> 8) & 0xFF;
			LinePD[6] = (b_vals >> 16) & 0xFF;
			LinePD[7] = (g_vals >> 16) & 0xFF;
			LinePD[8] = (r_vals >> 16) & 0xFF;
			LinePD[9] = (b_vals >> 24) & 0xFF;
			LinePD[10] = (g_vals >> 24) & 0xFF;
			LinePD[11] = (r_vals >> 24) & 0xFF;

			LinePS += 16;
			LinePD += 12;
		}

		for (; X < Width; X++, LinePS += 4, LinePD += 3) {
			LinePD[0] = (unsigned char)(LinePS[0] + 0.5f);
			LinePD[1] = (unsigned char)(LinePS[1] + 0.5f);
			LinePD[2] = (unsigned char)(LinePS[2] + 0.5f);
		}
	}
}

void ConvertBGRAF2BGR8U_SSE_Float_Correct(float* Src, unsigned char* Dest, int Width, int Height, int Stride, int extWidth, int extHeight) 
{
	const int BlockSize = 4;
	int Block = Width / BlockSize;
	const __m128 RoundConst = _mm_set1_ps(0.5f);
	const __m128i Zero = _mm_setzero_si128();

	const __m128i PackMask = _mm_setr_epi8(
		0, 4, 8,        // 第0组：B0, G0, R0
		1, 5, 9,        // 第1组：B1, G1, R1
		2, 6, 10,       // 第2组：B2, G2, R2
		3, 7, 11,       // 第3组：B3, G3, R3
		-1, -1, -1, -1
	);

	for (int Y = 0; Y < Height; Y++) {
		float* LinePS = Src + Y * extWidth * 4;
		unsigned char* LinePD = Dest + Y * Stride;
		int X = 0;

		for (; X < Block * BlockSize; X += BlockSize) {
			__m128 v0 = _mm_load_ps(LinePS + 0);
			__m128 v1 = _mm_load_ps(LinePS + 4);
			__m128 v2 = _mm_load_ps(LinePS + 8);
			__m128 v3 = _mm_load_ps(LinePS + 12);

			__m128 t0 = _mm_unpacklo_ps(v0, v2);
			__m128 t1 = _mm_unpackhi_ps(v0, v2);
			__m128 t2 = _mm_unpacklo_ps(v1, v3);
			__m128 t3 = _mm_unpackhi_ps(v1, v3);

			__m128 B_float = _mm_unpacklo_ps(t0, t2);
			__m128 G_float = _mm_unpackhi_ps(t0, t2);
			__m128 R_float = _mm_unpacklo_ps(t1, t3);

			// float → int32 → int8
			__m128i B_int = _mm_cvtps_epi32(_mm_add_ps(B_float, RoundConst));
			__m128i G_int = _mm_cvtps_epi32(_mm_add_ps(G_float, RoundConst));
			__m128i R_int = _mm_cvtps_epi32(_mm_add_ps(R_float, RoundConst));

			__m128i B_16 = _mm_packs_epi32(B_int, Zero);
			__m128i G_16 = _mm_packs_epi32(G_int, Zero);
			__m128i R_16 = _mm_packs_epi32(R_int, Zero);

			__m128i B_8 = _mm_packus_epi16(B_16, Zero);
			__m128i G_8 = _mm_packus_epi16(G_16, Zero);
			__m128i R_8 = _mm_packus_epi16(R_16, Zero);


			__m128i BG = _mm_unpacklo_epi8(B_8, G_8);
			// BG = [B0, G0, B1, G1, B2, G2, B3, G3, 0, 0, 0, 0, 0, 0, 0, 0]


			__m128i BGR = _mm_setzero_si128();


			__m128i R_expanded = _mm_shuffle_epi8(R_8, _mm_setr_epi8(
				0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, -1, -1, -1, -1
			));
			// R_expanded = [R0, 0, 0, R1, 0, 0, R2, 0, 0, R3, 0, 0, 0, 0, 0, 0]



			int b_vals = _mm_cvtsi128_si32(B_8);
			int g_vals = _mm_cvtsi128_si32(G_8);
			int r_vals = _mm_cvtsi128_si32(R_8);

			LinePD[0] = (b_vals >> 0) & 0xFF;
			LinePD[1] = (g_vals >> 0) & 0xFF;
			LinePD[2] = (r_vals >> 0) & 0xFF;
			LinePD[3] = (b_vals >> 8) & 0xFF;
			LinePD[4] = (g_vals >> 8) & 0xFF;
			LinePD[5] = (r_vals >> 8) & 0xFF;
			LinePD[6] = (b_vals >> 16) & 0xFF;
			LinePD[7] = (g_vals >> 16) & 0xFF;
			LinePD[8] = (r_vals >> 16) & 0xFF;
			LinePD[9] = (b_vals >> 24) & 0xFF;
			LinePD[10] = (g_vals >> 24) & 0xFF;
			LinePD[11] = (r_vals >> 24) & 0xFF;

			LinePS += 16;
			LinePD += 12;
		}

		for (; X < Width; X++, LinePS += 4, LinePD += 3) {
			LinePD[0] = (unsigned char)(LinePS[0] + 0.5f);
			LinePD[1] = (unsigned char)(LinePS[1] + 0.5f);
			LinePD[2] = (unsigned char)(LinePS[2] + 0.5f);
		}
	}
}

void GaussBlur(unsigned char* Src, unsigned char* Dest, int Width, int Height, int Stride, float Radius)
{
	float B0, B1, B2, B3;
	float* Buffer = (float*)malloc(Width * (Height + 6) * sizeof(float) * 3);
	CalcGaussCof(Radius, B0, B1, B2, B3);
	ConvertBGR8U2BGRAF(Src, Buffer + 3 * Width * 3, Width, Height, Stride);
	GaussBlurFromLeftToRight(Buffer + 3 * Width * 3, Width, Height, B0, B1, B2, B3);
	GaussBlurFromRightToLeft(Buffer + 3 * Width * 3, Width, Height, B0, B1, B2, B3);

	memcpy(Buffer + 0 * Width * 3, Buffer + 3 * Width * 3, Width * 3 * sizeof(float));
	memcpy(Buffer + 1 * Width * 3, Buffer + 3 * Width * 3, Width * 3 * sizeof(float));
	memcpy(Buffer + 2 * Width * 3, Buffer + 3 * Width * 3, Width * 3 * sizeof(float));

	GaussBlurFromTopToBottom(Buffer, Width, Height, B0, B1, B2, B3);

	memcpy(Buffer + (Height + 3) * Width * 3, Buffer + (Height + 2) * Width * 3, Width * 3 * sizeof(float));
	memcpy(Buffer + (Height + 4) * Width * 3, Buffer + (Height + 2) * Width * 3, Width * 3 * sizeof(float));
	memcpy(Buffer + (Height + 5) * Width * 3, Buffer + (Height + 2) * Width * 3, Width * 3 * sizeof(float));

	GaussBlurFromBottomToTop(Buffer, Width, Height, B0, B1, B2, B3);

	ConvertBGRAF2BGR8U(Buffer + 3 * Width * 3, Dest, Width, Height, Stride);

	free(Buffer);
}

void GaussBlur_SSE(unsigned char* Src, unsigned char* Dest, int Width, int Height, int Stride, float Radius)
{
	const int PAD = 3 * Radius;
	float B0, B1, B2, B3;
	CalcGaussCof(Radius, B0, B1, B2, B3);

	int extWidth = Width + 2 * PAD;
	int extHeight = Height + 2 * PAD;
	float* Buffer = (float*)_mm_malloc(extWidth * extHeight * sizeof(float) * 4, 16);
	if (!Buffer) 
		return;

	//for (int y = 0; y < Height; y++) 
	//{
	//	unsigned char* srcRow = Src + y * Stride;
	//	float* dstRow = Buffer + (PAD + y) * extWidth * 4 + PAD * 4;
	//	for (int x = 0; x < Width; x++) 
	//	{
	//		dstRow[0] = srcRow[0];
	//		dstRow[1] = srcRow[1];
	//		dstRow[2] = srcRow[2];
	//		dstRow[3] = 0.0f;
	//		srcRow += 3;
	//		dstRow += 4;
	//	}
	//}
	//ConvertBGR8U2BGRAF_SSE(Src, Buffer + PAD * extWidth * 4, Width, Height, Stride, extWidth * 4, PAD * 4);
	ConvertBGR8U2BGRAF_async(Src, Buffer + PAD * extWidth * 4, Width, Height, Stride, extWidth * 4, PAD * 4);

	// 复制左右边界
	for (int y = 0; y < extHeight; y++) 
	{
		float* row = Buffer + y * extWidth * 4;
		// 左边界：复制有效区第一列
		float* firstPixel = row + PAD * 4;
		for (int i = 0; i < PAD; i++) 
		{
			memcpy(row + i * 4, firstPixel, 4 * sizeof(float));
		}
		// 有边界：复制有效区最后一列
		float* lastPixel = row + (PAD + Width - 1) * 4;
		for (int i = 0; i < PAD; i++) {
			memcpy(row + (PAD + Width + i) * 4, lastPixel, 4 * sizeof(float));
		}
	}
	// 复制上下边界（整行复制，包含已扩展的左右边界）
	float* topRow = Buffer + PAD * extWidth * 4;
	for (int i = 0; i < PAD; i++) 
	{
		memcpy(Buffer + i * extWidth * 4, topRow, extWidth * 4 * sizeof(float));
	}
	float* bottomRow = Buffer + (PAD + Height - 1) * extWidth * 4;
	for (int i = 0; i < PAD; i++) 
	{
		memcpy(Buffer + (PAD + Height + i) * extWidth * 4, bottomRow, extWidth * 4 * sizeof(float));
	}

	//GaussBlurFromLeftToRight_SSE(Buffer, extWidth, extHeight, B0, B1, B2, B3);
	//GaussBlur_asyncLeftToRight(Buffer, extWidth, extHeight, B0, B1, B2, B3);
	GaussBlur_asyncProcess(Buffer, extWidth, extHeight, B0, B1, B2, B3, extHeight, GaussBlurFromLeftToRight_test);
	//GaussBlurFromRightToLeft_SSE(Buffer, extWidth, extHeight, B0, B1, B2, B3);
	//GaussBlur_asyncRightToLeft(Buffer, extWidth, extHeight, B0, B1, B2, B3);
	GaussBlur_asyncProcess(Buffer, extWidth, extHeight, B0, B1, B2, B3, extHeight, GaussBlurFromRightToLeft_test);
	//GaussBlurFromTopToBottom_SSE(Buffer, extWidth, extHeight, B0, B1, B2, B3);
	//GaussBlur_asyncTopToBottom(Buffer, extWidth, extHeight, B0, B1, B2, B3);
	GaussBlur_asyncProcess(Buffer, extWidth, extHeight, B0, B1, B2, B3, extWidth, GaussBlurFromTopToBottom_test);
	//GaussBlurFromBottomToTop_SSE(Buffer, extWidth, extHeight, B0, B1, B2, B3);
	//GaussBlur_asyncBottomToTop(Buffer, extWidth, extHeight, B0, B1, B2, B3);
	GaussBlur_asyncProcess(Buffer, extWidth, extHeight, B0, B1, B2, B3, extWidth, GaussBlurFromBottomToTop_test);

	//ConvertBGRAF2BGR8U_SSE_Float_Correct(Buffer + PAD * extWidth * 4 + PAD * 4, Dest, Width, Height, Stride, extWidth, extHeight);
	ConvertBGRAF2BGR8U_async(Buffer + PAD * extWidth * 4 + PAD * 4, Dest, Width, Height, Stride, extWidth, extHeight);

	_mm_free(Buffer);
}

static void ConvertGray8U2Float_async(unsigned char* Src, float* Dest, int Width, int Height, int Stride, int DstRowStride, int DstColOffset)
{
	const int32_t hw_concur = std::min(100, static_cast<int32_t>(std::thread::hardware_concurrency()));
	std::vector<std::future<void>> fut(hw_concur);
	const int thread_stride = (Height - 1) / hw_concur + 1;
	int i = 0, start = 0;
	for (; i < std::min(Height, hw_concur); i++, start += thread_stride)
	{
		fut[i] = std::async(std::launch::async, [=]() {
			ConvertGray8U2Float_SSE_test(Src, Dest, Width, Height, Stride, DstRowStride, DstColOffset, start, thread_stride);
			});
	}
	for (int j = 0; j < i; ++j)
	{
		if (fut[j].valid())
			fut[j].wait();
	}
}

void GaussBlurGray_SSE(unsigned char* Src, unsigned char* Dest, int Width, int Height, int Stride, float Radius)
{
	const int PAD = (int)(3 * Radius);
	float B0, B1, B2, B3;
	CalcGaussCof(Radius, B0, B1, B2, B3);
	int extWidth = Width + 2 * PAD;
	int extHeight = Height + 2 * PAD;
	float* Buffer = (float*)_mm_malloc(extWidth * extHeight * sizeof(float), 16);
	if (!Buffer)
		return;
	ConvertGray8U2Float_async(Src, Buffer + PAD * extWidth, Width, Height, Stride, extWidth, PAD);
	// replicate left/right borders
	for (int y = 0; y < extHeight; y++)
	{
		float* row = Buffer + y * extWidth;
		float first = row[PAD];
		float last = row[PAD + Width - 1];
		for (int i = 0; i < PAD; i++)
		{
			row[i] = first;
			row[PAD + Width + i] = last;
		}
	}
	// replicate top/bottom borders
	float* topRow = Buffer + PAD * extWidth;
	for (int i = 0; i < PAD; i++)
		memcpy(Buffer + i * extWidth, topRow, extWidth * sizeof(float));
	float* bottomRow = Buffer + (PAD + Height - 1) * extWidth;
	for (int i = 0; i < PAD; i++)
		memcpy(Buffer + (PAD + Height + i) * extWidth, bottomRow, extWidth * sizeof(float));
	GaussBlur_asyncProcess(Buffer, extWidth, extHeight, B0, B1, B2, B3, extHeight, GaussBlurGrayFromLeftToRight_test);
	GaussBlur_asyncProcess(Buffer, extWidth, extHeight, B0, B1, B2, B3, extHeight, GaussBlurGrayFromRightToLeft_test);
	GaussBlur_asyncProcess(Buffer, extWidth, extHeight, B0, B1, B2, B3, extWidth, GaussBlurGrayFromTopToBottom_test);
	GaussBlur_asyncProcess(Buffer, extWidth, extHeight, B0, B1, B2, B3, extWidth, GaussBlurGrayFromBottomToTop_test);
	ConvertFloat2Gray8U_async(Buffer + PAD * extWidth + PAD, Dest, Width, Height, Stride, extWidth);
	_mm_free(Buffer);
}

void ConvertGray8U2Float_SSE_test(unsigned char* Src, float* Dest, int Width, int Height, int Stride, int DstRowStride, int DstColOffset, int start_row, int thread_stride)
{
	const int BlockSize = 4;
	int Block = Width / BlockSize;
	const __m128i Zero = _mm_setzero_si128();

	for (int Y = start_row; Y < (start_row + thread_stride) && Y < Height; Y++)
	{
		unsigned char* LinePS = Src + Y * Stride;
		float* LinePD = Dest + Y * DstRowStride + DstColOffset;
		int X = 0;
		for (; X < Block * BlockSize; X += BlockSize, LinePS += BlockSize, LinePD += BlockSize)
		{
			__m128i src8 = _mm_cvtsi32_si128(*(const int*)LinePS); // 4 uchar
			__m128i src16 = _mm_unpacklo_epi8(src8, Zero);
			__m128i src32 = _mm_unpacklo_epi16(src16, Zero);
			_mm_storeu_ps(LinePD, _mm_cvtepi32_ps(src32));
		}
		for (; X < Width; X++, LinePS++, LinePD++)
		{
			*LinePD = (float)(*LinePS);
		}
	}
}

void GaussBlurGrayFromLeftToRight_test(float* Data, int Width, int Height, float B0, float B1, float B2, float B3, int start_row, int thread_stride)
{
	// Horizontal IIR is causal along X; process rows independently (scalar per pixel).
	for (int Y = start_row; Y < (start_row + thread_stride) && Y < Height; Y++)
	{
		float* LinePD = Data + Y * Width;
		float V1 = LinePD[0], V2 = V1, V3 = V1;
		for (int X = 0; X < Width; X++)
		{
			float V0 = LinePD[X];
			float V = V0 * B0 + V1 * B1 + V2 * B2 + V3 * B3;
			V3 = V2;
			V2 = V1;
			V1 = V;
			LinePD[X] = V;
		}
	}
}

void GaussBlurGrayFromRightToLeft_test(float* Data, int Width, int Height, float B0, float B1, float B2, float B3, int start_row, int thread_stride)
{
	for (int Y = start_row; Y < (start_row + thread_stride) && Y < Height; Y++)
	{
		float* LinePD = Data + Y * Width + (Width - 1);
		float V1 = *LinePD, V2 = V1, V3 = V1;
		for (int X = Width - 1; X >= 0; X--, LinePD--)
		{
			float V0 = *LinePD;
			float V = V0 * B0 + V1 * B1 + V2 * B2 + V3 * B3;
			V3 = V2;
			V2 = V1;
			V1 = V;
			*LinePD = V;
		}
	}
}

void GaussBlurGrayFromTopToBottom_test(float* Data, int Width, int Height, float B0, float B1, float B2, float B3, int start_col, int thread_stride)
{
	const __m128 CofB0 = _mm_set1_ps(B0);
	const __m128 CofB1 = _mm_set1_ps(B1);
	const __m128 CofB2 = _mm_set1_ps(B2);
	const __m128 CofB3 = _mm_set1_ps(B3);
	const int end_col = std::min(start_col + thread_stride, Width);

	for (int Y = 0; Y < Height - 4; Y++)
	{
		int base = (Y + 3) * Width;
		float* LinePS0 = Data + base;
		float* LinePS1 = Data + base - Width;
		float* LinePS2 = Data + base - 2 * Width;
		float* LinePS3 = Data + base - 3 * Width;

		int X = start_col;
		for (; X + 4 <= end_col; X += 4)
		{
			__m128 V0 = _mm_loadu_ps(LinePS0 + X);
			__m128 V1 = _mm_loadu_ps(LinePS1 + X);
			__m128 V2 = _mm_loadu_ps(LinePS2 + X);
			__m128 V3 = _mm_loadu_ps(LinePS3 + X);
			__m128 V = _mm_add_ps(
				_mm_add_ps(_mm_mul_ps(CofB0, V0), _mm_mul_ps(CofB1, V1)),
				_mm_add_ps(_mm_mul_ps(CofB2, V2), _mm_mul_ps(CofB3, V3)));
			_mm_storeu_ps(LinePS0 + X, V);
		}
		for (; X < end_col; X++)
		{
			LinePS0[X] = LinePS0[X] * B0 + LinePS1[X] * B1 + LinePS2[X] * B2 + LinePS3[X] * B3;
		}
	}
}

void GaussBlurGrayFromBottomToTop_test(float* Data, int Width, int Height, float B0, float B1, float B2, float B3, int start_col, int thread_stride)
{
	const __m128 CofB0 = _mm_set1_ps(B0);
	const __m128 CofB1 = _mm_set1_ps(B1);
	const __m128 CofB2 = _mm_set1_ps(B2);
	const __m128 CofB3 = _mm_set1_ps(B3);
	const int end_col = std::min(start_col + thread_stride, Width);

	for (int Y = Height - 1 - 3; Y >= 0; Y--)
	{
		int base = Y * Width;
		float* LinePS0 = Data + base;
		float* LinePS1 = Data + base + Width;
		float* LinePS2 = Data + base + 2 * Width;
		float* LinePS3 = Data + base + 3 * Width;

		int X = start_col;
		for (; X + 4 <= end_col; X += 4)
		{
			__m128 V0 = _mm_loadu_ps(LinePS0 + X);
			__m128 V1 = _mm_loadu_ps(LinePS1 + X);
			__m128 V2 = _mm_loadu_ps(LinePS2 + X);
			__m128 V3 = _mm_loadu_ps(LinePS3 + X);
			__m128 V = _mm_add_ps(
				_mm_add_ps(_mm_mul_ps(CofB0, V0), _mm_mul_ps(CofB1, V1)),
				_mm_add_ps(_mm_mul_ps(CofB2, V2), _mm_mul_ps(CofB3, V3)));
			_mm_storeu_ps(LinePS0 + X, V);
		}
		for (; X < end_col; X++)
		{
			LinePS0[X] = LinePS0[X] * B0 + LinePS1[X] * B1 + LinePS2[X] * B2 + LinePS3[X] * B3;
		}
	}
}