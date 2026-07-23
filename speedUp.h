#pragma once

void CalcGaussCof(float Radius, float& B0, float& B1, float& B2, float& B3);

void ConvertBGR8U2BGRAF(unsigned char* Src, float* Dest, int Width, int Height, int Stride);

void ConvertBGR8U2BGRAF_SSE(unsigned char* Src, float* Dest, int Width, int Height, int Stride);

void GaussBlurFromLeftToRight(float* Data, int Width, int Height, float B0, float B1, float B2, float B3);

void GaussBlurFromLeftToRight_SSE(float* Data, int Width, int Height, float B0, float B1, float B2, float B3);

void GaussBlurFromRightToLeft(float* Data, int Width, int Height, float B0, float B1, float B2, float B3);

void GaussBlurFromRightToLeft_SSE(float* Data, int Width, int Height, float B0, float B1, float B2, float B3);


//w[n] w[n-1], w[n-2], w[n-3]
void GaussBlurFromTopToBottom(float* Data, int Width, int Height, float B0, float B1, float B2, float B3);

void GaussBlurFromTopToBottom_SSE(float* Data, int Width, int Height, float B0, float B1, float B2, float B3);
//w[n] w[n+1], w[n+2], w[n+3]
void GaussBlurFromBottomToTop(float* Data, int Width, int Height, float B0, float B1, float B2, float B3);

void GaussBlurFromBottomToTop_SSE(float* Data, int Width, int Height, float B0, float B1, float B2, float B3);
void ConvertBGRAF2BGR8U(float* Src, unsigned char* Dest, int Width, int Height, int Stride);


void ConvertBGRAF2BGR8U_SSE(unsigned char* Src, unsigned char* Dest, int Width, int Height, int Stride);

void GaussBlur(unsigned char* Src, unsigned char* Dest, int Width, int Height, int Stride, float Radius);

void GaussBlur_SSE(unsigned char* Src, unsigned char* Dest, int Width, int Height, int Stride, float Radius);

void GaussBlurFromLeftToRight_test(float* Data, int Width, int Height, float B0, float B1, float B2, float B3, int start_row, int thread_stride);

void GaussBlurFromRightToLeft_test(float* Data, int Width, int Height, float B0, float B1, float B2, float B3, int start_row, int thread_stride);

void GaussBlurFromTopToBottom_test(float* Data, int Width, int Height, float B0, float B1, float B2, float B3, int start_col, int thread_stride);

void GaussBlurFromBottomToTop_test(float* Data, int Width, int Height, float B0, float B1, float B2, float B3, int start_col, int thread_stride);

void ConvertBGRAF2BGR8U_SSE_Float_Correct_test(float* Src, unsigned char* Dest, int Width, int Height, int Stride, int extWidth, int extHeight, int start_row, int thread_stride);