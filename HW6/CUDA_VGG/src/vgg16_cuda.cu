#include "vgg16_cuda.h"
#define TILE_WIDTH 16
using namespace std;

__global__ void normalizeKernel(uint8_t* image, float* input, int B, int IC, int size) {
  // Initialize variables
  
  float max_int = 255.0;
  float mean = 0.5;
  float var = 0.5;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  if(Row < size && Col < size){
    for(int b = 0; b < B; ++b){
      for(int c = 0; c < IC; ++c){
        input[b * c * Row + Col] = image[b * c * Row + Col] / max_int;       // transforms.ToTensor();
        input[b * c * Row + Col] = (input[b * c * Row + Col] - mean) / var;  // transforms.Normalize();
      }
    }
  }
}

void normalize(const uint8_t* const image, float* input, int B, int IC, int size){
  dim3 dimGrid(ceil(size/TILE_WIDTH), ceil(size/TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  normalizeKernel<<<dimGrid, dimBlock>>>((uint8_t *)image, (float *)input, B, IC, size);
}

__global__ void padKernel(float* input, float* input_padded, int B, int C, int H, int W, int P){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  int H_OUT = H+2*P;
  int W_OUT = W+2*P;
  if (Col < W && Row < H) {
    for (int b = 0; b < B; b++){
      for (int c = 0; c < C; c++){
        // Init values
        int input_base = b * (C * H * W) + c * (H * W) + Row * (W) + Col;

        // Set output with max value
        int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) +
                            (Row+P) * W_OUT + (Col + P);
        input_padded[output_index] = input[input_base];
      }
    }
  }  
}

void pad(const float* const input, float* const input_padded, int B, int C, int H, int W, int P){
  dim3 dimGrid(ceil(H/TILE_WIDTH), ceil(W/TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  padKernel<<<dimGrid, dimBlock>>>((float *)input, (float *)input_padded, B, C, H, W, P);
}

__global__ void convKernel(float* input, float* output, float* weight,
  float* bias, int B, int H, int W, int IC, int OC, int K){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  // Initialize variable
  int H_OUT = H - (K - 1);
  int W_OUT = W - (K - 1);
  // Convolution
  if (Col < W_OUT && Row < H_OUT) {
    for (int b = 0; b < B; b++){              // mini-batch
      for (int oc = 0; oc < OC; oc++) {      // Output Channel
        int output_index = b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + Row * W_OUT + Col;
        output[output_index] = bias[oc];
        for (int ic = 0; ic < IC; ic++) {
          int input_base = b * (IC * H * W) + ic * (H * W) + Row * (W) + Col;
          int kernel_base = oc * (IC * K * K) + ic * (K * K);
          for (int kh = 0; kh < K; kh++){
            for (int kw = 0; kw < K; kw++) {
              float val = input[input_base + kh * (W) + kw] * weight[kernel_base + kh * (K) + kw];
              output[output_index] += val;
            }
          }
        }
      }
    }
  }
}

void conv(const float* const input, float* const output, float* const weight, float* const bias, int B, int H, int W, int IC, int OC, int K){
  dim3 dimGrid(ceil(H/TILE_WIDTH), ceil(W/TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  convKernel<<<dimGrid, dimBlock>>>((float *)input, (float *)output, weight, bias, B, H, W, IC, OC, K);
}

__global__ void ReLUKernel(float* feature_map, int size){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  if(Row * Col < size){
    if(feature_map[Row+Col] < 0.0){
      feature_map[Row+Col] = (float)0.0;
    }
    else{
      feature_map[Row+Col] = feature_map[Row+Col];
    }
  }
}

void relu(const float* const feature_map, int size){
  dim3 dimGrid(ceil(size/TILE_WIDTH), ceil(size/TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  ReLUKernel<<<dimGrid, dimBlock>>>((float *)feature_map, size);  
}

__global__ void poolKernel(float* input, float* output, int B, int C, int H, int W){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  int scale = 2;
  int H_OUT = H / scale;
  int W_OUT = W / scale;
  // Max Pooling
  for (int b = 0; b < B; b++)
    for (int c = 0; c < C; c++)
      for (int h = 0; h < H; h += 2)
        for (int w = 0; w < W; w += 2) {
          // Init values
          int input_base = b * (C * H * W) + c * (H * W) + h * (W) + w;
          int max_sh = 0;
          int max_sw = 0;
          float max_val = -1;
          // Find maximum
          for (int sh = 0; sh < scale; sh++)
            for (int sw = 0; sw < scale; sw++) {
              float val = input[input_base + sh * (W) + sw];
              if (val - max_val > 0.000001) {
                max_val = val;
                max_sh = sh;
                max_sw = sw;
              }
            }
          // Set output with max value
          int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) +
                             (h / 2) * W_OUT + (w / 2);
          output[output_index] = max_val;
        }
}

void pool(const float* const input, const float* output, int B, int C, int H, int W){
  dim3 dimGrid(ceil(H/TILE_WIDTH), ceil(W/TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  poolKernel<<<dimGrid, dimBlock>>>((float*)input, (float*)output, B, C, H, W);
}

__global__ void fcKernel(float* input, float* output, float* weight, float* bias, int B, int IC, int OC){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  // Fully Connected
  if(Row < OC && Col < IC){
    for (int b = 0; b < B; b++){
      output[b * OC + Row] = bias[Row];
      output[b * OC + Row] += weight[Row * IC + Col] * input[b * IC + Col];
    }
  }
}

void fc(const float* const input, const float* output, float* weight, float* bias, int B, int IC, int OC){
  dim3 dimGrid(ceil(OC/TILE_WIDTH), ceil(IC/TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  fcKernel<<<dimGrid, dimBlock>>>((float*)input, (float*)output, weight, bias, B, IC, OC);
}

void vgg16_cuda::predict(int batch) {
  
  normalize(d_image, d_input, batch, input_channel, input_size);
  
  //////////BLOCK 1/////////////////////////////////
  pad(d_input, d_input_padded, batch, input_channel, input_size, input_size, conv1_1_padding_size);
  conv(d_input_padded, d_C1_1_feature_map, d_conv1_1_weight, d_conv1_1_bias, batch, (input_size+2*conv1_1_padding_size), (input_size+2*conv1_1_padding_size), conv1_1_in_channel, conv1_1_out_channel, conv1_1_kernel_size);
  relu(d_C1_1_feature_map, batch * C1_1_channel * C1_1_size * C1_1_size);

  pad(d_C1_1_feature_map, d_C1_1_feature_map_padded, batch, C1_1_channel, C1_1_size, C1_1_size, conv1_2_padding_size);
  conv(d_C1_1_feature_map_padded, d_C1_2_feature_map, d_conv1_2_weight, d_conv1_2_bias, batch, (C1_1_size+2*conv1_2_padding_size), (C1_1_size+2*conv1_2_padding_size), conv1_2_in_channel, conv1_2_out_channel, conv1_2_kernel_size);
  relu(d_C1_2_feature_map, batch * C1_2_channel * C1_2_size * C1_2_size);
  pool(d_C1_2_feature_map, d_S1_feature_map, batch, C1_2_channel, C1_2_size, C1_2_size);
  printf("Block1\n\n");
  
  //////////BLOCK 2/////////////////////////////////
  pad(d_S1_feature_map, d_S1_feature_map_padded, batch, S1_channel, S1_size, S1_size, conv2_1_padding_size); 
  conv(d_S1_feature_map_padded, d_C2_1_feature_map, d_conv2_1_weight, d_conv2_1_bias, batch, (S1_size+2*conv2_1_padding_size), (S1_size+2*conv2_1_padding_size), S1_channel, conv2_1_out_channel, conv2_1_kernel_size);
  relu(d_C2_1_feature_map, batch * C2_1_channel * C2_1_size * C2_1_size);
  
  pad(d_C2_1_feature_map, d_C2_1_feature_map_padded, batch, C2_1_channel, C2_1_size, C2_1_size, conv2_2_padding_size); 
  conv(d_C2_1_feature_map_padded, d_C2_2_feature_map, d_conv2_2_weight, d_conv2_2_bias, batch, (C2_1_size+2*conv2_2_padding_size), (C2_1_size+2*conv2_2_padding_size), conv2_2_in_channel, conv2_2_out_channel, conv2_2_kernel_size);
  relu(d_C2_2_feature_map, batch * C2_2_channel * C2_2_size * C2_2_size);
  pool(d_C2_2_feature_map, d_S2_feature_map, batch, C2_2_channel, C2_2_size, C2_2_size);
  printf("Block2\n\n");

  //////////BLOCK 3/////////////////////////////////
  pad(d_S2_feature_map, d_S2_feature_map_padded, batch, S2_channel, S2_size, S2_size, conv3_1_padding_size); 
  conv(d_S2_feature_map_padded, d_C3_1_feature_map, d_conv3_1_weight, d_conv3_1_bias, batch, (S2_size+2*conv3_1_padding_size), (S2_size+2*conv3_1_padding_size), S2_channel, conv3_1_out_channel, conv3_1_kernel_size);
  relu(d_C3_1_feature_map, batch * C3_1_channel * C3_1_size * C3_1_size);

  pad(d_C3_1_feature_map, d_C3_1_feature_map_padded, batch, C3_1_channel, C3_1_size, C3_1_size, conv3_2_padding_size); 
  conv(d_C3_1_feature_map_padded, d_C3_2_feature_map, d_conv3_2_weight, d_conv3_2_bias, batch, (C3_1_size+2*conv3_2_padding_size), (C3_1_size+2*conv3_2_padding_size), conv3_2_in_channel, conv3_2_out_channel, conv3_2_kernel_size);
  relu(d_C3_2_feature_map, batch * C3_2_channel * C3_2_size * C3_2_size);

  pad(d_C3_2_feature_map, d_C3_2_feature_map_padded, batch, C3_2_channel, C3_2_size, C3_2_size, conv3_3_padding_size); 
  conv(d_C3_2_feature_map_padded, d_C3_3_feature_map, d_conv3_3_weight, d_conv3_3_bias, batch, (C3_2_size+2*conv3_3_padding_size), (C3_2_size+2*conv3_3_padding_size), conv3_3_in_channel, conv3_3_out_channel, conv3_3_kernel_size);
  relu(d_C3_3_feature_map, batch * C3_3_channel * C3_3_size * C3_3_size);
  pool(d_C3_3_feature_map, d_S3_feature_map, batch, C3_3_channel, C3_3_size, C3_3_size);
  printf("Block3\n\n");

  //////////BLOCK 4/////////////////////////////////
  pad(d_S3_feature_map, d_S3_feature_map_padded, batch, S3_channel, S3_size, S3_size, conv4_1_padding_size);
  // conv2d
  conv(d_S3_feature_map_padded, d_C4_1_feature_map, d_conv4_1_weight, d_conv4_1_bias, batch, S3_size+2*conv4_1_padding_size,
      S3_size+2*conv4_1_padding_size, conv4_1_in_channel, conv4_1_out_channel, conv4_1_kernel_size);
  relu(d_C4_1_feature_map, batch * C4_1_channel * C4_1_size * C4_1_size);
  // ZeroPad2d
  pad(d_C4_1_feature_map, d_C4_1_feature_map_padded, batch, C4_1_channel, C4_1_size, C4_1_size, conv4_2_padding_size);
  // conv2d
  conv(d_C4_1_feature_map_padded, d_C4_2_feature_map, d_conv4_2_weight, d_conv4_2_bias, batch, C4_1_size+2*conv4_2_padding_size,
    C4_1_size+2*conv4_2_padding_size, conv4_2_in_channel, conv4_2_out_channel, conv4_2_kernel_size);
  relu(d_C4_2_feature_map, batch * C4_2_channel * C4_2_size * C4_2_size);
  // ZeroPad2d
  pad(d_C4_2_feature_map, d_C4_2_feature_map_padded, batch, C4_2_channel, C4_2_size, C4_2_size, conv4_3_padding_size);
  // conv2d
  conv(d_C4_2_feature_map_padded, d_C4_3_feature_map, d_conv4_3_weight, d_conv4_3_bias, batch, C4_2_size+2*conv4_3_padding_size,
    C4_2_size+2*conv4_3_padding_size, conv4_3_in_channel, conv4_3_out_channel, conv4_3_kernel_size);
  relu(d_C4_3_feature_map, batch * C4_3_channel * C4_3_size * C4_3_size);
  // MaxPool2d
  pool(d_C4_3_feature_map, d_S4_feature_map, batch, C4_3_channel, C4_3_size, C4_3_size);
  printf("Block4\n\n");
  
  //////////BLOCK 5/////////////////////////////////
  pad(d_S4_feature_map, d_S4_feature_map_padded, batch, S4_channel, S4_size, S4_size, conv5_1_padding_size);
  // conv2d
  conv(d_S4_feature_map_padded, d_C5_1_feature_map, conv5_1_weight, d_conv5_1_bias, batch, S4_size+2*conv5_1_padding_size,
        S4_size+2*conv5_1_padding_size, conv5_1_in_channel, conv5_1_out_channel, conv5_1_kernel_size);
  relu(d_C5_1_feature_map, batch * C5_1_channel * C5_1_size * C5_1_size);
  // // ZeroPad2d
  pad(d_C5_1_feature_map, d_C5_1_feature_map_padded, batch, C5_1_channel, C5_1_size, C5_1_size, conv5_2_padding_size);
  // // conv2d
  conv(d_C5_1_feature_map_padded, d_C5_2_feature_map, d_conv5_2_weight, d_conv5_2_bias, batch, C5_1_size+2*conv5_2_padding_size,
       C5_1_size+2*conv5_2_padding_size, conv5_2_in_channel, conv5_2_out_channel, conv5_2_kernel_size);
  relu(d_C5_2_feature_map, batch * C5_2_channel * C5_2_size * C5_2_size);
  // // ZeroPad2d
  pad(d_C5_2_feature_map, d_C5_2_feature_map_padded, batch, C5_2_channel, C5_2_size, C5_2_size, conv5_3_padding_size);
  // // conv2d
  conv(d_C5_2_feature_map_padded, d_C5_3_feature_map, d_conv5_3_weight, d_conv5_3_bias, batch, C5_2_size+2*conv5_3_padding_size,
       C5_2_size+2*conv5_3_padding_size, conv5_3_in_channel, conv5_3_out_channel, conv5_3_kernel_size);
  relu(d_C5_3_feature_map, batch * C5_3_channel * C5_3_size * C5_3_size);
  // // MaxPool2d
  pool(d_C5_3_feature_map, d_S5_feature_map, batch, C5_3_channel, C5_3_size, C5_3_size);
  fc(d_S5_feature_map, d_output, d_fc1_weight, d_fc1_bias, batch, fc1_in_channel, fc1_out_channel);

  printf("Block5\n\n");
    // TODO: Implement fc1
    // TODO: Implement relu

    /* NOTE: unless you want to make a major change to this class structure, 
    *  you need to write your output to the device memory d_output 
    *  so that classify() can handle the rest.
    */
}

void vgg16_cuda::prepare_device_memory(uint8_t* image) {
  // Alloc Model Parameters

  //////////BLOCK 1/////////////////////////////////
  cudaMalloc((void**)&d_conv1_1_weight,
             sizeof(float) * conv1_1_in_channel * conv1_1_out_channel *
                 conv1_1_kernel_size * conv1_1_kernel_size);
  cudaMalloc((void**)&d_conv1_1_bias, sizeof(float) * conv1_1_out_channel);
  cudaMalloc((void**)&d_conv1_2_weight,
             sizeof(float) * conv1_2_in_channel * conv1_2_out_channel *
                 conv1_2_kernel_size * conv1_2_kernel_size);
  cudaMalloc((void**)&d_conv1_2_bias, sizeof(float) * conv1_2_out_channel);

  //////////BLOCK 2/////////////////////////////////
  cudaMalloc((void**)&d_conv2_1_weight,
             sizeof(float) * conv2_1_in_channel * conv2_1_out_channel *
                 conv2_1_kernel_size * conv2_1_kernel_size);
  cudaMalloc((void**)&d_conv2_1_bias, sizeof(float) * conv2_1_out_channel);
  cudaMalloc((void**)&d_conv2_2_weight,
             sizeof(float) * conv2_2_in_channel * conv2_2_out_channel *
                 conv2_2_kernel_size * conv2_2_kernel_size);
  cudaMalloc((void**)&d_conv2_2_bias, sizeof(float) * conv2_2_out_channel);

  //////////BLOCK 3/////////////////////////////////
  cudaMalloc((void**)&d_conv3_1_weight,
             sizeof(float) * conv3_1_in_channel * conv3_1_out_channel *
                 conv3_1_kernel_size * conv3_1_kernel_size);
  cudaMalloc((void**)&d_conv3_1_bias, sizeof(float) * conv3_1_out_channel);
  cudaMalloc((void**)&d_conv3_2_weight,
             sizeof(float) * conv3_2_in_channel * conv3_2_out_channel *
                 conv3_2_kernel_size * conv3_2_kernel_size);
  cudaMalloc((void**)&d_conv3_2_bias, sizeof(float) * conv3_2_out_channel);
  cudaMalloc((void**)&d_conv3_3_weight,
             sizeof(float) * conv3_3_in_channel * conv3_3_out_channel *
                 conv3_3_kernel_size * conv3_3_kernel_size);
  cudaMalloc((void**)&d_conv3_3_bias, sizeof(float) * conv3_3_out_channel);

  //////////BLOCK 4/////////////////////////////////
  cudaMalloc((void**)&d_conv4_1_weight,
             sizeof(float) * conv4_1_in_channel * conv4_1_out_channel *
                 conv4_1_kernel_size * conv4_1_kernel_size);
  cudaMalloc((void**)&d_conv4_1_bias, sizeof(float) * conv4_1_out_channel);
  cudaMalloc((void**)&d_conv4_2_weight,
             sizeof(float) * conv4_2_in_channel * conv4_2_out_channel *
                 conv4_2_kernel_size * conv4_2_kernel_size);
  cudaMalloc((void**)&d_conv4_2_bias, sizeof(float) * conv4_2_out_channel);
  cudaMalloc((void**)&d_conv4_3_weight,
             sizeof(float) * conv4_3_in_channel * conv4_3_out_channel *
                 conv4_3_kernel_size * conv4_3_kernel_size);
  cudaMalloc((void**)&d_conv4_3_bias, sizeof(float) * conv4_3_out_channel);

  //////////BLOCK 5/////////////////////////////////
  cudaMalloc((void**)&d_conv5_1_weight,
             sizeof(float) * conv5_1_in_channel * conv5_1_out_channel *
                 conv5_1_kernel_size * conv5_1_kernel_size);
  cudaMalloc((void**)&d_conv5_1_bias, sizeof(float) * conv5_1_out_channel);
  cudaMalloc((void**)&d_conv5_2_weight,
             sizeof(float) * conv5_2_in_channel * conv5_2_out_channel *
                 conv5_2_kernel_size * conv5_2_kernel_size);
  cudaMalloc((void**)&d_conv5_2_bias, sizeof(float) * conv5_2_out_channel);
  cudaMalloc((void**)&d_conv5_3_weight,
             sizeof(float) * conv5_3_in_channel * conv5_3_out_channel *
                 conv5_3_kernel_size * conv5_3_kernel_size);
  cudaMalloc((void**)&d_conv5_3_bias, sizeof(float) * conv5_3_out_channel);

  //////////FC 1////////////////////////////////////
  cudaMalloc((void**)&d_fc1_weight,
             sizeof(float) * fc1_in_channel * fc1_out_channel);
  cudaMalloc((void**)&d_fc1_bias, sizeof(float) * fc1_out_channel);

  // Alloc Activations
  cudaMalloc((void**)&d_image,
             sizeof(uint8_t) * batch * input_size * input_size * input_channel);
  cudaMalloc((void**)&d_input,
             sizeof(float) * batch * input_channel * input_size * input_size);

  //////////BLOCK 1/////////////////////////////////
  cudaMalloc((void**)&d_input_padded,
             sizeof(float) * batch * input_channel * (input_size+2*conv1_1_padding_size) * (input_size+2*conv1_1_padding_size));
  cudaMalloc((void**)&d_C1_1_feature_map,
             sizeof(float) * batch * C1_1_channel * C1_1_size * C1_1_size);
  cudaMalloc((void**)&d_C1_1_feature_map_padded,
             sizeof(float) * batch * C1_1_channel * (C1_1_size+2*conv1_2_padding_size) * (C1_1_size+2*conv1_2_padding_size));
  cudaMalloc((void**)&d_C1_2_feature_map,
             sizeof(float) * batch * C1_2_channel * C1_2_size * C1_2_size);
  cudaMalloc((void**)&d_S1_feature_map,
             sizeof(float) * batch * S1_channel * S1_size * S1_size);

  //////////BLOCK 2/////////////////////////////////
  cudaMalloc((void**)&d_S1_feature_map_padded,
             sizeof(float) * batch * S1_channel * (S1_size+2*conv2_1_padding_size) * (S1_size+2*conv2_1_padding_size));
  cudaMalloc((void**)&d_C2_1_feature_map,
             sizeof(float) * batch * C2_1_channel * C2_1_size * C2_1_size);
  cudaMalloc((void**)&d_C2_1_feature_map_padded,
             sizeof(float) * batch * C2_1_channel * (C2_1_size+2*conv2_2_padding_size) * (C2_1_size+2*conv2_2_padding_size));
  cudaMalloc((void**)&d_C2_2_feature_map,
             sizeof(float) * batch * C2_2_channel * C2_2_size * C2_2_size);
  cudaMalloc((void**)&d_S2_feature_map,
             sizeof(float) * batch * S2_channel * S2_size * S2_size);

  //////////BLOCK 3/////////////////////////////////
  cudaMalloc((void**)&d_S2_feature_map_padded,
             sizeof(float) * batch * S2_channel * (S2_size+2*conv3_1_padding_size) * (S2_size+2*conv3_1_padding_size));
  cudaMalloc((void**)&d_C3_1_feature_map,
             sizeof(float) * batch * C3_1_channel * C3_1_size * C3_1_size);
  cudaMalloc((void**)&d_C3_1_feature_map_padded,
             sizeof(float) * batch * C3_1_channel * (C3_1_size+2*conv3_2_padding_size) * (C3_1_size+2*conv3_2_padding_size));
  cudaMalloc((void**)&d_C3_2_feature_map,
             sizeof(float) * batch * C3_2_channel * C3_2_size * C3_2_size);
  cudaMalloc((void**)&d_C3_2_feature_map_padded,
             sizeof(float) * batch * C3_2_channel * (C3_2_size+2*conv3_3_padding_size) * (C3_2_size+2*conv3_3_padding_size));
  cudaMalloc((void**)&d_C3_3_feature_map,
             sizeof(float) * batch * C3_3_channel * C3_3_size * C3_3_size);
  cudaMalloc((void**)&d_S3_feature_map,
             sizeof(float) * batch * S3_channel * S3_size * S3_size);

  //////////BLOCK 4/////////////////////////////////
  cudaMalloc((void**)&d_S3_feature_map_padded,
             sizeof(float) * batch * S3_channel * (S3_size+2*conv4_1_padding_size) * (S3_size+2*conv4_1_padding_size));
  cudaMalloc((void**)&d_C4_1_feature_map,
             sizeof(float) * batch * C4_1_channel * C4_1_size * C4_1_size);
  cudaMalloc((void**)&d_C4_1_feature_map_padded,
             sizeof(float) * batch * C4_1_channel * (C4_1_size+2*conv4_2_padding_size) * (C4_1_size+2*conv4_2_padding_size));
  cudaMalloc((void**)&d_C4_2_feature_map,
             sizeof(float) * batch * C4_2_channel * C4_2_size * C4_2_size);
  cudaMalloc((void**)&d_C4_2_feature_map_padded,
             sizeof(float) * batch * C4_2_channel * (C4_2_size+2*conv4_3_padding_size) * (C4_2_size+2*conv4_3_padding_size));
  cudaMalloc((void**)&d_C4_3_feature_map,
             sizeof(float) * batch * C4_3_channel * C4_3_size * C4_3_size);
  cudaMalloc((void**)&d_S4_feature_map,
             sizeof(float) * batch * S4_channel * S4_size * S4_size);

  //////////BLOCK 5/////////////////////////////////
  cudaMalloc((void**)&d_S4_feature_map_padded,
             sizeof(float) * batch * S4_channel * (S4_size+2*conv5_1_padding_size) * (S4_size+2*conv5_1_padding_size));
  cudaMalloc((void**)&d_C5_1_feature_map,
             sizeof(float) * batch * C5_1_channel * C5_1_size * C5_1_size);
  cudaMalloc((void**)&d_C5_1_feature_map_padded,
             sizeof(float) * batch * C5_1_channel * (C5_1_size+2*conv5_2_padding_size) * (C5_1_size+2*conv5_2_padding_size));
  cudaMalloc((void**)&d_C5_2_feature_map,
             sizeof(float) * batch * C5_2_channel * C5_2_size * C5_2_size);
  cudaMalloc((void**)&d_C5_2_feature_map_padded,
             sizeof(float) * batch * C5_2_channel * (C5_2_size+2*conv5_3_padding_size) * (C5_2_size+2*conv5_3_padding_size));
  cudaMalloc((void**)&d_C5_3_feature_map,
             sizeof(float) * batch * C5_3_channel * C5_3_size * C5_3_size);
  cudaMalloc((void**)&d_S5_feature_map,
             sizeof(float) * batch * S5_channel * S5_size * S5_size);


  cudaMalloc((void**)&d_output, sizeof(float) * batch * output_size);

  // Copy Parameters
  //////////BLOCK 1/////////////////////////////////
  cudaMemcpy(d_conv1_1_weight, conv1_1_weight,
             sizeof(float) * conv1_1_in_channel * conv1_1_out_channel *
                 conv1_1_kernel_size * conv1_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv1_1_bias, conv1_1_bias, sizeof(float) * conv1_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv1_2_weight, conv1_2_weight,
              sizeof(float) * conv1_2_in_channel * conv1_2_out_channel *
                  conv1_2_kernel_size * conv1_2_kernel_size,
              cudaMemcpyHostToDevice);
   cudaMemcpy(d_conv1_2_bias, conv1_2_bias, sizeof(float) * conv1_2_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 2/////////////////////////////////
  cudaMemcpy(d_conv2_1_weight, conv2_1_weight,
             sizeof(float) * conv2_1_in_channel * conv2_1_out_channel *
                 conv2_1_kernel_size * conv2_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_1_bias, conv2_1_bias, sizeof(float) * conv2_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_2_weight, conv2_2_weight,
              sizeof(float) * conv2_2_in_channel * conv2_2_out_channel *
                  conv2_2_kernel_size * conv2_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_2_bias, conv2_2_bias, sizeof(float) * conv2_2_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 3/////////////////////////////////
  cudaMemcpy(d_conv3_1_weight, conv3_1_weight,
             sizeof(float) * conv3_1_in_channel * conv3_1_out_channel *
                 conv3_1_kernel_size * conv3_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_1_bias, conv3_1_bias, sizeof(float) * conv3_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_2_weight, conv3_2_weight,
              sizeof(float) * conv3_2_in_channel * conv3_2_out_channel *
                  conv3_2_kernel_size * conv3_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_2_bias, conv3_2_bias, sizeof(float) * conv3_2_out_channel,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_3_weight, conv3_3_weight,
              sizeof(float) * conv3_3_in_channel * conv3_3_out_channel *
                  conv3_3_kernel_size * conv3_3_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_3_bias, conv3_3_bias, sizeof(float) * conv3_3_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 4/////////////////////////////////
  cudaMemcpy(d_conv4_1_weight, conv4_1_weight,
             sizeof(float) * conv4_1_in_channel * conv4_1_out_channel *
                 conv4_1_kernel_size * conv4_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_1_bias, conv4_1_bias, sizeof(float) * conv4_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_2_weight, conv4_2_weight,
              sizeof(float) * conv4_2_in_channel * conv4_2_out_channel *
                  conv4_2_kernel_size * conv4_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_2_bias, conv4_2_bias, sizeof(float) * conv4_2_out_channel,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_3_weight, conv4_3_weight,
              sizeof(float) * conv4_3_in_channel * conv4_3_out_channel *
                  conv4_3_kernel_size * conv4_3_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_3_bias, conv4_3_bias, sizeof(float) * conv4_3_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 5/////////////////////////////////
  cudaMemcpy(d_conv5_1_weight, conv5_1_weight,
             sizeof(float) * conv5_1_in_channel * conv5_1_out_channel *
                 conv5_1_kernel_size * conv5_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_1_bias, conv5_1_bias, sizeof(float) * conv5_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_2_weight, conv5_2_weight,
              sizeof(float) * conv5_2_in_channel * conv5_2_out_channel *
                  conv5_2_kernel_size * conv5_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_2_bias, conv5_2_bias, sizeof(float) * conv5_2_out_channel,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_3_weight, conv5_3_weight,
              sizeof(float) * conv5_3_in_channel * conv5_3_out_channel *
                  conv5_3_kernel_size * conv5_3_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_3_bias, conv5_3_bias, sizeof(float) * conv5_3_out_channel,
              cudaMemcpyHostToDevice);


  cudaMemcpy(d_fc1_weight, fc1_weight,
             sizeof(float) * fc1_in_channel * fc1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc1_bias, fc1_bias, sizeof(float) * fc1_out_channel,
             cudaMemcpyHostToDevice);

  // copy input image
  size_t image_size = batch * input_size * input_size * input_channel;
  cudaMemcpy(d_image, image, image_size * sizeof(uint8_t),
             cudaMemcpyHostToDevice);
}

void vgg16_cuda::classify(int* predict, int batch) {
  // read logits back to cpu
  cudaMemcpy(output, d_output, sizeof(float) * output_size * batch,
             cudaMemcpyDeviceToHost);
  // Softmax
  softmax(output, predict, batch, output_size);
}

vgg16_cuda::~vgg16_cuda() {
  cudaFree(d_conv1_1_weight);   
  cudaFree(d_conv1_2_weight);   
  cudaFree(d_conv2_1_weight);   
  cudaFree(d_conv2_2_weight);  
  cudaFree(d_conv3_1_weight);   
  cudaFree(d_conv3_2_weight);   
  cudaFree(d_conv3_3_weight);   
  cudaFree(d_conv4_1_weight);   
  cudaFree(d_conv4_2_weight);   
  cudaFree(d_conv4_3_weight); 
  cudaFree(d_conv5_1_weight);   
  cudaFree(d_conv5_2_weight);   
  cudaFree(d_conv5_3_weight);   
 
  cudaFree(d_conv1_1_bias);   
  cudaFree(d_conv1_2_bias);   
  cudaFree(d_conv2_1_bias);   
  cudaFree(d_conv2_2_bias);  
  cudaFree(d_conv3_1_bias);   
  cudaFree(d_conv3_2_bias);   
  cudaFree(d_conv3_3_bias);   
  cudaFree(d_conv4_1_bias);   
  cudaFree(d_conv4_2_bias);   
  cudaFree(d_conv4_3_bias); 
  cudaFree(d_conv5_1_bias);   
  cudaFree(d_conv5_2_bias);   
  cudaFree(d_conv5_3_bias);   
   
  cudaFree(d_fc1_weight);     
  cudaFree(d_fc1_bias);        

  cudaFree(d_image);          
  cudaFree(d_input); 

  cudaFree(d_input_padded);          
  cudaFree(d_C1_1_feature_map); 
  cudaFree(d_C1_1_feature_map_padded); 
  cudaFree(d_C1_2_feature_map); 
  cudaFree(d_S1_feature_map); 

  cudaFree(d_S1_feature_map_padded); 
  cudaFree(d_C2_1_feature_map); 
  cudaFree(d_C2_1_feature_map_padded); 
  cudaFree(d_C2_2_feature_map); 
  cudaFree(d_S2_feature_map); 

  cudaFree(d_S2_feature_map_padded); 
  cudaFree(d_C3_1_feature_map); 
  cudaFree(d_C3_1_feature_map_padded); 
  cudaFree(d_C3_2_feature_map); 
  cudaFree(d_C3_2_feature_map_padded); 
  cudaFree(d_C3_3_feature_map); 
  cudaFree(d_S3_feature_map); 

  cudaFree(d_S3_feature_map_padded); 
  cudaFree(d_C4_1_feature_map); 
  cudaFree(d_C4_1_feature_map_padded); 
  cudaFree(d_C4_2_feature_map); 
  cudaFree(d_C4_2_feature_map_padded); 
  cudaFree(d_C4_3_feature_map); 
  cudaFree(d_S4_feature_map); 

  cudaFree(d_S4_feature_map_padded); 
  cudaFree(d_C5_1_feature_map); 
  cudaFree(d_C5_1_feature_map_padded); 
  cudaFree(d_C5_2_feature_map); 
  cudaFree(d_C5_2_feature_map_padded); 
  cudaFree(d_C5_3_feature_map); 
  cudaFree(d_S5_feature_map); 
 
  cudaFree(d_output);       
  cudaFree(d_predict_cuda);   
}
