#include "infer.hpp"
#include "trtengine.hpp"


namespace trtengine {
using namespace std;

#define checkRuntime(call)                                                                 \
  do {                                                                                     \
    auto ___call__ret_code__ = (call);                                                     \
    if (___call__ret_code__ != cudaSuccess) {                                              \
      INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call,                         \
           cudaGetErrorString(___call__ret_code__), cudaGetErrorName(___call__ret_code__), \
           ___call__ret_code__);                                                           \
      abort();                                                                             \
    }                                                                                      \
  } while (0)

#define checkKernel(...)                 \
  do {                                   \
    { (__VA_ARGS__); }                   \
    checkRuntime(cudaPeekAtLastError()); \
  } while (0)

enum class NormType : int { None = 0, MeanStd = 1, AlphaBeta = 2 };


/* 归一化操作，可以支持均值标准差，alpha beta */
struct Norm {
  float mean[3];
  float std[3];
  float alpha, beta;
  NormType type = NormType::None;

  // out = (x * alpha - mean) / std
  static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f);

  // out = x * alpha + beta
  static Norm alpha_beta(float alpha = 1 / 255.0f, float beta = 0);
};

Norm Norm::mean_std(const float mean[3], const float std[3], float alpha) {
  Norm out;
  out.type = NormType::MeanStd;
  out.alpha = alpha;
  memcpy(out.mean, mean, sizeof(out.mean));
  memcpy(out.std, std, sizeof(out.std));
  return out;
}

Norm Norm::alpha_beta(float alpha, float beta) {
  Norm out;
  out.type = NormType::AlphaBeta;
  out.alpha = alpha;
  out.beta = beta;
  return out;
}

inline int upbound(int n, int align = 32) { return (n + align - 1) / align * align; }

static __global__ void normalize_plane_kernel(uint8_t* src, float* dst, int dst_width, int dst_height, Norm norm) {
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;

    if (dx < dst_width && dy < dst_height) {
        int src_index = dy * dst_width + dx;
        uint8_t* src_pixel = src + src_index * 3;
        float c0 = src_pixel[0];
        float c1 = src_pixel[1];
        float c2 = src_pixel[2];

        if (norm.type == NormType::MeanStd) {
            c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
            c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
            c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
        }
        else if (norm.type == NormType::AlphaBeta) {
            c0 = c0 * norm.alpha + norm.beta;
            c1 = c1 * norm.alpha + norm.beta;
            c2 = c2 * norm.alpha + norm.beta;
        }
        int area = dst_width * dst_height;
        float* pdst_c0 = dst + src_index;
        float* pdst_c1 = pdst_c0 + area;
        float* pdst_c2 = pdst_c1 + area;
        *pdst_c0 = c0;
        *pdst_c1 = c1;
        *pdst_c2 = c2;
    }
}

static void normalize_plane(uint8_t* src, float* dst, int dst_width, int dst_height, const Norm& norm, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);

    checkKernel(normalize_plane_kernel << <grid, block, 0, stream >> > (src, dst, dst_width, dst_height, norm));
}


class InferImpl : public Infer {
 public:
  shared_ptr<trt::Infer> trt_;
  vector<Image> images_;
  vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;
  trt::Memory<float> input_buffer_;
  vector<trt::Memory<float>> outputs_buffer_;
  Norm normalize_;
  vector<int> input_dims_;
  vector<vector<int>> outputs_dims_;
  vector<void*> bindings_;
  int nb_, network_input_width_, network_input_height_, outputs_num_;
  bool isdynamic_model_ = false;

  virtual ~InferImpl() = default;

  void copyToDeviceAsync(void* dst, const void* src, size_t size, void* stream) {
      checkRuntime(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream));
  }

  void copyToHostAsync(void* dst, const void* src, size_t size, void* stream) {
      checkRuntime(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, (cudaStream_t)stream));
  }

  void set_input(uint8_t* data, int nChannel, int imgWid, int imgHei, bool bBGR = false) override {
      this->images_.push_back(Image(data, nChannel, imgWid, imgHei));
  }

  void set_input(float* data, int nChannel, int imgWid, int imgHei, bool bBGR = false) override {
      this->images_.push_back(Image(data, nChannel, imgWid, imgHei));
  }

  std::vector<int> get_input_dims(int inputIndex = 0) override {
      if (trt_->is_input(inputIndex)) {
          auto dims = trt_->static_dims(inputIndex);
          return dims;
      }
      return std::vector<int>();
  }

  std::vector<int> get_output_dims(int outputIndex) override {
      for (int index = 0, i = 0; i < nb_; i++) {
          if (!trt_->is_input(i)) {
              if (index == outputIndex) {
                  auto dims = trt_->static_dims(i);
                  return dims;
              }
              index++;
          }
      }
      return std::vector<int>();
  }

  int get_input_dims_count() override {
      int count = 0;
      for (int i = 0; i < nb_; i++) {
          if (trt_->is_input(i)) count++;
      }
      return count;
  }

  int get_output_dims_count() override {
      int count = 0;
      for (int i = 0; i < nb_; i++) {
          if (!trt_->is_input(i)) count++;
      }
      return count;
  }


  void adjust_memory(int batch_size) override {
    // //the inference batch_size
    for (int i=0; i < nb_; i++) {
        if (trt_->is_input(i)) {
            input_buffer_.gpu(trt_->get_head_byteSize(i));
        }else {
            if ((int)outputs_buffer_.size() < outputs_num_) {
                for (int j = outputs_buffer_.size(); j < outputs_num_; j++) {
                    outputs_buffer_.push_back(trt::Memory<float>());
                    size_t output_buffer_size = trt_->get_head_byteSize(i);
                    outputs_buffer_[j].gpu(output_buffer_size);
                    outputs_buffer_[j].cpu(output_buffer_size);
                }
            }
        }
    }
    if ((int)preprocess_buffers_.size() < batch_size) {
      for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
        preprocess_buffers_.push_back(make_shared<trt::Memory<unsigned char>>());
    }
  }

  void preprocess(int ibatch, const Image& image, shared_ptr<trt::Memory<unsigned char>> preprocess_buffer, void* stream) {
      size_t input_numel = network_input_width_ * network_input_height_ * 3;
      float* input_device = input_buffer_.gpu() + ibatch * input_numel;
      size_t size_image = image.width * image.height * 3;
      uint8_t* image_device = (uint8_t*)(preprocess_buffer->gpu(size_image));
      copyToDeviceAsync(image_device, image.bgrptr, size_image, (cudaStream_t)stream);
      normalize_plane(image_device, input_device, network_input_width_, network_input_height_, normalize_, (cudaStream_t)stream);
  }

  HC_AI_Result infer() override {
      int num_image = this->images_.size();
      if (num_image == 0) return HC_AI_Result{};
      int infer_batch_size = input_dims_[0];
      
      if (infer_batch_size != num_image) {
          if (isdynamic_model_) {
              infer_batch_size = num_image;
              for (int i = 0; i < nb_; i++) {
                  if (trt_->is_input(i)) {
                      input_dims_[0] = num_image;
                      if (!trt_->set_run_dims(i, input_dims_)) return HC_AI_Result{};
                  }
                  else {
                      outputs_dims_[i][0] = num_image;
                      if (!trt_->set_run_dims(i, outputs_dims_[i])) return HC_AI_Result{};
                  }
              }
          }else {
              if (infer_batch_size < num_image) {
                  INFO(
                      "When using static shape model, number of images[%d] must be "
                      "less than or equal to the maximum batch[%d].",
                      num_image, infer_batch_size);
                  return HC_AI_Result{};
              }else {
                  infer_batch_size = std::min(num_image, input_dims_[0]);
              }
          }
      }
      adjust_memory(infer_batch_size);

      cudaStream_t preprocess_stream;
      cudaStreamCreate(&preprocess_stream);
      cudaStream_t inference_stream;
      cudaStreamCreate(&inference_stream);

      for (int i = 0; i < num_image; ++i) {
          preprocess(i, this->images_[i], preprocess_buffers_[i], preprocess_stream);
      }
      checkRuntime(cudaStreamSynchronize(preprocess_stream));

      bindings_.push_back(input_buffer_.gpu());
      for (int i = 0; i < outputs_num_; i++) {
          bindings_.push_back(outputs_buffer_[i].gpu());
      }
      if (!trt_->forward(bindings_, inference_stream)) {
          INFO("Failed to tensorRT forward.");
          return HC_AI_Result{};
      }

      std::vector<void*> results;
      for (int i = 0; i < outputs_num_; i++) {
          copyToHostAsync(outputs_buffer_[i].cpu(), outputs_buffer_[i].gpu(), outputs_buffer_[i].gpu_bytes(), inference_stream);
          results.push_back(outputs_buffer_[i].cpu());
      }
      checkRuntime(cudaStreamSynchronize(inference_stream));

      cudaStreamDestroy(preprocess_stream);
      cudaStreamDestroy(inference_stream);
      images_.clear();
      return HC_AI_Result{ results };
  }

  bool load(const string &engine_file) {
    trt_ = trt::load(engine_file);
    if (trt_ == nullptr) return false;
    trt_->print();
    nb_ = trt_->num_bindings();
    outputs_num_ = get_output_dims_count();
    for (int i = 0; i < nb_; i++) {
        if (trt_->is_input(i)){ 
            input_dims_ = get_input_dims(i); 
        }else {
            outputs_dims_.push_back(trt_->static_dims(i));
        }
    }
    network_input_width_ = input_dims_[3];
    network_input_height_ = input_dims_[2];
    isdynamic_model_ = trt_->has_dynamic_dim();
    normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f);
    return true;
  }
};

shared_ptr<trtengine::Infer> load_engine(const string &engine_file) {
    auto impl = std::make_shared<InferImpl>();
    if (!impl->load(engine_file)) {
        impl = nullptr;
    }
    return impl;
}

};  // namespace trtengine
