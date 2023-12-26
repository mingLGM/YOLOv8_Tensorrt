#ifndef __TRTENGINE_HPP__
#define __TRTENGINE_HPP__

#include <future>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "IEngine.h"
#include "TensorRTEngine.h"


namespace trtengine {
struct Image {
  const void *bgrptr = nullptr;
  int channels = 3, width = 0, height = 0;

  Image() = default;
  Image(const void *bgrptr, int channels, int width, int height) : bgrptr(bgrptr), channels(channels), width(width), height(height) {}
};

class  Infer : public IEngine {
 public:
  virtual HC_AI_Result infer() = 0;
  virtual std::vector<int> get_input_dims(int inputIndex = 0) = 0;
  virtual std::vector<int> get_output_dims(int outIdex = 0) = 0;
  virtual int get_input_dims_count() = 0;
  virtual int get_output_dims_count() = 0;
  virtual void set_input(uint8_t* data, int nChannel, int imgWid, int imgHei, bool bBGR = false) = 0;
  virtual void set_input(float* data, int nChannel, int imgWid, int imgHei, bool bBGR = false) = 0;
private:
  virtual void adjust_memory(int batch_size) = 0;
};
 std::shared_ptr<Infer> load_engine(const std::string &engine_file);
};  // namespace trtengine

#endif  // __TRTENGINE_HPP__