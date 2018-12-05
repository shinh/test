#include <chainerx/array.h>
#include <chainerx/context.h>
#include <chainerx/routines/creation.h>

#include <chainerx/testing/array.h>

#include <iostream>
#include <vector>

namespace {

std::shared_ptr<void> MakeSharedPtrData(chainerx::Dtype dtype, chainerx::Shape shape, const void* src) {
    int64_t size = chainerx::GetItemSize(dtype) * shape.GetTotalSize();
    std::shared_ptr<void> data(new char[size], std::default_delete<char[]>());
    std::memcpy(data.get(), src, size);
    return data;
}

}  // namespace

chainerx::Array MakeArray(chainerx::Dtype dtype, chainerx::Shape shape, const void* src) {
    std::shared_ptr<void> data(MakeSharedPtrData(dtype, shape, src));
    chainerx::Array array(chainerx::FromContiguousHostData(shape, dtype, data));
    return array;
}

int main() {
    chainerx::Context ctx;
    chainerx::SetGlobalDefaultContext(&ctx);

    std::cerr << MakeArray(chainerx::Dtype::kInt64, chainerx::Shape{3, 2}, std::vector<int64_t>({1, 2, 3, 5, 7, 9}).data()) << std::endl;
    std::cerr << MakeArray(chainerx::Dtype::kFloat32, chainerx::Shape{2, 3}, std::vector<float>({1, 2, 3, 5, 7, 9}).data()) << std::endl;
    // Should have been kFloat64.
    std::cerr << MakeArray(chainerx::Dtype::kFloat32, chainerx::Shape{2, 3}, std::vector<double>({1, 2, 3, 5, 7, 9}).data()) << std::endl;

    // Or with ArrayBuilder.
    std::cerr << chainerx::testing::BuildArray(chainerx::Shape{2, 3}).WithData(std::vector<int>{4, 5, 6, 10, 12, 14}).Build() << std::endl;
    std::cerr << chainerx::testing::BuildArray(chainerx::Shape{2, 3}).WithData(std::vector<double>{4, 5, 6, 10, 12, 14}).Build() << std::endl;
}
