#include "operators/matmul.h"
#include "core/common.h"
#include "core/tensor.h"
#include <optional>
#include <utility>
#include <vector>

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        IT_ASSERT(inputs.size() >= 2);
        auto shapeA = inputs[0]->getDims();
        auto shapeB = inputs[1]->getDims();
        IT_ASSERT(shapeA.size() >= 2 && shapeB.size() >= 2);
        if (transA) {
            std::swap(shapeA[shapeA.size() - 1], shapeA[shapeA.size() - 2]);
        }
        if (transB) {
            std::swap(shapeB[shapeB.size() - 1], shapeB[shapeB.size() - 2]);
        }
        
        Shape shape(shapeA.begin(), shapeA.end() - 2);
    
        shape.push_back(shapeA[shapeA.size() - 2]);
        shape.push_back(shapeB[shapeB.size() - 1]);
        
        return std::make_optional(vector<Shape>{shape});
    }

} // namespace infini