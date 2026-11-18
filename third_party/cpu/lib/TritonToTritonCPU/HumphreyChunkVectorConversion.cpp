
#include "TypeConverter.h"
#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
//include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Analysis/SliceAnalysis.h"

#include <numeric>


#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <functional>

namespace mlir {
namespace triton {
namespace cpu{
#define GEN_PASS_DEF_HUMPHREYCHUNKVECTORCONVERSION
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class ChunkVectorConversionTarget : public ConversionTarget {
public:
  explicit ChunkVectorConversionTarget(MLIRContext &ctx, TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();


    //addIllegalOp<triton::?>();
  }
};

// converter
struct ChunkVectorConversion : public OpRewritePattern<vector::MaskedStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  // Layout We know
  // mask loads ...
  // mul
  // add
  // div
  // store 

  LogicalResult matchAndRewrite(
    vector::MaskedStoreOp op,
    PatternRewriter &rewriter) const override {

      Value divResult = op.getValueToStore();
      llvm::SetVector<Operation *> slice;
      

      Operation *root = divResult.getDefiningOp();


      if (failed(mlir::getBackwardSlice(root, &slice)))
          return failure();

      for (Operation *producer : slice)
          if (auto load = dyn_cast<vector::MaskedLoadOp>(producer)) {
            llvm::errs() << "\nLOAD:\n";
            load.dump();

            llvm::errs() << "Mask type: ";
            load.getMask().getType().dump();

            llvm::errs() << "Result type: ";
            load.getResult().getType().dump();
          }




          return llvm::failure();
        }


};

// pass
struct HumphreyChunkVectorConversion : public triton::cpu::impl::HumphreyChunkVectorConversionBase<HumphreyChunkVectorConversion> {
    HumphreyChunkVectorConversion() = default;


  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();


    TritonToTritonCPUTypeConverter typeConverter;
    ChunkVectorConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<ChunkVectorConversion>(context);

    if (failed(applyPatternsGreedily(mod, std::move(patterns)))) {
        return signalPassFailure();
    }}
  
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createHumphreyChunkVectorConversion() {
  return std::make_unique<HumphreyChunkVectorConversion>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
