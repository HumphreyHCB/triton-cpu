#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "cpu/include/ScalarizePass/ScalarizeInterface.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_HUMPHREYSSPLITINTEGERADDITION
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class AdditionOpConversionTarget : public ConversionTarget {
public:
  explicit AdditionOpConversionTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<arith::ArithDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    addIllegalOp<mlir::arith::AddIOp>();
  }
};

struct SplitIntegerAdditionRewrite : public OpRewritePattern<arith::AddIOp> {
    using OpRewritePattern::OpRewritePattern;


  LogicalResult matchAndRewrite(
    arith::AddIOp op,
    PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    
    Value A = op.getLhs();

    Value B = op.getRhs();

    Value zero = arith::ConstantOp::create(
    rewriter,
    loc,
    rewriter.getZeroAttr(op.getType())); // this cause if the og add's lhs or rhs was a tensor then we nee to watch for hardcoding an into, instead this is more
        

    Value negate = arith::SubIOp::create(rewriter,loc, zero, B);

    Value res = arith::SubIOp::create(rewriter,loc, A, negate);

    

    rewriter.replaceOp(op, res);




    return success();
    }
    
  };


struct HumphreysSplitIntegerAddition : public triton::cpu::impl::HumphreysSplitIntegerAdditionBase<HumphreysSplitIntegerAddition> {
  HumphreysSplitIntegerAddition() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();


    AdditionOpConversionTarget convTarget(*context);
    TritonToTritonCPUTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    patterns.add<SplitIntegerAdditionRewrite>(context);

    if (failed(applyPatternsGreedily(mod, std::move(patterns)))) {
        return signalPassFailure();
    }}
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createHumphreysSplitIntegerAddition() {
  return std::make_unique<HumphreysSplitIntegerAddition>();
}


} // namespace cpu
} // namespace triton
} // namespace mlir
