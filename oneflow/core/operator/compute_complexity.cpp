/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/operator/compute_complexity.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/compute_complexity_fn_context.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/user/ops/math_unary_elementwise_seq.h"

namespace oneflow {

double GetDefaultPiecewiseLineVal(int x) {
  GenericPiecewiseLine2D functor{17.55002336908022, 2.000003612364722, 3.7338046849336295e-07,
                                 0.9863286392959302};
  double value = functor.GetVal(x);
  return value;
}

double GetDefaultPiecewise2PlaneVal(int m, int n) {
  GenericPiecewise2Plane3D functor{
      {0, 0, -1, 2.0}, {0.9620057779088316, 0.9538400739098106, -1, -14.869189017712706}};

  double value = functor.GetVal(m, n);
  return value;
}

double GetDefaultPiecewise3PlaneVal(int m, int n) {
  GenericPiecewise3Plane3D functor{
      {0, 0, -1, 2.0},
      {0.11199696007460436, 0.9841384346704612, -1, -10.470334834343731},
      {0.9458215352874206, 1.014120028508652, -1, -14.656635853594809}};

  double value = functor.GetVal(m, n);
  return value;
}

// Maybe<double> GetUnaryOpComputeComplexity(user_op::ComputeComplexityFnContext* ctx) {
//   const auto& inputs = ctx->inputs();
//   CHECK_EQ_OR_RETURN(inputs.size(), 1);  // NOLINT
//   int m = ctx->Shape4ArgNameAndIndex(inputs[0].first, inputs[0].second).elem_cnt();
//   double cost = GetPiecewiseLineValDefault(m);

//   const auto& outputs = ctx->outputs();
//   std::cout << "inputs(" << inputs.size() << ") \t:";
//   for (auto& p : inputs) { std::cout << "\t(" << p.first << ", " << p.second << ")\n"; }
//   std::cout << "outputs(" << outputs.size() << ") \t:";
//   for (auto& p : outputs) { std::cout << "\t(" << p.first << ", " << p.second << ")\n"; }
//   std::cout << ctx->user_op_conf().op_name() << " Op::GetComputeComplexity: "
//             << "m = " << m << "\tcost = " << cost << std::endl;
//   return cost;
// }

// Maybe<double> GetBinaryOpComputeComplexity(user_op::ComputeComplexityFnContext* ctx) {
//   const auto& inputs = ctx->inputs();
//   CHECK_EQ_OR_RETURN(inputs.size(), 1);  // NOLINT
//   int m = ctx->Shape4ArgNameAndIndex(inputs[0].first, inputs[0].second).elem_cnt();
//   double cost = GetPiecewiseLineValDefault(m);

//   const auto& outputs = ctx->outputs();
//   std::cout << "inputs(" << inputs.size() << ") \t:";
//   for (auto& p : inputs) { std::cout << "\t(" << p.first << ", " << p.second << ")\n"; }
//   std::cout << "outputs(" << outputs.size() << ") \t:";
//   for (auto& p : outputs) { std::cout << "\t(" << p.first << ", " << p.second << ")\n"; }
//   std::cout << ctx->user_op_conf().op_name() << " Op::GetComputeComplexity: "
//             << "m = " << m << "\tcost = " << cost << std::endl;
//   return cost;
// }

/*static*/ Maybe<double> SoftmaxOp::GetComputeComplexity(user_op::ComputeComplexityFnContext* ctx) {
  const Shape& input_shape = ctx->Shape4ArgNameAndIndex("in", 0);

  int m = input_shape.Count(0, input_shape.NumAxes() - 1);
  int n = input_shape.At(input_shape.NumAxes() - 1);

  GenericPiecewise3Plane3D functor{
      {0, 0, -1, 2.0},
      {0.11199696007460436, 0.9841384346704612, -1, -10.470334834343731},
      {0.9458215352874206, 1.014120028508652, -1, -14.656635853594809}};

  double computation_cost = functor.GetVal(m, n);
  std::cout << "SoftmaxOp::GetComputeComplexity: "
            << "m = " << m << "\tn = " << n << "\tcost = " << computation_cost << std::endl;

  return computation_cost;
}

/*static*/ Maybe<double> LogSoftmaxOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  const Shape& input_shape = ctx->Shape4ArgNameAndIndex("in", 0);

  int m = input_shape.Count(0, input_shape.NumAxes() - 1);
  int n = input_shape.At(input_shape.NumAxes() - 1);

  GenericPiecewise3Plane3D functor{
      {0, 0, -1, 2.0},
      {0.11199696007460436, 0.9841384346704612, -1, -10.470334834343731},
      {0.9458215352874206, 1.014120028508652, -1, -14.656635853594809}};

  double computation_cost = functor.GetVal(m, n);

  std::cout << "LogSoftmaxOp::GetComputeComplexity: "
            << "m = " << m << "\tn = " << n << "\tcost = " << computation_cost << std::endl;
  return computation_cost;
}

/*static*/ Maybe<double> LayerNormOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  const Shape& input_shape = ctx->Shape4ArgNameAndIndex("x", 0);

  int m = input_shape.Count(0, input_shape.NumAxes() - 1);
  int n = input_shape.At(input_shape.NumAxes() - 1);

  GenericPiecewise3Plane3D functor{
      {0, 0, -1, 2.0},
      {0.1296369900299345, 0.8961367288344754, -1, -9.230739369938997},
      {0.9840272420666372, 0.9775039645900081, -1, -14.822361383446248}};

  double computation_cost = functor.GetVal(m, n);

  std::cout << "LayerNormOp::GetComputeComplexity: "
            << "m = " << m << "\tn = " << n << "\tcost = " << computation_cost << std::endl;
  return computation_cost;
}

/*static*/ Maybe<double> ReduceOpGetComputeComplexity(user_op::ComputeComplexityFnContext* ctx) {
  const Shape& input_shape = ctx->Shape4ArgNameAndIndex("input_tensor", 0);
  CHECK_OR_RETURN(input_shape.NumAxes() > 0);
  int m = input_shape.Count(0, input_shape.NumAxes() - 1);
  int n = input_shape.At(input_shape.NumAxes() - 1);

  GenericPiecewise3Plane3D functor{
      {0, 0, -1, 2.0},
      {0.2690958290916445, 0.927487932894259, -1, -7.5874540472625585},
      {0.7839981384068997, 0.8125043077388764, -1, -10.560222182961828}};

  double computation_cost = functor.GetVal(m, n);

  std::cout << ctx->user_op_conf().op_name() << " m = " << m << "\tn = " << n
            << "\tcost = " << computation_cost << std::endl;
  return computation_cost;
}

// #define REDUCE_COMPUTE_COST_OP_SEQ   \
//   OF_PP_MAKE_TUPLE_SEQ(ReduceAll)    \
//   OF_PP_MAKE_TUPLE_SEQ(ReduceAny)    \
//   OF_PP_MAKE_TUPLE_SEQ(ReduceProd)   \
//   OF_PP_MAKE_TUPLE_SEQ(ReduceMax)    \
//   OF_PP_MAKE_TUPLE_SEQ(ReduceMin)    \
//   OF_PP_MAKE_TUPLE_SEQ(ReduceNanSum) \
//   OF_PP_MAKE_TUPLE_SEQ(ReduceSum)

/*static*/ Maybe<double> ReduceAllOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return ReduceOpGetComputeComplexity(ctx);
}
/*static*/ Maybe<double> ReduceAnyOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return ReduceOpGetComputeComplexity(ctx);
}
/*static*/ Maybe<double> ReduceProdOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return ReduceOpGetComputeComplexity(ctx);
}
/*static*/ Maybe<double> ReduceMaxOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return ReduceOpGetComputeComplexity(ctx);
}
/*static*/ Maybe<double> ReduceMinOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return ReduceOpGetComputeComplexity(ctx);
}
/*static*/ Maybe<double> ReduceNanSumOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return ReduceOpGetComputeComplexity(ctx);
}
/*static*/ Maybe<double> ReduceSumOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return ReduceOpGetComputeComplexity(ctx);
}

/*static*/ Maybe<double> NLLOp::GetComputeComplexity(user_op::ComputeComplexityFnContext* ctx) {
  const Shape& input_shape = ctx->Shape4ArgNameAndIndex("input", 0);

  int n = input_shape.Count(0, input_shape.NumAxes() - 1);
  int m = input_shape.At(input_shape.NumAxes() - 1);

  GenericPiecewise2Plane3D functor{
      {0, 0, -1, 2.0}, {0.2717762811278621, 0.9735486408486381, -1, -13.529939021992103}};

  double computation_cost = functor.GetVal(m, n);

  std::cout << "NLLOp::GetComputeComplexity: "
            << "m = " << m << "\tn = " << n << "\tcost = " << computation_cost << std::endl;
  return computation_cost;
}

/*static*/ Maybe<double> BroadcastLikeOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  const Shape& input_shape = ctx->Shape4ArgNameAndIndex("like", 0);

  int m = input_shape.Count(0, input_shape.NumAxes() - 1);
  int n = input_shape.At(input_shape.NumAxes() - 1);

  // GenericPiecewise2Plane3D functor{
  //     {0, 0, -1, 2.0}, {0.9620057779088316, 0.9538400739098106, -1, -14.869189017712706}};
  // double computation_cost = functor.GetVal(m, n);

  double computation_cost = GetDefaultPiecewise2PlaneVal(m, n);

  std::cout << ctx->user_op_conf().op_name() << " ::GetComputeComplexity: "
            << "m = " << m << "\tn = " << n << "\tcost = " << computation_cost << std::endl;
  return computation_cost;
}

/*static*/ Maybe<double> FusedBiasAddGeluOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  const Shape& a_shape = ctx->Shape4ArgNameAndIndex("a", 0);
  const Shape& b_shape = ctx->Shape4ArgNameAndIndex("b", 0);

  int n = b_shape.elem_cnt();
  int m = a_shape.elem_cnt() / n;

  double computation_cost = GetDefaultPiecewise2PlaneVal(m, n);
  std::cout << ctx->user_op_conf().op_name() << " ::GetComputeComplexity: "
            << "m = " << m << "\tn = " << n << "\tcost = " << computation_cost << std::endl;
  return computation_cost;
}

/*static*/ Maybe<double> FusedBiasAddMaskScaleOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  const Shape& a_shape = ctx->Shape4ArgNameAndIndex("a", 0);
  const Shape& b_shape = ctx->Shape4ArgNameAndIndex("b", 0);

  int n = b_shape.elem_cnt();
  int m = a_shape.elem_cnt() / n;

  double computation_cost = GetDefaultPiecewise2PlaneVal(m, n);
  std::cout << ctx->user_op_conf().op_name() << " ::GetComputeComplexity: "
            << "m = " << m << "\tn = " << n << "\tcost = " << computation_cost << std::endl;
  return computation_cost;
}

/*static*/ Maybe<double> FusedScaleMaskSoftmaxDropoutOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  const Shape& input_shape = ctx->Shape4ArgNameAndIndex("x", 0);

  int m = input_shape.Count(0, input_shape.NumAxes() - 1);
  int n = input_shape.At(input_shape.NumAxes() - 1);

  double computation_cost = GetDefaultPiecewise2PlaneVal(m, n);
  std::cout << ctx->user_op_conf().op_name() << " ::GetComputeComplexity: "
            << "m = " << m << "\tn = " << n << "\tcost = " << computation_cost << std::endl;
  return computation_cost;
}

// /*static*/ Maybe<double> BroadcastBinaryGetComputeComplexity(
//     user_op::ComputeComplexityFnContext* ctx) {
//   const Shape& input_shape1 = ctx->Shape4ArgNameAndIndex("x", 0);
//   const Shape& input_shape2 = ctx->Shape4ArgNameAndIndex("y", 0);

//   int m1 = input_shape1.Count(0, input_shape1.NumAxes() - 1);
//   int m2 = input_shape2.Count(0, input_shape2.NumAxes() - 1);

//   int n = input_shape1.At(input_shape1.NumAxes() - 1);
//   int m = std::max(m1, m2);

//   GenericPiecewise2Plane3D functor{
//       {0, 0, -1, 2.0}, {0.9575383365844942, 0.9766950173578609, -1, -14.973116182724954}};

//   double computation_cost = functor.GetVal(m, n);
//   std::cout << ctx->user_op_conf().op_name() << " ::GetComputeComplexity: "
//             << "m = " << m << "\tn = " << n << "\tcost = " << computation_cost << std::endl;
//   return computation_cost;
// }

#define BINARY_BROADCAST_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastAdd)          \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastSub)          \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastPow)          \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastDiv)          \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastMul)          \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastFmod)         \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastGreaterEqual) \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastLessEqual)    \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastNotEqual)     \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastLogicalOr)    \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastLogicalAnd)   \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastLogicalXor)   \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastEqual)        \
  OF_PP_MAKE_TUPLE_SEQ(BroadcastLess)

Maybe<double> BinaryBroadcastOpGetComputeComplexity(user_op::ComputeComplexityFnContext* ctx) {
  std::cout << ctx->user_op_conf().op_name() << " ::GetComputeComplexity: " << std::endl;
  const Shape& input_shape1 = ctx->Shape4ArgNameAndIndex("x", 0);
  const Shape& input_shape2 = ctx->Shape4ArgNameAndIndex("y", 0);
  const auto& inputs = ctx->inputs();
  std::cout << "inputs(" << inputs.size() << ") \t:";
  for (auto& p : inputs) {
    std::cout << "\t(" << p.first << ", " << p.second
              << "), shape: " << ctx->Shape4ArgNameAndIndex(p.first, p.second)
              << ", elemcnt: " << ctx->Shape4ArgNameAndIndex(p.first, p.second).elem_cnt()
              << std::endl;
  }
  if (input_shape1.NumAxes() == 0 && input_shape2.NumAxes() == 0) { return 4; }
  int m1 = input_shape1.NumAxes() == 1 ? 1 : input_shape1.Count(0, input_shape1.NumAxes() - 1);
  int m2 = input_shape2.NumAxes() == 1 ? 1 : input_shape2.Count(0, input_shape2.NumAxes() - 1);
  int n = input_shape1.At(input_shape1.NumAxes() - 1);
  int m = std::max(m1, m2);
  GenericPiecewise2Plane3D functor{
      {0, 0, -1, 2.0}, {0.9575383365844942, 0.9766950173578609, -1, -14.973116182724954}};
  double computation_cost = functor.GetVal(m, n);
  std::cout << ctx->user_op_conf().op_name() << " ::GetComputeComplexity: "
            << "m = " << m << "\tn = " << n << "\tcost = " << computation_cost << std::endl;
  return computation_cost;
}

#define REGISTER_BINARY_ELEMENTWISE_OP_GETCOMPUTECOMPLEXITY(func_prefix) \
  /* static */ Maybe<double> func_prefix##Op::GetComputeComplexity(      \
      user_op::ComputeComplexityFnContext* ctx) {                        \
    return BinaryBroadcastOpGetComputeComplexity(ctx);                   \
  }

OF_PP_FOR_EACH_TUPLE(REGISTER_BINARY_ELEMENTWISE_OP_GETCOMPUTECOMPLEXITY, BINARY_BROADCAST_SEQ)

Maybe<double> GetOpComputeComplexityByName(user_op::ComputeComplexityFnContext* ctx) { return 4; }

#define CONSTANT_CPU_COMPUTE_COST_OP_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(Identity)         \
  OF_PP_MAKE_TUPLE_SEQ(Narrow)           \
  OF_PP_MAKE_TUPLE_SEQ(Reshape)          \
  OF_PP_MAKE_TUPLE_SEQ(Transpose)

#define CONSTANT_GPU_COMPUTE_COST_OP_SEQ OF_PP_MAKE_TUPLE_SEQ(Slice)

#define REGISTER_CONSTANT_OP_GETCOMPUTECOMPLEXITY(func_prefix)      \
  /* static */ Maybe<double> func_prefix##Op::GetComputeComplexity( \
      user_op::ComputeComplexityFnContext* ctx) {                   \
    return GetConstantOpComputeComplexity(ctx);                     \
  }

Maybe<double> GetConstantOpComputeComplexity(user_op::ComputeComplexityFnContext* ctx) { return 4; }

OF_PP_FOR_EACH_TUPLE(REGISTER_CONSTANT_OP_GETCOMPUTECOMPLEXITY, CONSTANT_GPU_COMPUTE_COST_OP_SEQ)
OF_PP_FOR_EACH_TUPLE(REGISTER_CONSTANT_OP_GETCOMPUTECOMPLEXITY, CONSTANT_CPU_COMPUTE_COST_OP_SEQ)

#define DEFAULT_PIECEWISE_LINE_OP_SEQ OF_PP_MAKE_TUPLE_SEQ(NotEqualZero)

#define REGISTER_DEFAULT_PIECEWISE_LINE_OP_GETCOMPUTECOMPLEXITY(func_prefix)                  \
  /* static */ Maybe<double> func_prefix##Op::GetComputeComplexity(                           \
      user_op::ComputeComplexityFnContext* ctx) {                                             \
    std::cout << ctx->user_op_conf().op_name() << " Op::GetComputeComplexity: " << std::endl; \
    const auto& inputs = ctx->inputs();                                                       \
    std::cout << "inputs(" << inputs.size() << ") \t:";                                       \
    for (auto& p : inputs) { std::cout << "\t(" << p.first << ", " << p.second << ")\n"; }    \
    CHECK_EQ_OR_RETURN(inputs.size(), 1);                                                     \
    int m = ctx->Shape4ArgNameAndIndex(inputs[0].first, inputs[0].second).elem_cnt();         \
    double cost = GetDefaultPiecewiseLineVal(m);                                              \
    const auto& outputs = ctx->outputs();                                                     \
    std::cout << "outputs(" << outputs.size() << ") \t:";                                     \
    for (auto& p : outputs) { std::cout << "\t(" << p.first << ", " << p.second << ")\n"; }   \
    std::cout << " Op::GetComputeComplexity Cost: "                                           \
              << "m = " << m << "\tcost = " << cost << std::endl;                             \
    return cost;                                                                              \
  }

// OF_PP_FOR_EACH_TUPLE(REGISTER_DEFAULT_PIECEWISE_LINE_OP_GETCOMPUTECOMPLEXITY,
//                      DEFAULT_PIECEWISE_LINE_OP_SEQ)

#define UNARY_PIECEWISE_LINE_OP_SEQ                                                    \
  OF_PP_MAKE_TUPLE_SEQ(17.73980, 2.00001, 0.00000, 1.00809, Abs)                       \
  OF_PP_MAKE_TUPLE_SEQ(18.34856, 2.79192, 0.05273, 0.98660, Acos)                      \
  OF_PP_MAKE_TUPLE_SEQ(18.41847, 2.86077, 0.04658, 0.98669, Acosh)                     \
  OF_PP_MAKE_TUPLE_SEQ(18.34858, 2.79192, 0.05273, 0.98660, Asin)                      \
  OF_PP_MAKE_TUPLE_SEQ(18.36801, 2.81091, 0.03264, 0.98676, Asinh)                     \
  OF_PP_MAKE_TUPLE_SEQ(18.33269, 2.66259, 0.04714, 1.00137, Atan)                      \
  OF_PP_MAKE_TUPLE_SEQ(17.55332, 2.12541, 0.01026, 0.97109, Atanh)                     \
  OF_PP_MAKE_TUPLE_SEQ(17.74157, 2.00001, 0.00000, 1.00824, Ceil)                      \
  OF_PP_MAKE_TUPLE_SEQ(17.75265, 2.12745, 0.01026, 0.99466, Celu)                      \
  OF_PP_MAKE_TUPLE_SEQ(17.76072, 2.12754, 0.01026, 0.99538, Cos)                       \
  OF_PP_MAKE_TUPLE_SEQ(17.76196, 2.12755, 0.01026, 0.99549, Cosh)                      \
  OF_PP_MAKE_TUPLE_SEQ(17.16970, 5.99682, 0.15360, 0.95612, Digamma)                   \
  OF_PP_MAKE_TUPLE_SEQ(17.75521, 2.12748, 0.01026, 0.99495, Elu)                       \
  OF_PP_MAKE_TUPLE_SEQ(17.76441, 2.12757, 0.01026, 0.99565, Erf)                       \
  OF_PP_MAKE_TUPLE_SEQ(17.53528, 2.12522, 0.01026, 0.96891, Erfc)                      \
  OF_PP_MAKE_TUPLE_SEQ(17.73802, 2.00001, 0.00000, 1.00794, Exp)                       \
  OF_PP_MAKE_TUPLE_SEQ(17.73989, 2.00001, 0.00000, 1.00816, Exp2)                      \
  OF_PP_MAKE_TUPLE_SEQ(17.75874, 2.12752, 0.01026, 0.99521, Expm1)                     \
  OF_PP_MAKE_TUPLE_SEQ(17.74065, 2.00001, 0.00000, 1.00816, Floor)                     \
  OF_PP_MAKE_TUPLE_SEQ(17.75889, 2.12752, 0.01026, 0.99521, Gelu)                      \
  OF_PP_MAKE_TUPLE_SEQ(17.74873, 2.00001, 0.00000, 1.00884, HardShrink)                \
  OF_PP_MAKE_TUPLE_SEQ(18.37832, 2.71019, 0.04618, 1.00109, Hardsigmoid)               \
  OF_PP_MAKE_TUPLE_SEQ(18.37832, 2.71019, 0.04618, 1.00110, Hardswish)                 \
  OF_PP_MAKE_TUPLE_SEQ(17.74742, 2.00104, 0.00008, 1.00874, Hardtanh)                  \
  OF_PP_MAKE_TUPLE_SEQ(17.74897, 2.00001, 0.00000, 1.00888, LeakyRelu)                 \
  OF_PP_MAKE_TUPLE_SEQ(18.58739, 3.35881, 0.06663, 0.95439, Lgamma)                    \
  OF_PP_MAKE_TUPLE_SEQ(18.58739, 3.35881, 0.06663, 0.95439, Trigamma)                  \
  OF_PP_MAKE_TUPLE_SEQ(17.76052, 2.12753, 0.01026, 0.99529, Log)                       \
  OF_PP_MAKE_TUPLE_SEQ(17.76103, 2.12754, 0.01026, 0.99536, Log10)                     \
  OF_PP_MAKE_TUPLE_SEQ(17.75472, 2.12747, 0.01026, 0.99490, Log1p)                     \
  OF_PP_MAKE_TUPLE_SEQ(17.76213, 2.12755, 0.01026, 0.99548, Log2)                      \
  OF_PP_MAKE_TUPLE_SEQ(17.75926, 2.12752, 0.01026, 0.99525, LogSigmoid)                \
  OF_PP_MAKE_TUPLE_SEQ(18.26682, 2.80341, 0.03118, 0.97493, Mish)                      \
  OF_PP_MAKE_TUPLE_SEQ(17.74275, 2.00001, 0.00000, 1.00838, Negative)                  \
  OF_PP_MAKE_TUPLE_SEQ(17.72876, 2.12721, 0.01026, 0.99226, Reciprocal)                \
  OF_PP_MAKE_TUPLE_SEQ(17.72876, 2.12721, 0.01026, 0.99226, ReciprocalNoNan)           \
  OF_PP_MAKE_TUPLE_SEQ(17.74070, 2.00001, 0.00000, 1.00819, Relu)                      \
  OF_PP_MAKE_TUPLE_SEQ(17.73965, 2.00001, 0.00000, 1.00807, Round)                     \
  OF_PP_MAKE_TUPLE_SEQ(17.73965, 2.00001, 0.00000, 1.00807, Rint)                      \
  OF_PP_MAKE_TUPLE_SEQ(17.73780, 2.00001, 0.00000, 1.00794, Rsqrt)                     \
  OF_PP_MAKE_TUPLE_SEQ(17.74914, 2.00001, 0.00000, 1.00884, ScalarAdd)                 \
  OF_PP_MAKE_TUPLE_SEQ(17.74624, 2.00001, 0.00000, 1.00871, ScalarDiv)                 \
  OF_PP_MAKE_TUPLE_SEQ(17.74552, 2.00001, 0.00000, 1.00863, ScalarFmod)                \
  OF_PP_MAKE_TUPLE_SEQ(18.60377, 2.11949, 0.00924, 1.01279, ScalarLogicalAnd)          \
  OF_PP_MAKE_TUPLE_SEQ(18.60354, 2.11949, 0.00924, 1.01284, ScalarLogicalEqual)        \
  OF_PP_MAKE_TUPLE_SEQ(18.60379, 2.11949, 0.00924, 1.01279, ScalarLogicalGreater)      \
  OF_PP_MAKE_TUPLE_SEQ(18.60377, 2.11949, 0.00924, 1.01278, ScalarLogicalGreaterEqual) \
  OF_PP_MAKE_TUPLE_SEQ(18.60371, 2.11949, 0.00924, 1.01280, ScalarLogicalLess)         \
  OF_PP_MAKE_TUPLE_SEQ(18.60375, 2.11949, 0.00924, 1.01279, ScalarLogicalLessEqual)    \
  OF_PP_MAKE_TUPLE_SEQ(18.60379, 2.11949, 0.00924, 1.01278, ScalarLogicalNotEqual)     \
  OF_PP_MAKE_TUPLE_SEQ(18.60377, 2.11949, 0.00924, 1.01279, ScalarLogicalOr)           \
  OF_PP_MAKE_TUPLE_SEQ(18.60355, 2.11949, 0.00924, 1.01284, ScalarLogicalXor)          \
  OF_PP_MAKE_TUPLE_SEQ(17.74883, 2.00001, 0.00000, 1.00883, ScalarMul)                 \
  OF_PP_MAKE_TUPLE_SEQ(18.38359, 2.94147, 0.04158, 0.97202, ScalarPow)                 \
  OF_PP_MAKE_TUPLE_SEQ(18.38246, 2.82543, 0.05223, 0.98662, ScalarReversePow)          \
  OF_PP_MAKE_TUPLE_SEQ(17.74340, 2.12736, 0.01026, 0.99381, Selu)                      \
  OF_PP_MAKE_TUPLE_SEQ(17.72236, 2.12714, 0.01026, 0.99167, Sigmoid)                   \
  OF_PP_MAKE_TUPLE_SEQ(17.74136, 2.00001, 0.00000, 1.00824, Sign)                      \
  OF_PP_MAKE_TUPLE_SEQ(18.30364, 2.63537, 0.04319, 1.00109, Silu)                      \
  OF_PP_MAKE_TUPLE_SEQ(17.76153, 2.12754, 0.01026, 0.99546, Sin)                       \
  OF_PP_MAKE_TUPLE_SEQ(17.83960, 2.24541, 0.01932, 0.99154, Sinh)                      \
  OF_PP_MAKE_TUPLE_SEQ(18.36345, 2.80643, 0.03118, 0.98676, Softplus)                  \
  OF_PP_MAKE_TUPLE_SEQ(17.73845, 2.00235, 0.00019, 1.00760, SoftShrink)                \
  OF_PP_MAKE_TUPLE_SEQ(17.91682, 2.32387, 0.02499, 0.99127, Softsign)                  \
  OF_PP_MAKE_TUPLE_SEQ(18.27264, 2.71745, 0.04618, 0.98648, Sqrt)                      \
  OF_PP_MAKE_TUPLE_SEQ(17.74051, 2.00001, 0.00000, 1.00815, Square)                    \
  OF_PP_MAKE_TUPLE_SEQ(17.76109, 2.12754, 0.01026, 0.99546, Tan)                       \
  OF_PP_MAKE_TUPLE_SEQ(17.73520, 2.12727, 0.01026, 0.99301, Tanh)                      \
  OF_PP_MAKE_TUPLE_SEQ(17.74881, 2.00001, 0.00000, 1.00884, Threshold)                 \
  OF_PP_MAKE_TUPLE_SEQ(17.64621, 2.12636, 0.01026, 1.00907, Dropout)                   \
  OF_PP_MAKE_TUPLE_SEQ(17.87725, 2.00001, 0.00000, 0.90036, OnesLike)                  \
  OF_PP_MAKE_TUPLE_SEQ(17.87795, 2.26086, 0.01463, 0.98919, NotEqualZero)              \
  // 17.87795, 2.26086, 0.01463, 0.98919 default

#define REGISTER_UNARY_PIECEWISE_LINE_OP_GETCOMPUTECOMPLEXITY(x0, y0, k1, k2, func_prefix)  \
  /* static */ Maybe<double> func_prefix##Op::GetComputeComplexity(                         \
      user_op::ComputeComplexityFnContext* ctx) {                                           \
    GenericPiecewiseLine2D functor{x0, y0, k1, k2};                                         \
    const auto& inputs = ctx->inputs();                                                     \
    CHECK_EQ_OR_RETURN(inputs.size(), 1);                                                   \
    int m = ctx->Shape4ArgNameAndIndex(inputs[0].first, inputs[0].second).elem_cnt();       \
    double cost = functor.GetVal(m);                                                        \
    const auto& outputs = ctx->outputs();                                                   \
    std::cout << "inputs(" << inputs.size() << ") \t:";                                     \
    for (auto& p : inputs) { std::cout << "\t(" << p.first << ", " << p.second << ")\n"; }  \
    std::cout << "outputs(" << outputs.size() << ") \t:";                                   \
    for (auto& p : outputs) { std::cout << "\t(" << p.first << ", " << p.second << ")\n"; } \
    std::cout << ctx->user_op_conf().op_name() << " Op::GetComputeComplexity: "             \
              << "m = " << m << "\tcost = " << cost << std::endl;                           \
    return cost;                                                                            \
  }

OF_PP_FOR_EACH_TUPLE(REGISTER_UNARY_PIECEWISE_LINE_OP_GETCOMPUTECOMPLEXITY,
                     UNARY_PIECEWISE_LINE_OP_SEQ)

// #define BINARY_PIECEWISE_LINE_OP_SEQ                             \
//   OF_PP_MAKE_TUPLE_SEQ(17.09093, 2.24071, 0.02015, 0.97546, Pow) \
//   OF_PP_MAKE_TUPLE_SEQ(17.41145, 2.12395, 0.01026, 1.02292, AddN)

// #define REGISTER_BINARY_PIECEWISE_LINE_OP_GETCOMPUTECOMPLEXITY(x0, y0, k1, k2, func_prefix) \
//   /* static */ Maybe<double> func_prefix##Op::GetComputeComplexity(                         \
//       user_op::ComputeComplexityFnContext* ctx) {                                           \
//     GenericPiecewiseLine2D functor{x0, y0, k1, k2};                                         \
//     const auto& inputs = ctx->inputs();                                                     \
//     CHECK_EQ_OR_RETURN(inputs.size(), 2);                                                   \
//     int m = ctx->Shape4ArgNameAndIndex(inputs[0].first, inputs[0].second).elem_cnt();       \
//     double cost = functor.GetVal(m);                                                        \
//     const auto& outputs = ctx->outputs();                                                   \
//     std::cout << "inputs(" << inputs.size() << ") \t:";                                     \
//     for (auto& p : inputs) { std::cout << "\t(" << p.first << ", " << p.second << ")\n"; }  \
//     std::cout << "outputs(" << outputs.size() << ") \t:";                                   \
//     for (auto& p : outputs) { std::cout << "\t(" << p.first << ", " << p.second << ")\n"; } \
//     std::cout << ctx->user_op_conf().op_name() << " Op::GetComputeComplexity: "             \
//               << "m = " << m << "\tcost = " << cost << std::endl;                           \
//     return cost;                                                                            \
//   }

// OF_PP_FOR_EACH_TUPLE(REGISTER_BINARY_PIECEWISE_LINE_OP_GETCOMPUTECOMPLEXITY,
//                      BINARY_PIECEWISE_LINE_OP_SEQ)

/* static */ Maybe<double> PowOp::GetComputeComplexity(user_op::ComputeComplexityFnContext* ctx) {
  GenericPiecewiseLine2D functor{17.09093, 2.24071, 0.02015, 0.97546};
  const auto& inputs = ctx->inputs();
  CHECK_EQ_OR_RETURN(inputs.size(), 2);
  int m = ctx->Shape4ArgNameAndIndex(inputs[0].first, inputs[0].second).elem_cnt();
  double cost = functor.GetVal(m);
  const auto& outputs = ctx->outputs();
  std::cout << "inputs(" << inputs.size() << ") \t:";
  for (auto& p : inputs) { std::cout << "\t(" << p.first << ", " << p.second << ")\n"; }
  std::cout << "outputs(" << outputs.size() << ") \t:";
  for (auto& p : outputs) { std::cout << "\t(" << p.first << ", " << p.second << ")\n"; }
  std::cout << ctx->user_op_conf().op_name() << " Op::GetComputeComplexity: "
            << "m = " << m << "\tcost = " << cost << std::endl;
  return cost;
}

/* static */ Maybe<double> AddNOp::GetComputeComplexity(user_op::ComputeComplexityFnContext* ctx) {
  GenericPiecewiseLine2D functor{17.41145, 2.12395, 0.01026, 1.02292};
  const auto& inputs = ctx->inputs();
  // CHECK_EQ_OR_RETURN(inputs.size(), 2);
  // int m = 0;
  // for (const auto& pair : inputs)                                            {
  // int m = ctx->Shape4ArgNameAndIndex(pair.first, pair.second).elem_cnt();

  // }
  int m = ctx->Shape4ArgNameAndIndex(inputs[0].first, inputs[0].second).elem_cnt();
  double cost = functor.GetVal(m);
  const auto& outputs = ctx->outputs();
  std::cout << "inputs(" << inputs.size() << ") \t:";
  for (auto& p : inputs) { std::cout << "\t(" << p.first << ", " << p.second << ")\n"; }
  std::cout << "outputs(" << outputs.size() << ") \t:";
  for (auto& p : outputs) { std::cout << "\t(" << p.first << ", " << p.second << ")\n"; }
  std::cout << ctx->user_op_conf().op_name() << " Op::GetComputeComplexity: "
            << "m = " << m << "\tcost = " << cost << std::endl;
  return cost;
}

Maybe<double> GetComputationCost(user_op::ComputeComplexityFnContext* ctx) {
  // bool transpose_b = ctx->Attr<bool>("transpose_b");
  const Shape& shape_a = ctx->Shape4ArgNameAndIndex("a", 0);
  const Shape& shape_b = ctx->Shape4ArgNameAndIndex("b", 0);

  int batch_a = shape_a.NumAxes() <= 2 ? 1 : shape_a.Count(0, shape_a.NumAxes() - 2);
  int batch_b = shape_b.NumAxes() <= 2 ? 1 : shape_b.Count(0, shape_b.NumAxes() - 2);
  int batch = std::max(batch_a, batch_b);
  int m = shape_a.At(shape_a.NumAxes() - 2);
  int k = shape_a.At(shape_a.NumAxes() - 1);

  int64_t n = 0;
  if (k != shape_b.At(shape_b.NumAxes() - 1)) {
    n = shape_b.At(shape_b.NumAxes() - 1);
  } else {
    n = shape_b.At(shape_b.NumAxes() - 2);
  }

  // double logical_computation_cost = 2 * ctx->Shape4ArgNameAndIndex("a", 0).elem_cnt() * n;
  // const auto& nd_sbp_a = ctx->NdSbp4ArgNameAndIndex("a", 0);
  // const auto& nd_sbp_b = ctx->NdSbp4ArgNameAndIndex("b", 0);
  // const auto& parallel_hierarchy = ctx->parallel_desc().hierarchy();
  // for (int32_t sbp_dim = 0; sbp_dim < nd_sbp_a.sbp_parallel_size(); sbp_dim++) {
  //   if (nd_sbp_a.sbp_parallel(sbp_dim).has_split_parallel()
  //       || nd_sbp_b.sbp_parallel(sbp_dim).has_split_parallel()) {
  //     logical_computation_cost /= parallel_hierarchy->At(sbp_dim);
  //   }
  // }

  // --------------------
  GenericPiecewiseLine2D functor{18.530333209019545, 3.4928505799970475, 0.1248270251999581,
                                 0.710384096884887};
  double cost = functor.GetVal(batch * (m * k + m * n + n * k));
  // --------------------

  return cost;
}

/*static*/ Maybe<double> BroadcastMatmulOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return GetComputationCost(ctx);
}

/*static*/ Maybe<double> MatmulOp::GetComputeComplexity(user_op::ComputeComplexityFnContext* ctx) {
  return GetComputationCost(ctx);
}

/*static*/ Maybe<double> BatchMatmulOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return GetComputationCost(ctx);
}

/*static*/ Maybe<double> BroadcastMatmulGradBOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return GetComputationCost(ctx);
}

Maybe<double> Operator::GetComputeComplexity(
    NdSbpSignature* sbp_signature,
    std::function<const BlobDesc&(const std::string& bn)> logical_blob_desc4bn,
    const ParallelDesc& parallel_desc) const {
  const auto& sbp_bn_in_op2nd_sbp = sbp_signature->bn_in_op2nd_sbp();
  double complexity = 0;
  const auto& parallel_hierarchy = *parallel_desc.hierarchy();

  auto ComputeComplexity4Blobs = [&](const PbRpf<std::string>& bns) -> Maybe<void> {
    for (const auto& bn : bns) {
      const BlobDesc& logical_blob_desc = logical_blob_desc4bn(bn);
      const NdSbp& nd_sbp = sbp_bn_in_op2nd_sbp.at(bn);
      CHECK_EQ_OR_RETURN(nd_sbp.sbp_parallel_size(), parallel_hierarchy.NumAxes())
          << "At this moment, the dimension of nd SBP should be equal to the depth of hierarchy "
             "in "
          << "parallel description.";

      double total_cost = logical_blob_desc.shape().elem_cnt();
      for (int32_t sbp_dim = 0; sbp_dim < nd_sbp.sbp_parallel_size(); sbp_dim++) {
        const auto& sbp = nd_sbp.sbp_parallel(sbp_dim);
        if (sbp.has_split_parallel()) {
          const int64_t axis = sbp.split_parallel().axis();
          if (axis >= logical_blob_desc.shape().NumAxes()
              || logical_blob_desc.shape().At(axis) < parallel_hierarchy.At(sbp_dim)) {
            complexity = GetMaxVal<float>();
            return Maybe<void>::Ok();
          } else {
            total_cost /= parallel_hierarchy.At(sbp_dim);
          }
        }
      }
      complexity += total_cost;
    }
    return Maybe<void>::Ok();
  };
  JUST(ComputeComplexity4Blobs(input_bns()));
  JUST(ComputeComplexity4Blobs(output_bns()));
  return GetDefaultPiecewiseLineVal(complexity);
  // return complexity;
}

}  // namespace oneflow