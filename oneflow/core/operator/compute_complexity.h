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
#ifndef ONEFLOW_CORE_OPERATOR_COMPUTE_COMPLEXITY_H
#define ONEFLOW_CORE_OPERATOR_COMPUTE_COMPLEXITY_H

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/compute_complexity_fn_context.h"
namespace oneflow {

// z = Ax + By + C
class GenericLine2D {
 public:
  GenericLine2D(){};
  GenericLine2D(double A, double B, double C) : A_(A), B_(B), C_(C){};
  double GetVal(double x, double y) const { return A_ * x + B_ * y + C_; }
  double GetA() const { return A_; }
  double GetB() const { return B_; }
  double GetC() const { return C_; }

 private:
  double A_;
  double B_;
  double C_;
};

// Ax + By + Cz + D = 0
class GenericPlane3D {
 public:
  GenericPlane3D(double A, double B, double C, double D) : A_(A), B_(B), C_(C), D_(D){};
  double GetZ(double x, double y) const { return C_ == 0 ? 0 : (A_ * x + B_ * y + D_) / -C_; }
  double GetA() const { return A_; }
  double GetB() const { return B_; }
  double GetC() const { return C_; }
  double GetD() const { return D_; }

  GenericLine2D cross(const GenericPlane3D& other) {
    return GenericLine2D{this->GetB() * other.GetC() - this->GetC() * other.GetB(),
                         -(this->GetA() * other.GetC() - this->GetC() * other.GetA()),
                         this->GetA() * other.GetB() - this->GetB() * other.GetA()};
  }

  friend class GenericPiecewise3Plane3D;
  friend class GenericPiecewise2Plane3D;

 private:
  double A_;
  double B_;
  double C_;
  double D_;
};

// z = Ax + By + C
class GenericPiecewiseLine2D {
 public:
  GenericPiecewiseLine2D(double x0, double y0, double k1, double k2)
      : x0_(x0), y0_(y0), k1_(k1), k2_(k2){};
  double GetLog2Val(double x) const {
    double value =
        (x < x0_) * (k1_ * x + y0_ - k1_ * x0_) + (x >= x0_) * (k2_ * x + y0_ - k2_ * x0_);
    return value;
  }
  double GetVal(int x) const { return std::pow(2, GetLog2Val(std::log2(x))); }

 private:
  double x0_;
  double y0_;
  double k1_;
  double k2_;
};

class GenericPiecewise2Plane3D {
 public:
  GenericPiecewise2Plane3D(const GenericPlane3D& plane1, const GenericPlane3D& plane2)
      : plane1_(plane1), plane2_(plane2){};

  double GetVal(int x, int y) const { return std::pow(2, GetLog2Val(std::log2(x), std::log2(y))); }

  double GetLog2Val(double x, double y) const {
    double z = plane1_.GetZ(0, 0);
    double vinp1 = plane1_.GetZ(x, y);
    double vinp2 = plane2_.GetZ(x, y);
    double value = vinp1 * ((vinp2 <= z)) + vinp2 * ((vinp2 > z));
    return value;
  }

 private:
  GenericPlane3D plane1_;
  GenericPlane3D plane2_;
};

class GenericPiecewise3Plane3D {
 public:
  GenericPiecewise3Plane3D(const GenericPlane3D& plane1, const GenericPlane3D& plane2,
                           const GenericPlane3D& plane3)
      : plane1_(plane1), plane2_(plane2), plane3_(plane3) {
    double y1 = (-plane3.A_ * plane2.D_ + plane2.A_ * plane3.D_)
                / (plane3.A_ * plane2.B_ - plane2.A_ * plane3.B_);
    double x1 = (-plane2.D_ - plane2.B_ * y1) / plane2.A_;
    double x2 = x1 + plane2.B_ * plane3.C_ - plane2.C_ * plane3.B_;
    double y2 = y1 - (plane2.A_ * plane3.C_ - plane2.C_ * plane3.A_);
    border_ = {y2 - y1, x1 - x2, -x1 * (y2 - y1) + y1 * (x2 - x1)};
  };

  double GetVal(int x, int y) const { return std::pow(2, GetLog2Val(std::log2(x), std::log2(y))); }

  double GetLog2Val(double x, double y) const {
    double z = plane1_.GetZ(0, 0);
    double vinp1 = plane1_.GetZ(x, y);
    double vinp2 = plane2_.GetZ(x, y);
    double vinp3 = plane3_.GetZ(x, y);
    double vinp4 = border_.GetVal(x, y);
    double value = (vinp1 * ((vinp2 <= z) && (vinp3 <= z)) + vinp2 * ((vinp2 > z) && (vinp4 > 0))
                    + vinp3 * ((vinp3 > z) && (vinp4 < 0)));
    return value;
  }

 private:
  GenericPlane3D plane1_;
  GenericPlane3D plane2_;
  GenericPlane3D plane3_;
  GenericLine2D border_;
};

Maybe<double> GetUnaryOpComputeComplexity(user_op::ComputeComplexityFnContext* ctx);
Maybe<double> GetOpComputeComplexityByName(user_op::ComputeComplexityFnContext* ctx);

}  // namespace oneflow
#endif  // ONEFLOW_CORE_OPERATOR_COMPUTE_COMPLEXITY_H
