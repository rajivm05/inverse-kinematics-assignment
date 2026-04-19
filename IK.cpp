#include "IK.h"
#include "FK.h"
#include "skinning.h"
#include "minivectorTemplate.h"
#include <Eigen/Dense>
#include <adolc/adolc.h>
#include <cassert>
#include <iostream>
#if defined(_WIN32) || defined(WIN32)
  #ifndef _USE_MATH_DEFINES
    #define _USE_MATH_DEFINES
  #endif
#endif
#include <math.h>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

namespace
{

// Converts degrees to radians.
template<typename real>
inline real deg2rad(real deg) { return deg * M_PI / 180.0; }

template<typename real>
Mat3<real> Euler2Rotation(const real angle[3], RotateOrder order)
{
  Mat3<real> RX = Mat3<real>::getElementRotationMatrix(0, deg2rad(angle[0]));
  Mat3<real> RY = Mat3<real>::getElementRotationMatrix(1, deg2rad(angle[1]));
  Mat3<real> RZ = Mat3<real>::getElementRotationMatrix(2, deg2rad(angle[2]));

  switch(order)
  {
    case RotateOrder::XYZ:
      return RZ * RY * RX;
    case RotateOrder::YZX:
      return RX * RZ * RY;
    case RotateOrder::ZXY:
      return RY * RX * RZ;
    case RotateOrder::XZY:
      return RY * RZ * RX;
    case RotateOrder::YXZ:
      return RZ * RX * RY;
    case RotateOrder::ZYX:
      return RX * RY * RZ;
  }
  assert(0);
}

// Performs forward kinematics, using the provided "fk" class.
// This is the function whose Jacobian matrix will be computed using adolc.
// numIKJoints and IKJointIDs specify which joints serve as handles for IK:
//   IKJointIDs is an array of integers of length "numIKJoints"
// Input: numIKJoints, IKJointIDs, fk, eulerAngles (of all joints)
// Output: handlePositions (world-coordinate positions of all the IK joints; length is 3 * numIKJoints)
// Common joint-chain forward pass used by both handle modes.
// Fills globalR/globalT (size = fk.getNumJoints()) from the flat Euler-angle input vector.
template<typename real>
void computeJointGlobalsFromEulers(
    const FK & fk, const std::vector<real> & eulerAngles,
    std::vector<Mat3<real>> & globalR, std::vector<Vec3<real>> & globalT)
{
  int numJoints = fk.getNumJoints();
  std::vector<Mat3<real>> localR(numJoints);
  std::vector<Vec3<real>> localT(numJoints);

  for (int i = 0; i < numJoints; i++)
  {
    real ea[3] = { eulerAngles[3 * i + 0], eulerAngles[3 * i + 1], eulerAngles[3 * i + 2] };
    Mat3<real> Reuler = Euler2Rotation(ea, fk.getJointRotateOrder(i));

    Vec3d orientD = fk.getJointOrient(i);
    real oa[3] = { real(orientD[0]), real(orientD[1]), real(orientD[2]) };
    Mat3<real> Rorient = Euler2Rotation(oa, RotateOrder::XYZ);

    localR[i] = Rorient * Reuler;

    Vec3d tD = fk.getJointRestTranslation(i);
    localT[i] = Vec3<real>(real(tD[0]), real(tD[1]), real(tD[2]));
  }

  for (int k = 0; k < numJoints; k++)
  {
    int jointID = fk.getJointUpdateOrder(k);
    int parentID = fk.getJointParent(jointID);
    if (parentID < 0)
    {
      globalR[jointID] = localR[jointID];
      globalT[jointID] = localT[jointID];
    }
    else
    {
      multiplyAffineTransform4ds(globalR[parentID], globalT[parentID],
                                 localR[jointID], localT[jointID],
                                 globalR[jointID], globalT[jointID]);
    }
  }
}

// Joint-handle FK: output positions are just the global translations of the IK joints.
template<typename real>
void forwardKinematicsFunction(
    int numIKJoints, const int * IKJointIDs, const FK & fk,
    const std::vector<real> & eulerAngles, std::vector<real> & handlePositions)
{
  int numJoints = fk.getNumJoints();
  std::vector<Mat3<real>> globalR(numJoints);
  std::vector<Vec3<real>> globalT(numJoints);
  computeJointGlobalsFromEulers(fk, eulerAngles, globalR, globalT);

  for (int k = 0; k < numIKJoints; k++)
  {
    int jointID = IKJointIDs[k];
    handlePositions[3 * k + 0] = globalT[jointID][0];
    handlePositions[3 * k + 1] = globalT[jointID][1];
    handlePositions[3 * k + 2] = globalT[jointID][2];
  }
}

// Vertex-handle FK: after the joint chain, run linear blend skinning on each handle vertex.
// Skin transform for joint j is M_skin_j = M_global_j * M_invRest_j, so applied to rest point p:
//   p' = globalR_j * (invRestR_j * p + invRestT_j) + globalT_j
// Then handle position = sum_j w_j * p'.
template<typename real>
void forwardKinematicsVertexFunction(
    int numIKVertices, const int * IKVertexIDs, const FK & fk, const Skinning & skinning,
    const std::vector<real> & eulerAngles, std::vector<real> & handlePositions)
{
  int numJoints = fk.getNumJoints();
  std::vector<Mat3<real>> globalR(numJoints);
  std::vector<Vec3<real>> globalT(numJoints);
  computeJointGlobalsFromEulers(fk, eulerAngles, globalR, globalT);

  const int influencesPerVertex = skinning.getNumJointsInfluencingEachVertex();
  const int * skinJoints = skinning.getMeshSkinningJoints();
  const double * skinWeights = skinning.getMeshSkinningWeights();
  const double * restPositions = skinning.getRestMeshVertexPositions();

  for (int k = 0; k < numIKVertices; k++)
  {
    int vtx = IKVertexIDs[k];
    real rx = restPositions[3 * vtx + 0];
    real ry = restPositions[3 * vtx + 1];
    real rz = restPositions[3 * vtx + 2];
    Vec3<real> rest(rx, ry, rz);

    real zero = 0;
    Vec3<real> skinned(zero, zero, zero);
    for (int j = 0; j < influencesPerVertex; j++)
    {
      int idx = vtx * influencesPerVertex + j;
      double w = skinWeights[idx];
      if (w == 0.0) continue;
      int jointID = skinJoints[idx];

      const RigidTransform4d & invRest = fk.getJointInvRestGlobalTransform(jointID);
      Mat3d invRestRd = invRest.getRotation();
      Vec3d invRestTd = invRest.getTranslation();

      // Promote the fixed (double) inv-rest transform to real-typed values.
      real m00 = invRestRd[0][0], m01 = invRestRd[0][1], m02 = invRestRd[0][2];
      real m10 = invRestRd[1][0], m11 = invRestRd[1][1], m12 = invRestRd[1][2];
      real m20 = invRestRd[2][0], m21 = invRestRd[2][1], m22 = invRestRd[2][2];
      Mat3<real> invRestR(m00, m01, m02, m10, m11, m12, m20, m21, m22);

      real tx = invRestTd[0], ty = invRestTd[1], tz = invRestTd[2];
      Vec3<real> invRestT(tx, ty, tz);

      Vec3<real> inLocal = invRestR * rest + invRestT;
      Vec3<real> world = globalR[jointID] * inLocal + globalT[jointID];
      real wr = w;
      skinned += world * wr;
    }

    handlePositions[3 * k + 0] = skinned[0];
    handlePositions[3 * k + 1] = skinned[1];
    handlePositions[3 * k + 2] = skinned[2];
  }
}

} // end anonymous namespaces

IK::IK(int numIKJoints, const int * IKJointIDs, FK * inputFK, int adolc_tagID)
{
  this->numIKJoints = numIKJoints;
  this->IKJointIDs = IKJointIDs;
  this->fk = inputFK;
  this->skinning = nullptr;
  this->useVertexHandles = false;
  this->adolc_tagID = adolc_tagID;

  FKInputDim = fk->getNumJoints() * 3;
  FKOutputDim = numIKJoints * 3;

  train_adolc();
}

IK::IK(int numIKVertices, const int * IKVertexIDs, FK * inputFK, Skinning * inputSkinning, int adolc_tagID)
{
  this->numIKJoints = numIKVertices;   // reuse the same count/array as "number of handles"
  this->IKJointIDs = IKVertexIDs;
  this->fk = inputFK;
  this->skinning = inputSkinning;
  this->useVertexHandles = true;
  this->adolc_tagID = adolc_tagID;

  FKInputDim = fk->getNumJoints() * 3;
  FKOutputDim = numIKVertices * 3;

  train_adolc();
}

void IK::train_adolc()
{
  trace_on(adolc_tagID);

  std::vector<adouble> eulerAngles(FKInputDim);
  for (int i = 0; i < FKInputDim; i++)
    eulerAngles[i] <<= 0.0;

  std::vector<adouble> handlePositions(FKOutputDim);
  if (useVertexHandles)
    forwardKinematicsVertexFunction<adouble>(numIKJoints, IKJointIDs, *fk, *skinning, eulerAngles, handlePositions);
  else
    forwardKinematicsFunction<adouble>(numIKJoints, IKJointIDs, *fk, eulerAngles, handlePositions);

  std::vector<double> output(FKOutputDim);
  for (int i = 0; i < FKOutputDim; i++)
    handlePositions[i] >>= output[i];

  trace_off();
}

void IK::doIK(const Vec3d * targetHandlePositions, Vec3d * jointEulerAngles)
{
  int numJoints = fk->getNumJoints(); // Note that is NOT the same as numIKJoints!

  int n = FKInputDim;   // 3 * numJoints
  int m = FKOutputDim;  // 3 * numIKJoints

  std::vector<double> input(n);
  auto loadEulersIntoInput = [&]()
  {
    for (int i = 0; i < numJoints; i++)
      for (int d = 0; d < 3; d++)
        input[3 * i + d] = jointEulerAngles[i][d];
  };

  // Snapshot starting handle positions so we can linearly interpolate intermediate sub-targets.
  loadEulersIntoInput();
  std::vector<double> startingHandlePositions(m);
  ::function(adolc_tagID, m, n, input.data(), startingHandlePositions.data());

  // Choose the number of sub-steps from the largest per-handle displacement.
  // The Jacobian is only accurate locally, so if the user dragged a handle a long
  // distance we break it up into maxStepDistance-sized chunks.
  double maxDisp = 0.0;
  for (int k = 0; k < numIKJoints; k++)
  {
    double dx = targetHandlePositions[k][0] - startingHandlePositions[3 * k + 0];
    double dy = targetHandlePositions[k][1] - startingHandlePositions[3 * k + 1];
    double dz = targetHandlePositions[k][2] - startingHandlePositions[3 * k + 2];
    double d = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (d > maxDisp) maxDisp = d;
  }
  int numSubSteps = 1;
  if (subSteppingEnabled && maxDisp > maxStepDistance)
    numSubSteps = std::min(maxSubSteps, (int)std::ceil(maxDisp / maxStepDistance));
  if (numSubSteps < 1) numSubSteps = 1;
  if (numSubSteps > 1)
    std::cout << "IK sub-steps: " << numSubSteps << " (max handle drag " << maxDisp << ")" << std::endl;

  std::vector<double> currentHandlePositions(m);
  std::vector<double> jacobianMatrix(m * n);
  std::vector<double *> jacobianMatrixEachRow(m);
  for (int i = 0; i < m; i++)
    jacobianMatrixEachRow[i] = &jacobianMatrix[i * n];

  const double alpha = 0.01;

  for (int step = 1; step <= numSubSteps; step++)
  {
    double t = (double)step / (double)numSubSteps; // 0 < t <= 1

    loadEulersIntoInput();

    // Evaluate current handle positions (they have changed with each sub-step).
    ::function(adolc_tagID, m, n, input.data(), currentHandlePositions.data());

    // Jacobian at the current pose.
    ::jacobian(adolc_tagID, m, n, input.data(), jacobianMatrixEachRow.data());

    Eigen::MatrixXd J(m, n);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        J(i, j) = jacobianMatrix[i * n + j];

    // Sub-target = (1 - t) * starting + t * final. Deltab points from current to sub-target.
    Eigen::VectorXd deltab(m);
    for (int k = 0; k < numIKJoints; k++)
    {
      for (int d = 0; d < 3; d++)
      {
        double subTarget = (1.0 - t) * startingHandlePositions[3 * k + d]
                         + t         * targetHandlePositions[k][d];
        deltab(3 * k + d) = subTarget - currentHandlePositions[3 * k + d];
      }
    }

    // Tikhonov: (J^T J + alpha I) dTheta = J^T deltab.
    Eigen::MatrixXd JT = J.transpose();
    Eigen::MatrixXd A = JT * J + alpha * Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd rhs = JT * deltab;
    Eigen::VectorXd deltaTheta = A.ldlt().solve(rhs);

    for (int i = 0; i < numJoints; i++)
      for (int d = 0; d < 3; d++)
        jointEulerAngles[i][d] += deltaTheta(3 * i + d);
  }
}

