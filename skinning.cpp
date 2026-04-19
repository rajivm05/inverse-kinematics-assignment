#include "skinning.h"
#include "vec3d.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

Skinning::Skinning(int numMeshVertices, const double * restMeshVertexPositions,
    const std::string & meshSkinningWeightsFilename)
{
  this->numMeshVertices = numMeshVertices;
  this->restMeshVertexPositions = restMeshVertexPositions;

  cout << "Loading skinning weights..." << endl;
  ifstream fin(meshSkinningWeightsFilename.c_str());
  assert(fin);
  int numWeightMatrixRows = 0, numWeightMatrixCols = 0;
  fin >> numWeightMatrixRows >> numWeightMatrixCols;
  assert(fin.fail() == false);
  assert(numWeightMatrixRows == numMeshVertices);
  int numJoints = numWeightMatrixCols;

  vector<vector<int>> weightMatrixColumnIndices(numWeightMatrixRows);
  vector<vector<double>> weightMatrixEntries(numWeightMatrixRows);
  fin >> ws;
  while(fin.eof() == false)
  {
    int rowID = 0, colID = 0;
    double w = 0.0;
    fin >> rowID >> colID >> w;
    weightMatrixColumnIndices[rowID].push_back(colID);
    weightMatrixEntries[rowID].push_back(w);
    assert(fin.fail() == false);
    fin >> ws;
  }
  fin.close();

  // Build skinning joints and weights.
  numJointsInfluencingEachVertex = 0;
  for (int i = 0; i < numMeshVertices; i++)
    numJointsInfluencingEachVertex = std::max(numJointsInfluencingEachVertex, (int)weightMatrixEntries[i].size());
  assert(numJointsInfluencingEachVertex >= 2);

  // Copy skinning weights from SparseMatrix into meshSkinningJoints and meshSkinningWeights.
  meshSkinningJoints.assign(numJointsInfluencingEachVertex * numMeshVertices, 0);
  meshSkinningWeights.assign(numJointsInfluencingEachVertex * numMeshVertices, 0.0);
  for (int vtxID = 0; vtxID < numMeshVertices; vtxID++)
  {
    vector<pair<double, int>> sortBuffer(numJointsInfluencingEachVertex);
    for (size_t j = 0; j < weightMatrixEntries[vtxID].size(); j++)
    {
      int frameID = weightMatrixColumnIndices[vtxID][j];
      double weight = weightMatrixEntries[vtxID][j];
      sortBuffer[j] = make_pair(weight, frameID);
    }
    sortBuffer.resize(weightMatrixEntries[vtxID].size());
    assert(sortBuffer.size() > 0);
    sort(sortBuffer.rbegin(), sortBuffer.rend()); // sort in descending order using reverse_iterators
    for(size_t i = 0; i < sortBuffer.size(); i++)
    {
      meshSkinningJoints[vtxID * numJointsInfluencingEachVertex + i] = sortBuffer[i].second;
      meshSkinningWeights[vtxID * numJointsInfluencingEachVertex + i] = sortBuffer[i].first;
    }

    // Note: When the number of joints used on this vertex is smaller than numJointsInfluencingEachVertex,
    // the remaining empty entries are initialized to zero due to vector::assign(XX, 0.0) .
  }
}

void Skinning::applySkinning(const RigidTransform4d * jointSkinTransforms, double * newMeshVertexPositions) const
{
  if (skinningMode == DUAL_QUATERNION)
    applyDQS(jointSkinTransforms, newMeshVertexPositions);
  else
    applyLBS(jointSkinTransforms, newMeshVertexPositions);
}

void Skinning::applyLBS(const RigidTransform4d * jointSkinTransforms, double * newMeshVertexPositions) const
{
  for (int i = 0; i < numMeshVertices; i++)
  {
    Vec3d rest(restMeshVertexPositions[3 * i + 0],
               restMeshVertexPositions[3 * i + 1],
               restMeshVertexPositions[3 * i + 2]);
    Vec3d newPos(0.0, 0.0, 0.0);
    for (int j = 0; j < numJointsInfluencingEachVertex; j++)
    {
      int idx = i * numJointsInfluencingEachVertex + j;
      double w = meshSkinningWeights[idx];
      if (w == 0.0) continue;
      int jointID = meshSkinningJoints[idx];
      Vec3d transformed = jointSkinTransforms[jointID].transformPoint(rest);
      newPos += w * transformed;
    }
    newMeshVertexPositions[3 * i + 0] = newPos[0];
    newMeshVertexPositions[3 * i + 1] = newPos[1];
    newMeshVertexPositions[3 * i + 2] = newPos[2];
  }
}

// ========== Dual Quaternion Skinning (DQS) ==========
//
// Each joint's skinning transform (R, t) is converted into a unit dual quaternion
//   q_hat = q_r + epsilon * q_d,  where q_d = 0.5 * (0, t) * q_r
// For each vertex we linearly blend the per-joint dual quaternions with the skinning
// weights (after antipodality correction), re-normalize, and apply the resulting
// unit dual quaternion to the rest-pose position. This removes the volume-loss /
// "candy-wrapper" artifacts seen with linear blend skinning at large rotations.
namespace
{

struct Quat
{
  double w, x, y, z;
  Quat() : w(1), x(0), y(0), z(0) {}
  Quat(double w_, double x_, double y_, double z_) : w(w_), x(x_), y(y_), z(z_) {}

  Quat operator+(const Quat & q) const { return Quat(w + q.w, x + q.x, y + q.y, z + q.z); }
  Quat operator*(double s) const { return Quat(w * s, x * s, y * s, z * s); }
  // Hamilton product
  Quat operator*(const Quat & q) const
  {
    return Quat(
      w * q.w - x * q.x - y * q.y - z * q.z,
      w * q.x + x * q.w + y * q.z - z * q.y,
      w * q.y - x * q.z + y * q.w + z * q.x,
      w * q.z + x * q.y - y * q.x + z * q.w);
  }
};

inline double dot4(const Quat & a, const Quat & b)
{ return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z; }

// Build a unit quaternion from a row-major 3x3 rotation matrix (Shepperd's method).
Quat quatFromRotation(const Mat3d & R)
{
  double m00 = R[0][0], m01 = R[0][1], m02 = R[0][2];
  double m10 = R[1][0], m11 = R[1][1], m12 = R[1][2];
  double m20 = R[2][0], m21 = R[2][1], m22 = R[2][2];
  double tr = m00 + m11 + m22;
  Quat q;
  if (tr > 0.0)
  {
    double s = std::sqrt(tr + 1.0) * 2.0; // s = 4 * q.w
    q.w = 0.25 * s;
    q.x = (m21 - m12) / s;
    q.y = (m02 - m20) / s;
    q.z = (m10 - m01) / s;
  }
  else if (m00 > m11 && m00 > m22)
  {
    double s = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
    q.w = (m21 - m12) / s;
    q.x = 0.25 * s;
    q.y = (m01 + m10) / s;
    q.z = (m02 + m20) / s;
  }
  else if (m11 > m22)
  {
    double s = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
    q.w = (m02 - m20) / s;
    q.x = (m01 + m10) / s;
    q.y = 0.25 * s;
    q.z = (m12 + m21) / s;
  }
  else
  {
    double s = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
    q.w = (m10 - m01) / s;
    q.x = (m02 + m20) / s;
    q.y = (m12 + m21) / s;
    q.z = 0.25 * s;
  }
  return q;
}

struct DQ
{
  Quat real, dual;
  DQ() : real(1, 0, 0, 0), dual(0, 0, 0, 0) {}
  DQ(const Quat & r, const Quat & d) : real(r), dual(d) {}
  DQ operator+(const DQ & o) const { return DQ(real + o.real, dual + o.dual); }
  DQ operator*(double s) const { return DQ(real * s, dual * s); }
};

DQ dqFromRigid(const RigidTransform4d & xf)
{
  Quat qr = quatFromRotation(xf.getRotation());
  Vec3d t = xf.getTranslation();
  Quat qt(0.0, t[0], t[1], t[2]);
  Quat qd = qt * qr * 0.5;
  return DQ(qr, qd);
}

// Apply a unit dual quaternion directly to a 3D point.
// Derivation: decode (R, t) from q_hat and compute R*p + t.
// R from unit quaternion (w,x,y,z):
//   [ 1-2(y^2+z^2)  2(xy-wz)     2(xz+wy)    ]
//   [ 2(xy+wz)      1-2(x^2+z^2) 2(yz-wx)    ]
//   [ 2(xz-wy)      2(yz+wx)     1-2(x^2+y^2)]
// t from 2 * q_d * conj(q_r), take vector part.
Vec3d dqTransformPoint(const DQ & q, const Vec3d & p)
{
  const Quat & r = q.real;
  const Quat & d = q.dual;

  // rotate p by r
  double xx = r.x * r.x, yy = r.y * r.y, zz = r.z * r.z;
  double xy = r.x * r.y, xz = r.x * r.z, yz = r.y * r.z;
  double wx = r.w * r.x, wy = r.w * r.y, wz = r.w * r.z;

  Vec3d rp(
    (1 - 2 * (yy + zz)) * p[0] + 2 * (xy - wz)     * p[1] + 2 * (xz + wy)     * p[2],
    2 * (xy + wz)       * p[0] + (1 - 2 * (xx + zz)) * p[1] + 2 * (yz - wx)   * p[2],
    2 * (xz - wy)       * p[0] + 2 * (yz + wx)     * p[1] + (1 - 2 * (xx + yy)) * p[2]);

  // translation t = 2 * (d * conj(r)) -- vector part
  // conj(r) = (r.w, -r.x, -r.y, -r.z)
  // d * conj(r) vector part:
  double tx =  d.w * (-r.x) + d.x * r.w     + d.y * (-r.z) - d.z * (-r.y);
  double ty =  d.w * (-r.y) - d.x * (-r.z) + d.y * r.w     + d.z * (-r.x);
  double tz =  d.w * (-r.z) + d.x * (-r.y) - d.y * (-r.x) + d.z * r.w;
  // the above simplifies to:
  tx = 2.0 * (-d.w * r.x + d.x * r.w - d.y * r.z + d.z * r.y);
  ty = 2.0 * (-d.w * r.y + d.x * r.z + d.y * r.w - d.z * r.x);
  tz = 2.0 * (-d.w * r.z - d.x * r.y + d.y * r.x + d.z * r.w);

  return Vec3d(rp[0] + tx, rp[1] + ty, rp[2] + tz);
}

} // anonymous namespace

void Skinning::applyDQS(const RigidTransform4d * jointSkinTransforms, double * newMeshVertexPositions) const
{
  for (int i = 0; i < numMeshVertices; i++)
  {
    Vec3d rest(restMeshVertexPositions[3 * i + 0],
               restMeshVertexPositions[3 * i + 1],
               restMeshVertexPositions[3 * i + 2]);

    DQ blended;
    blended.real = Quat(0, 0, 0, 0);
    blended.dual = Quat(0, 0, 0, 0);

    // Pivot quaternion used for antipodality correction: the first nonzero-weight joint's real part.
    Quat pivot(0, 0, 0, 0);
    bool havePivot = false;

    for (int j = 0; j < numJointsInfluencingEachVertex; j++)
    {
      int idx = i * numJointsInfluencingEachVertex + j;
      double w = meshSkinningWeights[idx];
      if (w == 0.0) continue;
      int jointID = meshSkinningJoints[idx];
      DQ q = dqFromRigid(jointSkinTransforms[jointID]);

      if (!havePivot)
      {
        pivot = q.real;
        havePivot = true;
      }
      else if (dot4(pivot, q.real) < 0.0)
      {
        w = -w; // flip to go through the shorter arc
      }

      blended.real = blended.real + q.real * w;
      blended.dual = blended.dual + q.dual * w;
    }

    // Normalize by the magnitude of the real part; the dual part scales identically.
    double n2 = dot4(blended.real, blended.real);
    if (n2 > 0.0)
    {
      double inv = 1.0 / std::sqrt(n2);
      blended.real = blended.real * inv;
      blended.dual = blended.dual * inv;
    }
    else
    {
      // Fallback: leave rest position unchanged if weights summed to zero.
      newMeshVertexPositions[3 * i + 0] = rest[0];
      newMeshVertexPositions[3 * i + 1] = rest[1];
      newMeshVertexPositions[3 * i + 2] = rest[2];
      continue;
    }

    Vec3d newPos = dqTransformPoint(blended, rest);
    newMeshVertexPositions[3 * i + 0] = newPos[0];
    newMeshVertexPositions[3 * i + 1] = newPos[1];
    newMeshVertexPositions[3 * i + 2] = newPos[2];
  }
}

