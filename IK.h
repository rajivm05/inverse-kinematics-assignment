#ifndef IK_H
#define IK_H

// Use adol-c to compute the gradient of the forward kinematics (the "Jacobian matrix"),
// then use Tikhonov regularization and the Jacobian matrix to perform IK.

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

#include <cfloat>

class FK;
class Vec3d;
class Skinning;

class IK
{
public:
  // IK constructor for joint handles.
  // numIKJoints, IKJointIDs: the number of IK handle joints, and their indices (using the joint numbering as defined in the FK class).
  // FK: pointer to an already initialized forward kinematics class.
  // adolc_tagID: an ID used in adol-c to represent a particular function for evaluation. Different functions should have different tagIDs.
  IK(int numIKJoints, const int * IKJointIDs, FK * fk, int adolc_tagID = 1);

  // IK constructor for mesh-vertex handles (extra credit).
  // numIKVertices, IKVertexIDs: the number of IK handle vertices and their indices into the mesh.
  // FK: pointer to an initialized FK class.
  // skinning: pointer to an initialized Skinning class. The FK path now goes through linear blend skinning
  //   so the handle positions we solve for are world positions of the skinned mesh vertices.
  // adolc_tagID: ADOL-C function tag; must be different from any other IK tag in the process.
  IK(int numIKVertices, const int * IKVertexIDs, FK * fk, Skinning * skinning, int adolc_tagID = 2);

  // input: an array of numIKJoints Vec3d's giving the positions of the IK handles, current joint Euler angles
  // output: the computed joint Euler angles; same meaning as in the FK class
  // Note: eulerAngles is both input and output
  void doIK(const Vec3d * targetHandlePositions, Vec3d * eulerAngles);

  // IK parameters
  int getFKInputDim() const { return FKInputDim; }
  int getFKOutputDim() const { return FKOutputDim; }
  int getIKInputDim() const { return FKOutputDim; }
  int getIKOutputDim() const { return FKInputDim; }

  // Sub-stepping: if the user yanks a handle further than maxStepDistance, doIK breaks
  // the motion into up to maxSubSteps intermediate IK solves. Larger drags -> more sub-steps.
  void setSubStepping(int maxSteps, double maxDistance)
  { maxSubSteps = maxSteps; maxStepDistance = maxDistance; }
  int getMaxSubSteps() const { return maxSubSteps; }
  double getMaxStepDistance() const { return maxStepDistance; }

  void setSubSteppingEnabled(bool on) { subSteppingEnabled = on; }
  bool isSubSteppingEnabled() const { return subSteppingEnabled; }

protected:
  int numIKJoints = 0;
  const int * IKJointIDs = nullptr;
  FK * fk = nullptr;
  Skinning * skinning = nullptr; // non-null in vertex-handle mode
  bool useVertexHandles = false;
  int adolc_tagID = 0; // tagID
  int FKInputDim = 0; // forward dynamics input dimension
  int FKOutputDim = 0; // forward dynamics output dimension

  // Sub-stepping defaults. Units are the same as the model (world coords).
  // These can be tuned via setSubStepping() at runtime from the driver if needed.
  int maxSubSteps = 20;
  double maxStepDistance = 0.05;
  bool subSteppingEnabled = true;

  void train_adolc();
};

#endif

